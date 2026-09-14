"""Export converted Raiden episodes to the LeRobot v2.1 dataset format.

Produces a single merged dataset (all selected tasks share one episode index
space) laid out exactly as LeRobot expects::

    <output_dir>/
        data/chunk-000/episode_000000.parquet
        videos/chunk-000/observation.images.cam_high/episode_000000.mp4
        videos/chunk-000/observation.images.cam_left_wrist/episode_000000.mp4
        videos/chunk-000/observation.images.cam_right_wrist/episode_000000.mp4
        meta/info.json
        meta/tasks.jsonl
        meta/episodes.jsonl
        meta/episodes_stats.jsonl

The 14-D state/action vectors are ``[l_arm(6), l_grip(1), r_arm(6), r_grip(1)]``
— the same block layout RoboTwin uses, so downstream configs need no reordering.
Note this is a block-layout match only: Raiden drives an I2RT YAM while RoboTwin
defaults to aloha-agilex, so the values themselves are a different embodiment and
normalization statistics must be recomputed from scratch.

Depth, camera intrinsics/extrinsics and the 26-D end-effector pose vectors are
intentionally dropped — no LeRobot consumer reads them.
"""

import dataclasses
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from raiden.shardify import _load_episode_frames

# ---------------------------------------------------------------------------
# LeRobot format constants
# ---------------------------------------------------------------------------

#: Version of the LeRobot on-disk layout we emit.  v2.1 stores per-episode
#: statistics in ``meta/episodes_stats.jsonl``; v2.0 used a single
#: ``meta/stats.json``.  FastWAM's vendored reader pins v2.1.
CODEBASE_VERSION = "v2.1"

DEFAULT_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
DEFAULT_VIDEO_PATH = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)

STATE_DIM = 14

#: Raiden camera name → LeRobot feature suffix.  The ordering of this mapping is
#: significant: FastWAM's ``concat_multi_camera="robotwin"`` path indexes the
#: decoded videos positionally as (top, left wrist, right wrist).
DEFAULT_CAMERA_MAP: Dict[str, str] = {
    "scene_camera": "cam_high",
    "left_wrist_camera": "cam_left_wrist",
    "right_wrist_camera": "cam_right_wrist",
}

JOINT_NAMES: List[str] = (
    [f"left_joint_{i}" for i in range(1, 7)]
    + ["left_gripper"]
    + [f"right_joint_{i}" for i in range(1, 7)]
    + ["right_gripper"]
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LeRobotExportConfig:
    """Parameters controlling the LeRobot export."""

    # Required
    output_dir: Path

    #: Keep every N-th frame.  1 = raiden's native 30 fps, and that is the right
    #: default: the reference ``yuanty/robotwin2.0-fastwam`` dataset is 50 fps, so
    #: a 33-frame window there spans 0.64 s.  Native 30 fps gives 1.07 s; anything
    #: coarser drifts further from the motion scale the checkpoint was pretrained
    #: on (stride 2 would give 2.13 s).  Raiden's cameras cap at 30 fps, so 30 is
    #: as close as the data can get without interpolating frames.
    stride: int = 1

    #: Frame rate written to ``meta/info.json`` and used for the mp4 and for the
    #: ``timestamp`` column.  Must equal ``30 / stride`` for timestamps to line up
    #: with real time.
    #:
    #: Note this value is bookkeeping, not a knob on the model's temporal scale:
    #: FastWAM builds ``delta_timestamps`` as ``t / fps`` and then recovers frame
    #: offsets with ``round(delta_ts * fps)``, so it cancels out and the model
    #: always sees 33 *consecutive* frames.  What actually matters is the real
    #: motion between them, which is fixed by the capture rate.
    fps: int = 30

    #: Episodes per chunk directory.  LeRobot's default is 1000.
    chunks_size: int = 1000

    #: x264 constant rate factor.  Lower is higher quality / larger files.
    crf: int = 23

    #: Keyframe interval.  Small GOPs keep torchcodec's "approximate" seek mode
    #: accurate.
    gop: int = 15

    camera_map: Dict[str, str] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_CAMERA_MAP)
    )

    #: Number of frames sampled per episode when computing image statistics.
    image_stats_samples: int = 100

    robot_type: str = "yam_bimanual"

    #: Run the structural consistency check (parquet rows == episode length ==
    #: mp4 frame count) after writing each episode.
    verify: bool = True

    #: Overwrite ``output_dir`` if it already exists.
    overwrite: bool = False


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _vector_stats(arr: np.ndarray) -> Dict[str, Any]:
    """Per-dimension statistics for a (T, D) array, in LeRobot's stats schema.

    Args:
        arr: (T, D) float array.

    Returns:
        Dict with ``min``/``max``/``mean``/``std`` as length-D lists and an
        integer ``count``.
    """
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }


def _image_stats(png_paths: Sequence[Path], max_samples: int) -> Dict[str, Any]:
    """Per-channel RGB statistics over a subsample of frames.

    Values are normalized to [0, 1] and shaped (3, 1, 1), matching what LeRobot
    writes for video features.

    Args:
        png_paths: Frame paths for one camera of one episode.
        max_samples: Maximum number of frames to read.

    Returns:
        Stats dict with nested (3, 1, 1) lists.
    """
    n = len(png_paths)
    idx = np.unique(np.linspace(0, n - 1, min(max_samples, n)).astype(int))

    count = 0
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    vmin = np.full(3, np.inf)
    vmax = np.full(3, -np.inf)

    for i in idx:
        bgr = cv2.imread(str(png_paths[i]))
        if bgr is None:
            continue
        # cv2 gives BGR; reverse to RGB to match the channel order implied by
        # the PNG's actual content.
        rgb = bgr[:, :, ::-1].astype(np.float64) / 255.0
        flat = rgb.reshape(-1, 3)
        total += flat.sum(axis=0)
        total_sq += (flat**2).sum(axis=0)
        vmin = np.minimum(vmin, flat.min(axis=0))
        vmax = np.maximum(vmax, flat.max(axis=0))
        count += flat.shape[0]

    if count == 0:
        raise ValueError("No readable frames while computing image statistics")

    mean = total / count
    var = np.maximum(total_sq / count - mean**2, 0.0)
    std = np.sqrt(var)

    def _chw(v: np.ndarray) -> List[List[List[float]]]:
        return [[[float(x)]] for x in v]

    return {
        "min": _chw(vmin),
        "max": _chw(vmax),
        "mean": _chw(mean),
        "std": _chw(std),
        "count": [len(idx)],
    }


# ---------------------------------------------------------------------------
# Video encoding
# ---------------------------------------------------------------------------


def _encode_video(
    png_paths: Sequence[Path],
    out_path: Path,
    fps: int,
    crf: int,
    gop: int,
) -> None:
    """Encode a list of PNG frames into a constant-frame-rate h264 mp4.

    Frames are piped to ffmpeg rather than globbed from disk because the
    subsampled filenames are not consecutive.  Exact CFR matters: LeRobot matches
    video frames to parquet rows with a 1e-4 s tolerance, and torchcodec resolves
    a query as ``round(timestamp * average_fps)``.

    Args:
        png_paths: Ordered frame paths.
        out_path: Destination .mp4.
        fps: Output frame rate.
        crf: x264 quality factor.
        gop: Keyframe interval.

    Raises:
        RuntimeError: If ffmpeg exits non-zero.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-g",
        str(gop),
        # ffmpeg 4.2 predates -fps_mode; -vsync cfr is the equivalent here.
        "-vsync",
        "cfr",
        "-r",
        str(fps),
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None and proc.stderr is not None
    try:
        for p in png_paths:
            proc.stdin.write(p.read_bytes())
    except BrokenPipeError:
        # ffmpeg died early; the error text is waiting on stderr.
        pass
    finally:
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
    # -loglevel error keeps stderr far below the pipe buffer, so a plain read
    # after closing stdin cannot deadlock.
    stderr = proc.stderr.read()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {out_path.name}: {stderr.decode(errors='replace')}"
        )


def _count_video_frames(path: Path) -> int:
    """Return the exact frame count of a video via ffprobe."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip())


# ---------------------------------------------------------------------------
# Episode writing
# ---------------------------------------------------------------------------


def _frame_paths(ep_dir: Path, camera: str, keep: Sequence[int]) -> List[Path]:
    """Resolve the PNG paths for the kept frame indices of one camera.

    Raises:
        FileNotFoundError: If the camera directory or any kept frame is missing.
    """
    cam_dir = ep_dir / "rgb" / camera
    if not cam_dir.is_dir():
        raise FileNotFoundError(f"Missing camera directory {cam_dir}")
    paths = []
    for i in keep:
        p = cam_dir / f"{i:010d}.png"
        if not p.exists():
            raise FileNotFoundError(f"Missing frame {p}")
        paths.append(p)
    return paths


def _write_episode_parquet(
    out_path: Path,
    state: np.ndarray,
    action: np.ndarray,
    episode_index: int,
    task_index: int,
    index_start: int,
    fps: int,
) -> None:
    """Write one episode's low-dimensional data as a LeRobot parquet file."""
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    n = state.shape[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vec = pa.list_(pa.float32(), STATE_DIM)
    table = pa.table(
        {
            "observation.state": pa.FixedSizeListArray.from_arrays(
                pa.array(state.reshape(-1), type=pa.float32()), STATE_DIM
            ),
            "action": pa.FixedSizeListArray.from_arrays(
                pa.array(action.reshape(-1), type=pa.float32()), STATE_DIM
            ),
            "timestamp": pa.array(np.arange(n) / fps, type=pa.float32()),
            "frame_index": pa.array(np.arange(n), type=pa.int64()),
            "episode_index": pa.array(np.full(n, episode_index), type=pa.int64()),
            "index": pa.array(np.arange(index_start, index_start + n), type=pa.int64()),
            "task_index": pa.array(np.full(n, task_index), type=pa.int64()),
        },
        schema=pa.schema(
            [
                pa.field("observation.state", vec),
                pa.field("action", vec),
                pa.field("timestamp", pa.float32()),
                pa.field("frame_index", pa.int64()),
                pa.field("episode_index", pa.int64()),
                pa.field("index", pa.int64()),
                pa.field("task_index", pa.int64()),
            ]
        ),
    )
    pq.write_table(table, out_path)


def _build_features(
    camera_keys: Sequence[str], fps: int, height: int, width: int
) -> Dict[str, Any]:
    """Build the ``features`` block of meta/info.json."""
    features: Dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [STATE_DIM],
            "names": list(JOINT_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": [STATE_DIM],
            "names": list(JOINT_NAMES),
        },
    }
    for key in camera_keys:
        features[f"observation.images.{key}"] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.fps": float(fps),
                "video.height": height,
                "video.width": width,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    for key, dtype in (
        ("timestamp", "float32"),
        ("frame_index", "int64"),
        ("episode_index", "int64"),
        ("index", "int64"),
        ("task_index", "int64"),
    ):
        features[key] = {"dtype": dtype, "shape": [1], "names": None}
    return features


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_lerobot_export(
    task_episodes: List[Tuple[Path, List[Path]]],
    config: LeRobotExportConfig,
) -> None:
    """Export converted Raiden episodes to a single merged LeRobot v2.1 dataset.

    Args:
        task_episodes: ``(task_dir, episode_dirs)`` pairs as returned by
            :func:`raiden.shardify.select_processed_task`.  All tasks are merged
            into one dataset with a shared episode index space.
        config: Export parameters.

    Raises:
        FileNotFoundError: If a required camera stream or frame is missing.
        RuntimeError: If ffmpeg fails or a written episode fails verification.
    """
    t_start = time.time()
    out = Path(config.output_dir)
    if out.exists():
        if not config.overwrite:
            raise FileExistsError(
                f"{out} already exists; pass --overwrite to replace it"
            )
        shutil.rmtree(out)
    out.mkdir(parents=True)

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")

    cam_items = list(config.camera_map.items())
    camera_keys = [v for _, v in cam_items]

    tasks: List[str] = []
    task_to_index: Dict[str, int] = {}
    episodes_rows: List[Dict[str, Any]] = []
    stats_rows: List[Dict[str, Any]] = []

    episode_index = 0
    index_start = 0
    total_frames = 0
    height = width = None

    all_eps = [(td, ep) for td, eps in task_episodes for ep in eps]
    print(f"Exporting {len(all_eps)} episode(s) to {out}")

    for task_dir, ep_dir in tqdm(all_eps, unit="episode", dynamic_ncols=True):
        with open(ep_dir / "metadata.json") as f:
            ep_meta = json.load(f)

        prompts = ep_meta.get("language", {}).get("prompt") or [task_dir.name]
        task_str = prompts[0]
        if task_str not in task_to_index:
            task_to_index[task_str] = len(tasks)
            tasks.append(task_str)
        task_index = task_to_index[task_str]

        frames = _load_episode_frames(ep_dir)
        keep = list(range(0, len(frames), config.stride))
        n = len(keep)
        if n == 0:
            print(f"  SKIP {task_dir.name}/{ep_dir.name}: no frames after subsampling")
            continue

        state = np.stack([frames[i]["joints"] for i in keep]).astype(np.float32)
        action = np.stack([frames[i]["action_joints"] for i in keep]).astype(np.float32)
        if state.shape[1] != STATE_DIM or action.shape[1] != STATE_DIM:
            raise ValueError(
                f"{ep_dir}: expected {STATE_DIM}-D joints, got "
                f"state {state.shape} action {action.shape}"
            )

        chunk = episode_index // config.chunks_size
        parquet_path = out / DEFAULT_DATA_PATH.format(
            episode_chunk=chunk, episode_index=episode_index
        )
        _write_episode_parquet(
            parquet_path,
            state,
            action,
            episode_index=episode_index,
            task_index=task_index,
            index_start=index_start,
            fps=config.fps,
        )

        ep_stats: Dict[str, Any] = {
            "observation.state": _vector_stats(state),
            "action": _vector_stats(action),
        }

        for src_cam, key in cam_items:
            png_paths = _frame_paths(ep_dir, src_cam, keep)
            if height is None:
                probe = cv2.imread(str(png_paths[0]))
                height, width = int(probe.shape[0]), int(probe.shape[1])
            video_path = out / DEFAULT_VIDEO_PATH.format(
                episode_chunk=chunk,
                video_key=f"observation.images.{key}",
                episode_index=episode_index,
            )
            _encode_video(png_paths, video_path, config.fps, config.crf, config.gop)
            ep_stats[f"observation.images.{key}"] = _image_stats(
                png_paths, config.image_stats_samples
            )

            if config.verify:
                got = _count_video_frames(video_path)
                if got != n:
                    raise RuntimeError(
                        f"{video_path.name}: encoded {got} frames but the episode "
                        f"has {n}; the mp4 is not exact CFR"
                    )

        episodes_rows.append(
            {"episode_index": episode_index, "tasks": [task_str], "length": n}
        )
        stats_rows.append({"episode_index": episode_index, "stats": ep_stats})

        episode_index += 1
        index_start += n
        total_frames += n

    if episode_index == 0:
        raise RuntimeError("No episodes were exported")

    meta = out / "meta"
    _write_jsonl(meta / "episodes.jsonl", episodes_rows)
    _write_jsonl(meta / "episodes_stats.jsonl", stats_rows)
    _write_jsonl(
        meta / "tasks.jsonl",
        [{"task_index": i, "task": t} for i, t in enumerate(tasks)],
    )

    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": config.robot_type,
        "total_episodes": episode_index,
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": episode_index * len(camera_keys),
        "total_chunks": (episode_index - 1) // config.chunks_size + 1,
        "chunks_size": config.chunks_size,
        "fps": config.fps,
        "splits": {"train": f"0:{episode_index}"},
        "data_path": DEFAULT_DATA_PATH,
        "video_path": DEFAULT_VIDEO_PATH,
        "features": _build_features(camera_keys, config.fps, height, width),
    }
    with open(meta / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    elapsed = time.time() - t_start
    print(
        f"\nWrote {episode_index} episode(s), {total_frames} frames, "
        f"{len(tasks)} task(s) to {out} in {elapsed:.1f}s"
    )
    print(f"  cameras: {', '.join(camera_keys)}")
    print(f"  fps: {config.fps} (stride {config.stride})")
    for i, t in enumerate(tasks):
        print(f"  task {i}: {t}")
