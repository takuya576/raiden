# Exporting to LeRobot

The `rd export_lerobot` command converts converted Raiden episodes into a
[LeRobot](https://github.com/huggingface/lerobot) v2.1 dataset — Parquet files
for the low-dimensional data, one mp4 per camera, and a `meta/` directory. This
is the format consumed by LeRobot itself and by downstream trainers such as
FastWAM.

## Usage

```bash
rd export_lerobot
```

Running the command opens an interactive fzf selector. Tasks are listed newest
first. Use Tab to toggle individual tasks, Enter to confirm, or select
`*** ALL TASKS ***` to export everything. Unlike `rd shardify`, all selected
tasks are merged into a **single** dataset sharing one episode index space, with
one entry per distinct language instruction in `meta/tasks.jsonl`.

By default the export reads from `./data/processed/` and writes to
`./data/lerobot/yam_bimanual/`:

```bash
rd export_lerobot --data-dir /mnt/storage/robot_data --output-dir /mnt/storage/lerobot_ds
```

Re-running against an existing directory is refused unless you pass
`--overwrite`.

## What it produces

```
data/lerobot/yam_bimanual/
    data/chunk-000/
        episode_000000.parquet
        episode_000001.parquet
        ...
    videos/chunk-000/
        observation.images.cam_high/episode_000000.mp4
        observation.images.cam_left_wrist/episode_000000.mp4
        observation.images.cam_right_wrist/episode_000000.mp4
        ...
    meta/
        info.json
        tasks.jsonl
        episodes.jsonl
        episodes_stats.jsonl
```

Each Parquet file holds one episode with the standard LeRobot columns:

| column | dtype | contents |
| --- | --- | --- |
| `observation.state` | `float32[14]` | measured joint positions (`joints`) |
| `action` | `float32[14]` | commanded joint positions (`action_joints`) |
| `timestamp` | `float32` | `frame_index / fps` |
| `frame_index` | `int64` | index within the episode |
| `episode_index` | `int64` | index within the dataset |
| `index` | `int64` | global frame counter across all episodes |
| `task_index` | `int64` | row in `meta/tasks.jsonl` |

Both 14-D vectors use the layout
`[left arm ×6, left gripper, right arm ×6, right gripper]`, with grippers
normalized to `[0, 1]` where 1 is open.

Cameras are renamed to the keys most bimanual LeRobot datasets use:

| Raiden camera | LeRobot feature |
| --- | --- |
| `scene_camera` | `observation.images.cam_high` |
| `left_wrist_camera` | `observation.images.cam_left_wrist` |
| `right_wrist_camera` | `observation.images.cam_right_wrist` |

Depth, camera intrinsics/extrinsics and the 26-D end-effector pose vectors are
not exported — no LeRobot consumer reads them. Use `rd shardify` if you need
those.

## Frame rate

The export keeps every frame by default (`--stride 1 --fps 30`), i.e. raiden's
native rate. Keep it that way unless you have a specific reason not to.

The reason is the pretrained checkpoint's temporal scale. The reference
[`yuanty/robotwin2.0-fastwam`](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam)
dataset is **50 fps**, so FastWAM's 33-frame observation window spans 0.64 s
there. At raiden's native 30 fps the same window is 1.07 s; subsampling to 15 fps
would stretch it to 2.13 s and make the per-frame motion deltas roughly 3.3×
larger than the checkpoint expects. Since raiden's cameras cap at 30 fps, native
is as close as the data can get without interpolating frames.

Note that the `fps` number itself is only bookkeeping. FastWAM builds its
`delta_timestamps` as `t / fps` and recovers frame offsets with
`round(delta_ts * fps)`, so the value cancels out and the model always receives
33 *consecutive* frames. What matters is the real motion between them, which is
fixed by the capture rate — which is why subsampling, not relabelling, is what
changes the temporal scale.

If you do subsample, set `--fps` to `30 / stride`, or the `timestamp` column will
not correspond to real time.

## Video encoding

Videos are h264, `yuv420p`, encoded with ffmpeg at exact constant frame rate.
This matters: LeRobot matches video frames to Parquet rows within a 1e-4 s
tolerance, and torchcodec resolves a query as `round(timestamp * average_fps)`.
A variable-frame-rate mp4 will load but silently return the wrong frames.

`--verify` (on by default) checks after each episode that the Parquet row count,
the `meta/episodes.jsonl` length, and the actual mp4 frame count all agree.

## Camera orientation

The export copies frames straight out of `data/processed/`, so it inherits
whatever `_FLIP_CAMERAS` setting was in effect when each episode was converted.
Episodes converted under different settings will disagree with each other. If
you change `_FLIP_CAMERAS` in `raiden/converter.py`, re-run `rd convert
--reconvert` on any task converted before the change, and keep the matching set
in `raiden/server.py` in sync so inference sees the same orientation as training.
