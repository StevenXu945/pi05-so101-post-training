"""Convert the local SO-101 LeRobot v3 recording to the v2.1 layout used by openpi.

Run from the repository root with ``uv run scripts/convert_so101_v3_to_v21.py``.
The original dataset is read-only. Set HF_LEROBOT_HOME before conversion and training
if the converted copy should live outside the default Hugging Face cache.
"""

import json
from pathlib import Path

import av
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pyarrow.parquet as pq
import tyro

SOURCE = Path("test_20261013_101637")
REPO_ID = "local/so101_test_20261013_v21"
CAMERAS = ("observation.images.front", "observation.images.wrist")


class VideoFrames:
    """Decode packed v3 video files sequentially, including file transitions."""

    def __init__(self, root: Path, key: str):
        self.root = root
        self.key = key
        self.file_index = None
        self.container = None
        self.frames = None
        self.position = 0

    def read(self, file_index: int, expected_start: int) -> np.ndarray:
        if self.file_index != file_index:
            self.close()
            path = self.root / "videos" / self.key / "chunk-000" / f"file-{file_index:03d}.mp4"
            self.container = av.open(str(path))
            self.frames = self.container.decode(video=0)
            self.file_index = file_index
            self.position = 0
        if self.position != expected_start:
            raise ValueError(f"{self.key}: expected video frame {expected_start}, next frame is {self.position}")
        try:
            frame = next(self.frames)
        except StopIteration as exc:
            raise ValueError(f"{self.key}: video ended at frame {self.position}") from exc
        self.position += 1
        return frame.to_ndarray(format="rgb24")

    def close(self) -> None:
        if self.container is not None:
            self.container.close()
            self.container = None


def main(
    source: Path = SOURCE,
    repo_id: str = REPO_ID,
    max_episodes: int | None = None,
) -> None:
    source = source.resolve()
    with (source / "meta/info.json").open() as f:
        info = json.load(f)
    if info["codebase_version"] != "v3.0":
        raise ValueError(f"Expected LeRobot v3.0, got {info['codebase_version']}")

    destination = HF_LEROBOT_HOME / repo_id
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")

    episodes = pq.read_table(source / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    tasks = {r["task_index"]: r["task"] for r in pq.read_table(source / "meta/tasks.parquet").to_pylist()}
    data_files = sorted((source / "data").rglob("*.parquet"))
    rows = []
    for path in data_files:
        rows.extend(
            pq.read_table(path, columns=["action", "observation.state", "episode_index", "frame_index", "task_index"])
            .to_pylist()
        )
    if len(rows) != info["total_frames"] or len(episodes) > info["total_episodes"]:
        raise ValueError("Metadata and data frame counts disagree")

    features = {
        key: {
            "dtype": "video",
            "shape": tuple(info["features"][key]["shape"]),
            "names": ["height", "width", "channels"],
        }
        for key in CAMERAS
    }
    for key in ("action", "observation.state"):
        features[key] = {
            "dtype": "float32",
            "shape": (6,),
            "names": info["features"][key]["names"],
        }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=destination,
        robot_type=info["robot_type"],
        fps=info["fps"],
        features=features,
        use_videos=True,
        image_writer_threads=4,
    )
    videos = {key: VideoFrames(source, key) for key in CAMERAS}
    converted_frames = 0
    try:
        for episode in episodes:
            start = episode["dataset_from_index"]
            end = episode["dataset_to_index"]
            length = episode["length"]
            if end - start != length:
                raise ValueError(f"Episode {episode['episode_index']} length mismatch")
            for offset, row in enumerate(rows[start:end]):
                if row["episode_index"] != episode["episode_index"] or row["frame_index"] != offset:
                    raise ValueError(f"Episode {episode['episode_index']} has misaligned Parquet rows")
                frame = {
                    "action": np.asarray(row["action"], dtype=np.float32),
                    "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
                    "task": tasks[row["task_index"]],
                }
                for key in CAMERAS:
                    file_index = episode[f"videos/{key}/file_index"]
                    first_frame = round(episode[f"videos/{key}/from_timestamp"] * info["fps"])
                    frame[key] = videos[key].read(file_index, first_frame + offset)
                dataset.add_frame(frame)
            dataset.save_episode()
            converted_frames += length
            print(f"Converted episode {episode['episode_index'] + 1}/{len(episodes)} ({converted_frames} frames)")
    finally:
        for video in videos.values():
            video.close()

    if max_episodes is None and (converted_frames != info["total_frames"] or len(episodes) != info["total_episodes"]):
        raise ValueError("Converted dataset is incomplete")
    result = LeRobotDataset(repo_id, root=destination, download_videos=False)
    if len(result) != converted_frames:
        raise ValueError(f"Converted dataset has {len(result)} instead of {converted_frames} frames")
    print(f"Verified {len(result)} frames at {destination}")


if __name__ == "__main__":
    tyro.cli(main)
