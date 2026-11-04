"""LeRobot trajectory writer with explicit outcome sidecar metadata."""
import json
import os
from pathlib import Path
import numpy as np
from controller import KEYS, vector


class Recorder:
    def __init__(self, root, repo_id, fps, observation, policy_id):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        self.root, self.policy_id = Path(root), policy_id
        features = {
            "observation.state": {"dtype": "float32", "shape": (6,), "names": list(KEYS)},
            "action": {"dtype": "float32", "shape": (6,), "names": list(KEYS)},
            "complementary_info.selected_action": {"dtype": "float32", "shape": (6,), "names": list(KEYS)},
            "complementary_info.policy_action": {"dtype": "float32", "shape": (6,), "names": list(KEYS)},
        }
        for key in ("is_intervention", "is_paused", "policy_action_valid", "elapsed_s"):
            features[f"complementary_info.{key}"] = {"dtype": "float32", "shape": (1,), "names": [key]}
        for camera in ("front", "wrist"):
            features[f"observation.images.{camera}"] = {
                "dtype": "video", "shape": observation[camera].shape, "names": ["height", "width", "channels"]}
        self.dataset = LeRobotDataset.create(repo_id=repo_id, root=self.root, fps=fps, features=features,
                                            robot_type="so101_follower", use_videos=True, image_writer_threads=4)
        self.episode = 0
        self.frames = 0

    def add(self, observation, selected, accepted, policy_action, manual, elapsed, prompt, paused=False):
        self.dataset.add_frame({
            "observation.state": vector(observation), "action": accepted,
            "observation.images.front": observation["front"],
            "observation.images.wrist": observation["wrist"],
            "complementary_info.selected_action": selected,
            "complementary_info.policy_action": np.zeros(6, np.float32) if policy_action is None else policy_action,
            "complementary_info.is_intervention": np.array([manual], np.float32),
            "complementary_info.is_paused": np.array([paused], np.float32),
            "complementary_info.policy_action_valid": np.array([policy_action is not None], np.float32),
            "complementary_info.elapsed_s": np.array([elapsed], np.float32), "task": prompt,
        })
        self.frames += 1

    def finish(self, outcome):
        if not self.frames:
            return
        if outcome not in {"success", "failure", "aborted"}:
            raise ValueError("Explicit outcome required")
        self.dataset.save_episode(parallel_encoding=False)
        with (self.root / "outcomes.jsonl").open("a") as stream:
            stream.write(json.dumps({"episode_index": self.episode, "episode_outcome": outcome,
                                     "collector_policy_id": self.policy_id, "frames": self.frames}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.episode += 1
        self.frames = 0

    def close(self):
        self.dataset.finalize()
