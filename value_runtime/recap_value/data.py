"""LeRobot v2.1 reader independent of LeRobot package version."""
import json
from pathlib import Path
import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageOps
from .observation import Observation


class ValueDataset:
    def __init__(self, root, tokenizer_path, *, split="train", max_token_len=96):
        from gemma.gm.text._tokenizer import Gemma3Tokenizer
        self.root = Path(root)
        self.info = json.loads((self.root / "meta/info.json").read_text())
        from openpi.recap.data_manifest import load_manifest
        self.manifest = load_manifest(self.root)
        if self.manifest.get("label_recipe") != "paper-v1":
            raise ValueError("Convert with paper-v1 before value training")
        self.tokenizer = Gemma3Tokenizer(path=str(tokenizer_path))
        self.max_token_len = max_token_len
        self.tasks = {row["task_index"]: row["task"] for row in map(json.loads, (self.root / "meta/tasks.jsonl").read_text().splitlines())}
        allowed = set(self.manifest["splits"][split])
        rows = [row for file in sorted((self.root / "data").rglob("*.parquet"))
                for row in pq.read_table(file).to_pylist()]
        from openpi.recap.targets import rewards_and_returns
        if self.info["fps"] != self.manifest["fps"] or len(rows) != self.info["total_frames"]:
            raise ValueError("Dataset frame count/FPS differs from manifest")
        grouped = {}
        for row in rows:
            grouped.setdefault(row["episode_index"], []).append(row)
        if set(grouped) != {ep["episode_index"] for ep in self.manifest["episodes"]}:
            raise ValueError("Dataset episode IDs differ from manifest")
        for episode in self.manifest["episodes"]:
            frames = grouped[episode["episode_index"]]
            if len(frames) != episode["length"] or [r["frame_index"] for r in frames] != list(range(len(frames))):
                raise ValueError("Episode frame order/length differs from manifest")
            if {self.tasks[r["task_index"]] for r in frames} != {episode["task"]}:
                raise ValueError("Episode task differs from manifest")
            if sum(r["intervention"] for r in frames) != episode["intervention_frames"]:
                raise ValueError("Episode intervention count differs from manifest")
            expected = rewards_and_returns(len(frames), episode["outcome"], self.manifest["task_scales"][episode["task"]])
            for key, target in zip(("reward", "reward_label", "value_label"), expected, strict=True):
                if not np.allclose([r[key] for r in frames], target, atol=1e-6, rtol=1e-6):
                    raise ValueError(f"Episode {key} differs from fixed reward definition")
        self.rows = [row for row in rows if row["episode_index"] in allowed]
        if not self.rows:
            raise ValueError(f"No episodes in {split}; collect more data or set explicit validation split")

    def _image(self, episode, camera, index):
        path = self.root / self.info["video_path"].format(episode_chunk=episode // self.info["chunks_size"], episode_index=episode, video_key=camera)
        with av.open(str(path)) as reader:
            stream = reader.streams.video[0]
            seconds = index / self.info["fps"]
            reader.seek(int(seconds / stream.time_base), stream=stream, backward=True)
            for frame in reader.decode(stream):
                if frame.time + 0.5 / self.info["fps"] >= seconds:
                    image = ImageOps.pad(frame.to_image(), (224, 224), method=Image.Resampling.BILINEAR)
                    return np.asarray(image, np.float32) / 127.5 - 1
        raise ValueError(f"Missing video frame {episode}/{camera}/{index}")

    def batch(self, indices):
        rows = [self.rows[int(i)] for i in indices]
        images = {key: np.stack([self._image(row["episode_index"], camera, row["frame_index"]) for row in rows])
                  for camera, key in (("observation.images.front", "base_0_rgb"), ("observation.images.wrist", "left_wrist_0_rgb"))}
        tokens, masks = [], []
        for row in rows:
            state = " ".join(f"{float(value):.3g}" for value in row["observation.state"])
            text = self.tasks[row["task_index"]] + f"\nState: {state}\nValue:"
            encoded = self.tokenizer.encode(text, add_bos=True, add_eos=False)
            if len(encoded) > self.max_token_len:
                raise ValueError("Value prompt exceeds max_token_len")
            tokens.append(encoded + [0] * (self.max_token_len - len(encoded)))
            masks.append([True] * len(encoded) + [False] * (self.max_token_len - len(encoded)))
        obs = Observation(images=images, image_masks={key: np.ones(len(rows), bool) for key in images},
                          state=np.stack([row["observation.state"] for row in rows]),
                          tokenized_prompt=np.array(tokens, np.int32), tokenized_prompt_mask=np.array(masks, bool))
        return obs, np.array([row["value_label"] for row in rows], np.float32)
