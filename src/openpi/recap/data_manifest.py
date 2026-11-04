"""Versioned episode provenance, fixed task scales, and training split rules.

This module has no ML dependencies and is shared by collection and openpi runtimes.
"""
import hashlib
import json
from pathlib import Path
import math

MANIFEST_PATH = "meta/recap_manifest.json"


def validate_scales(scales):
    if not isinstance(scales, dict) or not scales:
        raise ValueError("Task scales must be a nonempty mapping keyed by task text")
    for task, scale in scales.items():
        if not isinstance(task, str) or not task:
            raise ValueError("Task names must be nonempty strings")
        n, cost, denominator = scale["max_length"], scale["failure_cost"], scale["scale"]
        if isinstance(n, bool) or int(n) != n or n < 1:
            raise ValueError("max_length must be a positive integer")
        if not all(math.isfinite(float(x)) and x > 0 for x in (cost, denominator)):
            raise ValueError("Task scales must be positive and finite")
        if not math.isclose(denominator, max(1.0, n - 1 + cost), rel_tol=1e-9):
            raise ValueError("Scale must equal max(1, max_length - 1 + failure_cost)")
    return scales


def load_scales(path):
    data = json.loads(Path(path).read_text())
    return validate_scales(data.get("task_scales", data))


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def episode_digest(rows, task, video_paths):
    # IDs, file locations, reward/advantage labels and provenance do not affect identity.
    data = {"task": task, "frames": [
        {key: row[key] for key in ("action", "observation.state", "intervention", "timestamp")}
        for row in rows], "videos": [file_digest(path) for path in video_paths]}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), default=lambda x: x.item()).encode()).hexdigest()


def validate_manifest(manifest):
    if manifest.get("schema_version") != 1 or manifest.get("label_recipe") != "paper-v1":
        raise ValueError("Expected RECAP manifest schema 1 / paper-v1; reconvert old data")
    validate_scales(manifest["task_scales"])
    if not math.isfinite(float(manifest["fps"])) or manifest["fps"] <= 0:
        raise ValueError("Manifest FPS must be positive and finite")
    if not manifest.get("episodes"):
        raise ValueError("Empty RECAP manifest")
    ids, uids, digests = set(), set(), set()
    for episode in manifest["episodes"]:
        index = episode["episode_index"]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in ids:
            raise ValueError("Invalid/duplicate episode index")
        ids.add(index)
        if episode["split"] not in {"train", "validation"}:
            raise ValueError("Unknown episode split")
        if episode["source_kind"] not in {"demonstration", "rollout"}:
            raise ValueError("Unknown source kind")
        if not isinstance(episode["round_id"], int) or episode["round_id"] < 0:
            raise ValueError("round_id must be a nonnegative integer")
        if episode["source_kind"] == "demonstration" and episode["round_id"] != 0:
            raise ValueError("Demonstrations must use round 0")
        if episode["source_kind"] == "rollout" and episode["round_id"] < 1:
            raise ValueError("Rollouts must use round >= 1")
        for key, seen in (("episode_uid", uids), ("content_sha256", digests)):
            if not episode.get(key) or episode[key] in seen:
                raise ValueError(f"Missing/duplicate {key}")
            seen.add(episode[key])
        if not episode.get("policy_id") or not episode.get("source_dataset_id"):
            raise ValueError("Missing policy/source identity")
        if episode["task"] not in manifest["task_scales"] or episode["outcome"] not in {"success", "failure"}:
            raise ValueError("Unknown task/outcome")
        if not 0 <= episode["intervention_frames"] <= episode["length"] <= manifest["task_scales"][episode["task"]]["max_length"]:
            raise ValueError("Invalid episode length/intervention count")
        if episode["length"] < 1:
            raise ValueError("Empty episode")
    if ids != set(range(len(ids))):
        raise ValueError("Manifest episode IDs must be contiguous")
    expected = {split: sorted(ep["episode_index"] for ep in manifest["episodes"] if ep["split"] == split)
                for split in ("train", "validation")}
    if manifest.get("splits") != expected:
        raise ValueError("Manifest splits disagree with episode records")
    return manifest


def dataset_version(manifest):
    data = {key: value for key, value in manifest.items() if key != "dataset_version"}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_manifest(root, scales, episodes, fps):
    manifest = {"schema_version": 1, "label_recipe": "paper-v1", "fps": fps,
                "task_scales": scales, "episodes": episodes,
                "splits": {s: sorted(ep["episode_index"] for ep in episodes if ep["split"] == s)
                           for s in ("train", "validation")}}
    validate_manifest(manifest)
    manifest["dataset_version"] = dataset_version(manifest)
    (Path(root) / MANIFEST_PATH).write_text(json.dumps(manifest, indent=2))
    return manifest


def load_manifest(root):
    manifest = validate_manifest(json.loads((Path(root) / MANIFEST_PATH).read_text()))
    if manifest.get("dataset_version") != dataset_version(manifest):
        raise ValueError("Manifest version mismatch; regenerate through conversion/merge")
    return manifest


def split_episodes(root, split):
    if split not in {"train", "validation"}:
        raise ValueError("Unknown split")
    ids = load_manifest(root)["splits"][split]
    if not ids:
        raise ValueError(f"No episodes in {split}")
    return ids


def threshold_mask(rows, manifest, round_id):
    """Only entire, intervention-free training rollouts in the requested round."""
    import numpy as np
    if round_id < 1:
        raise ValueError("Threshold round must be >= 1")
    eligible = {ep["episode_index"] for ep in manifest["episodes"]
                if ep["split"] == "train" and ep["source_kind"] == "rollout"
                and ep["round_id"] == round_id and ep["intervention_frames"] == 0}
    return np.array([row["episode_index"] in eligible for row in rows], dtype=bool)


def record_policy_run(config, root):
    """Audit a RECAP training run; called only after the trainer creates its output."""
    import dataclasses
    manifest = load_manifest(root)
    labeling = json.loads((Path(root) / "meta/advantage_labeling.json").read_text())
    if labeling.get("dataset_version") != manifest["dataset_version"]:
        raise ValueError("Advantage labels do not match dataset version")
    if config.resume:
        raise ValueError("RECAP rounds use new experiment names and fixed initialization, not --resume")
    loader = config.weight_loader
    if not loader.frozen_seed_params_path or Path(loader.adapter_params_path).resolve() != Path(loader.frozen_seed_params_path).resolve():
        raise ValueError("RECAP round must initialize from the fixed SFT seed; adapter and frozen seed must match")
    artifact = {"dataset_version": manifest["dataset_version"], "data_root": str(Path(root).resolve()),
                "task_scales": manifest["task_scales"], "splits": manifest["splits"],
                "advantage_labeling": labeling, "initialization": dataclasses.asdict(loader),
                "initialization_mode": "fixed SFT seed (local substitute for paper RL pretrained policy)"}
    path = config.checkpoint_dir / "recap_run.json"
    if path.parent.exists():
        if path.exists() and json.loads(path.read_text()) != artifact:
            raise ValueError("RECAP run provenance changed; use a new experiment")
        path.write_text(json.dumps(artifact, indent=2))
