"""Prepare v2.1 demonstrations or merge immutable, versioned RECAP datasets."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from dataset_tools import CAMERAS, labels
from convert_to_recap import convert_video, statistics, write_lines
from openpi.recap.data_manifest import episode_digest, file_digest, load_manifest, load_scales, write_manifest


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def data_path(root, info, index):
    return root / info["data_path"].format(episode_chunk=index // info["chunks_size"], episode_index=index)


def video_paths(root, info, index):
    return [root / info["video_path"].format(episode_chunk=index // info["chunks_size"], episode_index=index, video_key=camera)
            for camera in CAMERAS]


def check_rows(rows, index, length):
    if len(rows) != length or not rows:
        raise ValueError("Episode length mismatch")
    for i, row in enumerate(rows):
        if row["episode_index"] != index or row["frame_index"] != i:
            raise ValueError("Noncontiguous episode frames")
        for key in ("action", "observation.state"):
            array = np.asarray(row[key])
            if array.shape != (6,) or not np.isfinite(array).all():
                raise ValueError(f"Invalid {key}")
        if row.get("intervention", 0) not in (0, 1):
            raise ValueError("Invalid intervention")


def metadata(root, info, episodes, stats, tasks, frames):
    info = copy.deepcopy(info)
    n = len(episodes)
    info.update(total_episodes=n, total_frames=frames, total_tasks=len(tasks), total_videos=n * len(CAMERAS),
                total_chunks=(n + 999) // 1000, chunks_size=1000, splits={"train": f"0:{n}"},
                data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    (root / "meta").mkdir(exist_ok=True)
    (root / "meta/info.json").write_text(json.dumps(info, indent=2))
    write_lines(root / "meta/episodes.jsonl", episodes)
    write_lines(root / "meta/episodes_stats.jsonl", stats)
    write_lines(root / "meta/tasks.jsonl", [{"task_index": i, "task": text} for text, i in tasks.items()])


def write_episode(root, info, rows, index, videos):
    path = root / f"data/chunk-{index // 1000:03d}/episode_{index:06d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    for camera, source in zip(CAMERAS, videos, strict=True):
        target = root / f"videos/chunk-{index // 1000:03d}/{camera}/episode_{index:06d}.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def new_output(sources, destination):
    sources, destination = [Path(p).resolve() for p in sources], Path(destination).resolve()
    if destination.exists() or any(p == destination or p in destination.parents or destination in p.parents for p in sources):
        raise ValueError("Use a new destination outside all sources")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".partial-", dir=destination.parent))
    return sources, destination, temporary


def merge(sources, destination):
    sources, destination, temporary = new_output(sources, destination)
    episodes, records, stats, tasks, seen_uids, seen_content = [], [], [], {}, set(), set()
    total, first_info, first_manifest = 0, None, None
    try:
        for source in sources:
            manifest = load_manifest(source)
            info = json.loads((source / "meta/info.json").read_text())
            task_names = {x["task_index"]: x["task"] for x in read_lines(source / "meta/tasks.jsonl")}
            if info["codebase_version"] != "v2.1" or info["fps"] != manifest["fps"]:
                raise ValueError("Expected v2.1 with matching manifest FPS")
            if info["total_episodes"] != len(manifest["episodes"]):
                raise ValueError("Manifest episode count mismatch")
            if first_info is None:
                first_info, first_manifest = info, manifest
            elif manifest["task_scales"] != first_manifest["task_scales"]:
                raise ValueError("Task reward scales must be identical; reuse the fixed scales JSON")
            elif info["fps"] != first_info["fps"] or info["robot_type"] != first_info["robot_type"]:
                raise ValueError("Robot/FPS mismatch")
            for key in ("action", "observation.state", *CAMERAS):
                if info["features"][key] != first_info["features"][key]:
                    raise ValueError(f"Feature mismatch: {key}")
            count = 0
            for record in manifest["episodes"]:
                old = record["episode_index"]
                rows = pq.read_table(data_path(source, info, old)).to_pylist()
                check_rows(rows, old, record["length"])
                if {task_names[row["task_index"]] for row in rows} != {record["task"]}:
                    raise ValueError("Episode task mismatch")
                if sum(row["intervention"] for row in rows) != record["intervention_frames"]:
                    raise ValueError("Manifest intervention count mismatch")
                videos = video_paths(source, info, old)
                digest = episode_digest(rows, record["task"], videos)
                if digest != record["content_sha256"]:
                    raise ValueError("Episode contents changed after manifest creation")
                if record["episode_uid"] in seen_uids or digest in seen_content:
                    raise ValueError("Duplicate source episode/content; do not merge a pool with its own inputs")
                seen_uids.add(record["episode_uid"]); seen_content.add(digest)
                task = record["task"]
                task_index = tasks.setdefault(task, len(tasks))
                index = len(episodes)
                reward, dense, value, _ = labels(record["outcome"],
                    np.array([row["intervention"] for row in rows]), manifest["task_scales"][task])
                for i, row in enumerate(rows):
                    row.update(reward=reward[i], reward_label=dense[i], value_label=value[i])
                # Relabel the whole cumulative pool from scratch with the new round's value model.
                for i, row in enumerate(rows):
                    row.update(episode_index=index, frame_index=i, index=total + i, task_index=task_index,
                               adv_ind="positive" if row["intervention"] or record["source_kind"] == "demonstration" else "none")
                write_episode(temporary, info, rows, index, videos)
                records.append(dict(record, episode_index=index))
                episodes.append({"episode_index": index, "tasks": [task], "length": len(rows)})
                stats.append({"episode_index": index, "stats": statistics(rows)})
                total += len(rows); count += len(rows)
            if count != info["total_frames"]:
                raise ValueError("Source total_frames mismatch")
        if first_info is None:
            raise ValueError("No source datasets")
        metadata(temporary, first_info, episodes, stats, tasks, total)
        manifest = write_manifest(temporary, first_manifest["task_scales"], records, first_info["fps"])
        (temporary / "meta/merge.json").write_text(json.dumps({"sources": [str(x) for x in sources],
            "source_versions": [load_manifest(x)["dataset_version"] for x in sources],
            "dataset_version": manifest["dataset_version"]}, indent=2))
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return manifest


def prepare_demonstrations(source, destination, scales, validation_episodes, policy_id, assume_success=False, outcomes_path=None):
    """Import existing v2.1 SFT data without modifying video/action normalization."""
    if not policy_id or bool(assume_success) == bool(outcomes_path):
        raise ValueError("Set policy-id and exactly one of assume-success / outcomes-json")
    sources, destination, temporary = new_output([source], destination)
    source = sources[0]
    episodes, records, stats, tasks, total = [], [], [], {}, 0
    try:
        info = json.loads((source / "meta/info.json").read_text())
        if info["codebase_version"] != "v2.1":
            raise ValueError("Demonstration import expects v2.1; HIL v3 uses convert_to_recap.py")
        task_names = {x["task_index"]: x["task"] for x in read_lines(source / "meta/tasks.jsonl")}
        source_episodes = read_lines(source / "meta/episodes.jsonl")
        validation = set(validation_episodes)
        if not validation.issubset({ep["episode_index"] for ep in source_episodes}):
            raise ValueError("Unknown validation episode")
        outcomes = json.loads(Path(outcomes_path).read_text()) if outcomes_path else None
        source_id = file_digest(source / "meta/info.json")
        features = {key: copy.deepcopy(info["features"][key]) for key in ("action", "observation.state", *CAMERAS)}
        for key in ("timestamp", "reward", "reward_label", "value_label"):
            features[key] = {"dtype": "float32", "shape": [1], "names": None}
        for key in ("frame_index", "episode_index", "index", "task_index", "intervention", "source_frame_index"):
            features[key] = {"dtype": "int64", "shape": [1], "names": None}
        features["adv_ind"] = {"dtype": "string", "shape": [1], "names": None}
        for ep in source_episodes:
            old, index = ep["episode_index"], len(episodes)
            raw = pq.read_table(data_path(source, info, old)).to_pylist()
            check_rows(raw, old, ep["length"])
            if any(not np.isclose(row["timestamp"], i / info["fps"], atol=1e-5) for i, row in enumerate(raw)):
                raise ValueError("Demonstration timestamps are irregular; resample before importing")
            if any(row.get("intervention", 0) for row in raw):
                raise ValueError("Do not import intervention rollouts as demonstrations")
            names = {task_names[row["task_index"]] for row in raw}
            if len(names) != 1:
                raise ValueError("Only single-task episodes are supported")
            task = names.pop(); task_index = tasks.setdefault(task, len(tasks))
            if task not in scales:
                raise ValueError(f"Missing fixed reward scale for demonstration task: {task}")
            outcome = "success" if assume_success else outcomes[str(old)]
            reward, dense, value, indicators = labels(outcome, np.zeros(len(raw), int), scales[task], demonstration=True)
            rows = [{"action": row["action"], "observation.state": row["observation.state"],
                "timestamp": np.float32(i / info["fps"]), "frame_index": i, "episode_index": index,
                "index": total + i, "task_index": task_index, "intervention": 0, "source_frame_index": i,
                "reward": reward[i], "reward_label": dense[i], "value_label": value[i], "adv_ind": str(indicators[i])}
                for i, row in enumerate(raw)]
            videos = video_paths(source, info, old)
            staged = []
            for c, camera in enumerate(CAMERAS):
                if info["features"][camera]["info"]["video.codec"] != "h264":
                    target = temporary / f"staged_{index}_{c}.mp4"
                    convert_video(videos[c], target, 0, np.arange(len(rows)), info["fps"])
                    staged.append(target)
                    videos[c] = target
                features[camera]["info"]["video.codec"] = "h264"
                features[camera]["info"]["video.pix_fmt"] = "yuv420p"
            digest = episode_digest(rows, task, videos)
            write_episode(temporary, info, rows, index, videos)
            for staged_path in staged:
                staged_path.unlink()
            records.append({"episode_index": index, "episode_uid": f"{source_id}:{old}:{digest}",
                "source_dataset_id": source_id, "source_episode_index": old, "source_kind": "demonstration",
                "round_id": 0, "policy_id": policy_id, "task": task, "outcome": outcome,
                "length": len(rows), "intervention_frames": 0, "split": "validation" if old in validation else "train",
                "content_sha256": digest})
            episodes.append({"episode_index": index, "tasks": [task], "length": len(rows)})
            stats.append({"episode_index": index, "stats": statistics(rows)}); total += len(rows)
        if not episodes:
            raise ValueError("No demonstrations")
        metadata(temporary, dict(info, features=features), episodes, stats, tasks, total)
        manifest = write_manifest(temporary, scales, records, info["fps"])
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("merge")
    cmd.add_argument("sources", nargs="+", type=Path)
    cmd.add_argument("--output", type=Path, required=True)
    cmd = sub.add_parser("prepare-demonstrations")
    cmd.add_argument("source", type=Path); cmd.add_argument("destination", type=Path)
    cmd.add_argument("--task-scales-json", type=Path, required=True)
    cmd.add_argument("--validation-episodes", type=int, nargs="*", default=[])
    cmd.add_argument("--policy-id", required=True)
    outcome = cmd.add_mutually_exclusive_group(required=True)
    outcome.add_argument("--assume-success", action="store_true", help="Explicitly confirm all imported demos succeeded")
    outcome.add_argument("--outcomes-json", type=Path, help='JSON mapping original episode IDs to success/failure')
    args = parser.parse_args()
    if args.command == "merge":
        result = merge(args.sources, args.output)
    else:
        result = prepare_demonstrations(args.source, args.destination, load_scales(args.task_scales_json),
            args.validation_episodes, args.policy_id, args.assume_success, args.outcomes_json)
    print(f"Prepared {len(result['episodes'])} episodes; version={result['dataset_version']}")
