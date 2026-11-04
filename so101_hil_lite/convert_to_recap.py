"""Convert immutable HIL v3 trajectories to openpi-readable LeRobot v2.1."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from dataset_tools import CAMERAS, PREFIX, labels, load, resample_indices, scalar


def write_lines(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def statistics(rows):
    stats = {}
    for key in rows[0]:
        if key == "adv_ind":
            continue
        array = np.asarray([row[key] for row in rows], dtype=np.float64)
        if array.ndim == 1:
            array = array[:, None]
        stats[key] = {name: value.tolist() for name, value in {
            "min": array.min(0), "max": array.max(0), "mean": array.mean(0),
            "std": array.std(0), "count": np.array([len(rows)]),
        }.items()}
    return stats


def convert_video(source, destination, start, selected, fps):
    destination.parent.mkdir(parents=True, exist_ok=True)
    cursor = 0
    with av.open(str(source)) as reader, av.open(str(destination), "w") as writer:
        output = None
        for number, frame in enumerate(reader.decode(video=0)):
            if cursor == len(selected):
                break
            if number != start + selected[cursor]:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            if output is None:
                output = writer.add_stream("libx264", rate=fps)
                output.width, output.height = rgb.shape[1], rgb.shape[0]
                output.pix_fmt = "yuv420p"
                output.options = {"preset": "fast", "crf": "18"}
            while cursor < len(selected) and number == start + selected[cursor]:
                image = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                for packet in output.encode(image):
                    writer.mux(packet)
                cursor += 1
        if cursor != len(selected):
            raise ValueError(f"Video {source} ended before selected frame {selected[cursor]}")
        for packet in output.encode():
            writer.mux(packet)


def convert(source, destination, fps=30, max_gap_s=0.5, validation_episodes=(), failure_cost_multiplier=1.0, *, task_scales=None, task_max_steps=None,
            source_kind="rollout", round_id=1, policy_id=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if fps <= 0 or max_gap_s <= 0:
        raise ValueError("fps and max_gap_s must be positive")
    if destination.exists() or source == destination or source in destination.parents:
        raise ValueError("Destination must be new and outside the source dataset")
    info, grouped, metadata, outcomes, tasks = load(source)
    from openpi.recap.targets import fit_scales
    prepared = {index: resample_indices(frames, fps, max_gap_s) for index, frames in grouped.items()
                if outcomes[index]["episode_outcome"] != "aborted"}
    validation_episodes = set(validation_episodes)
    if not validation_episodes.issubset(prepared):
        raise ValueError("Validation IDs must refer to valid source episodes")
    needed_tasks = {tasks[grouped[index][0]["task_index"]] for index in prepared}
    if not needed_tasks:
        raise ValueError("No success/failure trajectories to convert")
    from openpi.recap.data_manifest import validate_scales, write_manifest, episode_digest
    import hashlib
    if source_kind not in {"demonstration", "rollout"} or round_id < 0:
        raise ValueError("Invalid source kind / round")
    if (source_kind == "demonstration" and round_id != 0) or (source_kind == "rollout" and round_id < 1):
        raise ValueError("Demonstrations use round 0; rollouts use round >= 1")
    if task_scales is not None and task_max_steps is not None:
        raise ValueError("Choose task-scales-json or task-max-steps")
    if task_scales is None:
        if task_max_steps is None:
            raise ValueError("Set an explicit task horizon or reuse saved task scales")
        task_scales = fit_scales([(task, task_max_steps) for task in needed_tasks], failure_cost_multiplier)
    scales = validate_scales(task_scales)
    if not needed_tasks.issubset(scales):
        raise ValueError("Missing task scales")
    dataset_id = hashlib.sha256((json.dumps(info, sort_keys=True) + (source / "outcomes.jsonl").read_text()).encode()).hexdigest()
    manifest_episodes = []
    split_ids = {"train": [], "validation": []}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".partial-", dir=destination.parent))
    episodes, all_stats, mapping = [], [], []
    total = 0
    try:
        (temporary / "meta").mkdir()
        features = {key: dict(info["features"][key]) for key in ("action", "observation.state", *CAMERAS)}
        for key in ("timestamp", "reward", "reward_label", "value_label"):
            features[key] = {"dtype": "float32", "shape": [1], "names": None}
        for key in ("frame_index", "episode_index", "index", "task_index", "intervention", "source_frame_index"):
            features[key] = {"dtype": "int64", "shape": [1], "names": None}
        features["adv_ind"] = {"dtype": "string", "shape": [1], "names": None}
        for original, frames in grouped.items():
            outcome = outcomes[original]["episode_outcome"]
            if outcome == "aborted":
                continue
            selected = prepared[original]
            intervention = np.array([int(scalar(frames[i][PREFIX + "is_intervention"])) for i in selected])
            reward, dense, value, advantage = labels(outcome, intervention, scales[tasks[frames[0]["task_index"]]], demonstration=source_kind == "demonstration")
            episode_index = len(episodes)
            split_ids["validation" if original in validation_episodes else "train"].append(episode_index)
            chunk = episode_index // 1000
            rows = []
            for frame_index, original_frame in enumerate(selected):
                raw = frames[original_frame]
                rows.append({"action": raw["action"], "observation.state": raw["observation.state"],
                             "timestamp": np.float32(frame_index / fps), "frame_index": frame_index,
                             "episode_index": episode_index, "index": total + frame_index,
                             "task_index": int(raw["task_index"]), "intervention": int(intervention[frame_index]),
                             "source_frame_index": int(original_frame), "reward": reward[frame_index],
                             "reward_label": dense[frame_index], "value_label": value[frame_index],
                             "adv_ind": str(advantage[frame_index])})
            path = temporary / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows), path)
            episode_meta = metadata[original]
            video_paths = []
            for camera in CAMERAS:
                prefix = f"videos/{camera}"
                file = source / f"videos/{camera}/chunk-{episode_meta[prefix + '/chunk_index']:03d}/file-{episode_meta[prefix + '/file_index']:03d}.mp4"
                start = round(episode_meta[prefix + "/from_timestamp"] * info["fps"])
                target = temporary / f"videos/chunk-{chunk:03d}/{camera}/episode_{episode_index:06d}.mp4"
                convert_video(file, target, start, selected, fps)
                video_paths.append(target)
                height, width, channels = features[camera]["shape"]
                features[camera]["info"] = {"video.height": height, "video.width": width,
                    "video.codec": "h264", "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                    "video.fps": fps, "video.channels": channels, "has_audio": False}
            episodes.append({"episode_index": episode_index, "tasks": sorted({tasks[row['task_index']] for row in rows}),
                             "length": len(rows)})
            all_stats.append({"episode_index": episode_index, "stats": statistics(rows)})
            mapping.append({"episode_index": episode_index, "source_episode_index": original,
                            "outcome": outcome, "source_frames": len(frames), "output_frames": len(rows),
                            "unique_source_frames": len(set(selected.tolist()))})
            task = tasks[frames[0]["task_index"]]
            if any(tasks[row["task_index"]] != task for row in rows):
                raise ValueError("Only single-task episodes are supported")
            recorded_policy = outcomes[original].get("policy_id") or outcomes[original].get("collector_policy_id")
            identity = policy_id or recorded_policy
            if not identity:
                raise ValueError("Set --policy-id when source outcome lacks policy identity")
            manifest_episodes.append({"episode_index": episode_index,
                "episode_uid": f"{dataset_id}:{original}", "source_dataset_id": dataset_id,
                "source_episode_index": original, "source_kind": source_kind,
                "round_id": round_id, "policy_id": identity, "task": task,
                "outcome": outcome, "length": len(rows), "intervention_frames": int(intervention.sum()),
                "split": "validation" if original in validation_episodes else "train",
                "content_sha256": episode_digest(rows, task, video_paths)})
            total += len(rows)
        if not episodes:
            raise ValueError("No success/failure trajectories to convert")
        result = {"codebase_version": "v2.1", "robot_type": "so101_follower", "total_episodes": len(episodes),
                  "total_frames": total, "total_tasks": len(tasks), "total_videos": len(episodes) * len(CAMERAS),
                  "total_chunks": (len(episodes) + 999) // 1000, "chunks_size": 1000, "fps": fps,
                  "splits": {"train": f"0:{len(episodes)}"}, "features": features,
                  "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                  "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"}
        (temporary / "meta/info.json").write_text(json.dumps(result, indent=2))
        write_lines(temporary / "meta/episodes.jsonl", episodes)
        write_lines(temporary / "meta/episodes_stats.jsonl", all_stats)
        write_lines(temporary / "meta/tasks.jsonl", [{"task_index": key, "task": text} for key, text in tasks.items()])
        (temporary / "meta/conversion.json").write_text(json.dumps({"source": str(source),
            "label_recipe": "paper-v1", "recipe_is_exact_limb_port": False,
            "task_scales": scales, "splits": split_ids, "failure_cost_multiplier": failure_cost_multiplier,
            "timing": "pause-compressed preceding-frame hold; terminal frame preserved", "fps": fps,
            "max_gap_s": max_gap_s, "episodes": mapping}, indent=2))
        write_manifest(temporary, scales, manifest_episodes, fps)
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    print(f"Converted {len(episodes)} episodes / {total} frames to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-gap-s", type=float, default=0.5)
    parser.add_argument("--validation-episodes", type=int, nargs="*", default=[])
    parser.add_argument("--failure-cost-multiplier", type=float, default=1.0)
    parser.add_argument("--task-scales-json", type=Path, help="Reuse scales from a prior recap_manifest.json or scales JSON")
    parser.add_argument("--task-max-steps", type=int, help="Explicit first-round horizon in resampled frames")
    parser.add_argument("--source-kind", choices=["demonstration", "rollout"], default="rollout")
    parser.add_argument("--round-id", type=int, default=1)
    parser.add_argument("--policy-id", help="Fallback/override collector identity")
    args = parser.parse_args()
    from openpi.recap.data_manifest import load_scales
    convert(args.source, args.destination, args.fps, args.max_gap_s, args.validation_episodes, args.failure_cost_multiplier,
            task_scales=load_scales(args.task_scales_json) if args.task_scales_json else None,
            task_max_steps=args.task_max_steps, source_kind=args.source_kind, round_id=args.round_id, policy_id=args.policy_id)
