"""Read-only validation, timing and explicit initial RECAP labels."""
import json
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

PREFIX = "complementary_info."
CAMERAS = ("observation.images.front", "observation.images.wrist")


def scalar(value):
    array = np.asarray(value)
    if array.size != 1 or not np.isfinite(array).all():
        raise ValueError(f"Expected finite scalar: {value}")
    return float(array.reshape(-1)[0])


def load(root):
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    if info["codebase_version"] != "v3.0":
        raise ValueError("Expected recorded LeRobot v3.0 dataset")
    rows = [row for path in sorted((root / "data").rglob("*.parquet"))
            for row in pq.read_table(path).to_pylist()]
    episodes = [row for path in sorted((root / "meta/episodes").rglob("*.parquet"))
                for row in pq.read_table(path).to_pylist()]
    outcomes = {}
    for line in (root / "outcomes.jsonl").read_text().splitlines():
        item = json.loads(line)
        index = item["episode_index"]
        if index in outcomes:
            raise ValueError(f"Duplicate outcome for episode {index}")
        if item["episode_outcome"] not in {"success", "failure", "aborted"}:
            raise ValueError(f"Invalid outcome: {item}")
        outcomes[index] = item
    tasks = pq.read_table(root / "meta/tasks.parquet").to_pylist()
    # Pandas-written task tables sometimes store task text as their index column.
    task_map = {int(row["task_index"]): row.get("task", row.get("__index_level_0__")) for row in tasks}
    if any(not isinstance(text, str) for text in task_map.values()):
        raise ValueError(f"Unsupported task table schema: {tasks}")
    rows.sort(key=lambda row: (row["episode_index"], row["frame_index"]))
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["episode_index"]), []).append(row)
    if len(rows) != info["total_frames"] or len(grouped) != info["total_episodes"]:
        raise ValueError("Frame/episode totals disagree with metadata")
    metadata = {int(ep["episode_index"]): ep for ep in episodes}
    if set(grouped) != set(outcomes) or set(grouped) != set(metadata):
        raise ValueError("Episode IDs must match data, metadata and outcomes exactly")
    for index, frames in grouped.items():
        if len(frames) != outcomes[index]["frames"] or len(frames) != metadata[index]["length"]:
            raise ValueError(f"Episode {index}: length mismatch")
        times = []
        for number, row in enumerate(frames):
            if row["frame_index"] != number or row["task_index"] not in task_map:
                raise ValueError(f"Episode {index}: frame/task index mismatch")
            for key in ("action", "observation.state"):
                value = np.asarray(row[key])
                if value.shape != (6,) or not np.isfinite(value).all():
                    raise ValueError(f"Episode {index}: invalid {key}")
            for key in ("is_intervention", "is_paused", "policy_action_valid"):
                if scalar(row[PREFIX + key]) not in (0, 1):
                    raise ValueError(f"Episode {index}: invalid {key}")
            if scalar(row[PREFIX + "is_paused"]) and scalar(row[PREFIX + "is_intervention"]):
                raise ValueError(f"Episode {index}: paused and manual simultaneously")
            times.append(scalar(row[PREFIX + "elapsed_s"]))
        if len(times) > 1 and np.any(np.diff(times) <= 0):
            raise ValueError(f"Episode {index}: elapsed_s must strictly increase")
    return info, grouped, metadata, outcomes, task_map


def resample_indices(frames, fps, max_gap_s):
    """Remove intervals whose originating row is paused, then zero-order hold.

    State/image/action always come from the same preceding source row. Large gaps
    require an explicit larger max_gap_s rather than silent interpolation.
    """
    times = np.array([scalar(row[PREFIX + "elapsed_s"]) for row in frames])
    paused = np.array([bool(scalar(row[PREFIX + "is_paused"])) for row in frames])
    active_time = np.r_[0.0, np.cumsum(np.diff(times) * (~paused[:-1]))]
    kept = np.flatnonzero(~paused)
    if not len(kept):
        raise ValueError("Episode contains no active frames")
    source_time = active_time[kept] - active_time[kept[0]]
    if len(source_time) > 1 and np.max(np.diff(source_time)) > max_gap_s:
        raise ValueError(f"Active sampling gap {np.max(np.diff(source_time)):.3f}s exceeds {max_gap_s}s")
    grid = np.arange(int(np.floor(source_time[-1] * fps)) + 1) / fps
    selected = kept[np.searchsorted(source_time, grid, side="right") - 1]
    # Preserve terminal outcome observation even if it is between grid ticks.
    if selected[-1] != kept[-1]:
        selected = np.r_[selected, kept[-1]]
    return selected


def labels(outcome, intervention, scale, *, demonstration=False):
    from openpi.recap.targets import rewards_and_returns
    raw, dense, value = rewards_and_returns(len(intervention), outcome, scale)
    return raw, dense, value, np.where(np.ones(len(intervention), bool) if demonstration else intervention, "positive", "none")


def report(root):
    info, grouped, _, outcomes, _ = load(root)
    result = {"source": str(Path(root).resolve()), "fps_declared": info["fps"], "episodes": []}
    for index, frames in grouped.items():
        times = np.array([scalar(row[PREFIX + "elapsed_s"]) for row in frames])
        gaps = np.diff(times)
        result["episodes"].append({
            "episode_index": index, "outcome": outcomes[index]["episode_outcome"], "frames": len(frames),
            "manual_frames": sum(scalar(row[PREFIX + "is_intervention"]) for row in frames),
            "paused_frames": sum(scalar(row[PREFIX + "is_paused"]) for row in frames),
            "median_interval_s": float(np.median(gaps)) if len(gaps) else None,
            "max_interval_s": float(np.max(gaps)) if len(gaps) else None,
        })
    return result
