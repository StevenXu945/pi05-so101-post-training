"""Offline open-loop evaluation of the served openpi SO-101 policy on train_dataset.

This does not connect to or move a robot. At each chunk boundary, use the recorded
state and both recorded camera frames to predict a 16-step absolute action chunk.
Compare it with the next 16 recorded absolute actions from the same episode.

Example (run the policy server separately):
    python scripts/eval_so101_open_loop.py --episode-ids 0 1 --max-chunks 10

This is a training-set fit diagnostic, not a held-out success or rollout metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import time

import numpy as np
from openpi_client import image_tools, msgpack_numpy
from websockets.sync.client import connect


JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
HORIZON = 16
LOG = logging.getLogger("eval_so101_open_loop")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("train_dataset"))
    parser.add_argument("--episode-ids", type=int, nargs="+", default=[0])
    parser.add_argument("--max-chunks", type=int, default=10, help="Maximum queries per episode")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default="Grab the toy into the box",
                        help="Training-time default prompt; override to test another instruction")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, default=Path("open_loop_eval"))
    args = parser.parse_args()
    if args.max_chunks < 1 or args.timeout <= 0 or not np.isfinite(args.timeout):
        parser.error("--max-chunks and --timeout must be positive")
    if any(episode < 0 for episode in args.episode_ids):
        parser.error("--episode-ids must be nonnegative")
    if not args.dataset.is_dir():
        parser.error(f"Dataset directory does not exist: {args.dataset}")
    return args


def image_for_policy(tensor) -> np.ndarray:
    """LeRobot returns float CHW; openpi expects uint8 HWC."""
    image = tensor.permute(1, 2, 0).cpu().numpy()
    image = image_tools.convert_to_uint8(image)
    return image_tools.resize_with_pad(image, 224, 224)


def infer(socket, frame: dict, prompt: str, timeout: float) -> tuple[np.ndarray, float]:
    observation = {
        "observation/image": image_for_policy(frame["observation.images.front"]),
        "observation/wrist_image": image_for_policy(frame["observation.images.wrist"]),
        "observation/state": frame["observation.state"].cpu().numpy().astype(np.float32),
        "prompt": prompt,
    }
    start = time.monotonic()
    socket.send(msgpack_numpy.packb(observation))
    response = socket.recv(timeout=timeout)
    elapsed_ms = (time.monotonic() - start) * 1000
    if isinstance(response, str):
        raise RuntimeError(f"Policy server error: {response}")
    actions = np.asarray(msgpack_numpy.unpackb(response)["actions"], dtype=np.float32)
    if actions.shape != (HORIZON, len(JOINTS)) or not np.isfinite(actions).all():
        raise ValueError(f"Expected finite ({HORIZON}, {len(JOINTS)}) actions, got {actions.shape}")
    return actions, elapsed_ms


def draw_episode(path: Path, episode_id: int, state: np.ndarray, gt: np.ndarray,
                 pred: np.ndarray, boundaries: list[int]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        LOG.warning("matplotlib is unavailable; skipping PNG plot (CSV and summary are still saved)")
        return
    fig, axes = plt.subplots(len(JOINTS), 1, figsize=(13, 14), sharex=True, constrained_layout=True)
    for joint, ax in enumerate(axes):
        ax.plot(state[:, joint], color="0.5", label="recorded state", linewidth=1)
        ax.plot(gt[:, joint], label="recorded action", linewidth=1.3)
        ax.plot(pred[:, joint], label="predicted action", linewidth=1.3)
        for boundary in boundaries:
            ax.axvline(boundary, color="0.8", linewidth=0.7)
        ax.set_ylabel(f"{JOINTS[joint]}\n{'deg' if joint < 5 else '0–100'}")
        ax.grid(alpha=0.2)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Compared action step; vertical lines mark inference points")
    fig.suptitle(f"SO-101 episode {episode_id}: open-loop absolute actions")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def evaluate_episode(dataset, socket, episode_id: int, max_chunks: int, prompt: str, timeout: float,
                     output_dir: Path) -> dict:
    episode = dataset.meta.episodes[episode_id]
    begin, end = int(episode["dataset_from_index"]), int(episode["dataset_to_index"])
    states, gt_actions, predictions, frame_ids, boundaries, latencies = [], [], [], [], [], []

    for chunk, start in enumerate(range(begin, end, HORIZON)):
        if chunk >= max_chunks:
            break
        frame = dataset[start]
        predicted, latency_ms = infer(socket, frame, prompt, timeout)
        length = min(HORIZON, end - start)
        for offset in range(length):
            raw = dataset.get_raw_item(start + offset)
            states.append(np.asarray(raw["observation.state"], dtype=np.float32))
            gt_actions.append(np.asarray(raw["action"], dtype=np.float32))
            predictions.append(predicted[offset])
            frame_ids.append(int(raw["frame_index"]))
        boundaries.append(len(frame_ids) - length)
        latencies.append(latency_ms)
        LOG.info("episode=%d chunk=%d frame=%d steps=%d infer=%.1f ms",
                 episode_id, chunk, start - begin, length, latency_ms)

    state = np.stack(states)
    gt = np.stack(gt_actions)
    pred = np.stack(predictions)
    error = pred - gt
    if not np.isfinite(state).all() or not np.isfinite(gt).all():
        raise ValueError(f"Episode {episode_id} has non-finite recorded joints or actions")

    with (output_dir / f"episode_{episode_id}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_index", *[f"state_{j}" for j in JOINTS],
                         *[f"gt_{j}" for j in JOINTS], *[f"pred_{j}" for j in JOINTS]])
        for frame_id, s, g, p in zip(frame_ids, state, gt, pred, strict=True):
            writer.writerow([frame_id, *s.tolist(), *g.tolist(), *p.tolist()])

    draw_episode(output_dir / f"episode_{episode_id}.png", episode_id, state, gt, pred, boundaries)
    return {
        "episode_id": episode_id,
        "steps": len(frame_ids),
        "chunks": len(latencies),
        "mae": float(np.mean(np.abs(error))),
        "mse": float(np.mean(error**2)),
        "mae_by_joint": dict(zip(JOINTS, np.mean(np.abs(error), axis=0).tolist(), strict=True)),
        "gt_min_by_joint": dict(zip(JOINTS, np.min(gt, axis=0).tolist(), strict=True)),
        "gt_max_by_joint": dict(zip(JOINTS, np.max(gt, axis=0).tolist(), strict=True)),
        "mean_infer_ms": float(np.mean(latencies)),
    }


def main() -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    args = parse_args()
    dataset = LeRobotDataset(args.dataset.name, root=args.dataset.resolve(), download_videos=False)
    if any(episode >= dataset.meta.total_episodes for episode in args.episode_ids):
        raise ValueError(f"Dataset has episode IDs 0 through {dataset.meta.total_episodes - 1}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    with connect(f"ws://{args.host}:{args.port}", compression=None, max_size=None,
                 open_timeout=args.timeout, close_timeout=1) as socket:
        metadata = msgpack_numpy.unpackb(socket.recv(timeout=args.timeout))
        LOG.info("Connected to policy server; metadata: %s", metadata)
        for episode_id in args.episode_ids:
            summary = evaluate_episode(dataset, socket, episode_id, args.max_chunks,
                                       args.prompt, args.timeout, args.output_dir)
            summaries.append(summary)
            LOG.info("episode=%d steps=%d MAE=%.3f MSE=%.3f",
                     episode_id, summary["steps"], summary["mae"], summary["mse"])
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"Saved plots, CSV files and summary to {args.output_dir.resolve()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
