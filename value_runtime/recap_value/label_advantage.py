"""Predict values and atomically label a new v2.1 dataset copy."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile


def main(args):
    import flax.nnx as nnx
    import jax
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from openpi.recap.targets import n_step_advantage, advantage_labels, fit_thresholds
    from .config import ValueModelConfig
    from .model import ValueModel
    from .data import ValueDataset
    from .weights import restore, validate_and_merge
    if args.output.exists() or args.data.resolve() in args.output.resolve().parents:
        raise ValueError("Use a new output outside the source dataset")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    from openpi.recap.data_manifest import threshold_mask
    train = ValueDataset(args.data, args.tokenizer, split="train")
    run = json.loads((args.checkpoint.parent / "run.json").read_text())
    if run.get("dataset_version") != train.manifest["dataset_version"]:
        raise ValueError("Relabel the same cumulative dataset version used to train the value model")
    if run["task_scales"] != train.manifest["task_scales"]:
        raise ValueError("Dataset and value checkpoint task scales differ")
    model = ValueModel(ValueModelConfig(), rngs=nnx.Rngs(0))
    state = nnx.state(model)
    state.replace_by_pure_dict(validate_and_merge(state.to_pure_dict(), restore(args.checkpoint)["params"]))
    nnx.update(model, state)
    compute = nnx.jit(lambda net, obs: net.compute_value(jax.random.key(0), obs, train=False))
    paths = sorted((args.data / "data").rglob("*.parquet"))
    rows = [row for path in paths for row in pq.read_table(path).to_pylist()]
    train.rows = rows
    predictions = []
    for i in range(0, len(rows), args.batch_size):
        observation = train.batch(range(i, min(i + args.batch_size, len(rows))))[0]
        predictions.append(np.asarray(compute(model, observation)))
        if i == 0 or i % (args.batch_size * 100) == 0:
            print(f"value inference: {min(i + args.batch_size, len(rows))}/{len(rows)}", flush=True)
    values = np.concatenate(predictions)
    advantages = np.zeros(len(rows), np.float32)
    for episode in sorted({row["episode_index"] for row in rows}):
        positions = np.array([i for i, row in enumerate(rows) if row["episode_index"] == episode])
        if [rows[i]["frame_index"] for i in positions] != list(range(len(positions))):
            raise ValueError("Episode frame order is not contiguous")
        advantages[positions] = n_step_advantage([rows[i]["reward_label"] for i in positions], values[positions], args.lookahead)
    manifest = train.manifest
    train_ids = set(manifest["splits"]["train"])
    train_mask = np.array([row["episode_index"] in train_ids for row in rows])
    task_names = np.array([train.tasks[row["task_index"]] for row in rows])
    fit_mask = threshold_mask(rows, manifest, args.threshold_round)
    # Validation labels stay unused; their values cannot influence task thresholds.
    thresholds = fit_thresholds(advantages[train_mask], task_names[train_mask], fit_mask[train_mask], args.positive_ratio)
    indicators = np.full(len(rows), "none", dtype="<U8")
    indicators[train_mask], _ = advantage_labels(advantages[train_mask],
        np.array([row["intervention"] for row in rows])[train_mask], task_names[train_mask], thresholds=thresholds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=args.output.name + ".partial-", dir=args.output.parent))
    try:
        shutil.copytree(args.data, temporary, dirs_exist_ok=True)
        offset = 0
        for path in paths:
            table = pq.read_table(path)
            count = table.num_rows
            labels = pa.array(indicators[offset:offset + count].tolist())
            table = table.set_column(table.schema.get_field_index("adv_ind"), "adv_ind", labels)
            pq.write_table(table, temporary / path.relative_to(args.data))
            offset += count
        (temporary / "meta/advantage_labeling.json").write_text(json.dumps({
            "checkpoint": str(args.checkpoint.resolve()), "thresholds": thresholds,
            "dataset_version": manifest["dataset_version"], "threshold_round": args.threshold_round,
            "threshold_selection": "train, current round, rollout, whole episode intervention-free",
            "threshold_frames": int(fit_mask.sum()), "validation_labels": "none",
            "lookahead": args.lookahead, "positive_ratio": args.positive_ratio,
            "terminal_bootstrap": 0, "task_scales": run["task_scales"]}, indent=2))
        pq.write_table(pa.table({"episode_index": [row["episode_index"] for row in rows],
            "frame_index": [row["frame_index"] for row in rows], "value": values,
            "advantage": advantages, "adv_ind": indicators.tolist(),
            "intervention": [row["intervention"] for row in rows]}), temporary / "meta/advantage_predictions.parquet")
        temporary.rename(args.output)
    except BaseException:
        shutil.rmtree(temporary)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "checkpoint", "tokenizer", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--threshold-round", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lookahead", type=int, default=50)
    parser.add_argument("--positive-ratio", type=float, default=0.4)
    return parser.parse_args(argv)


def cli(argv=None):
    main(parse_args(argv))


if __name__ == "__main__":
    cli()
