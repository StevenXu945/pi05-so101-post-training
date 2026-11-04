"""Independent RECAP value training; run with value_runtime environment."""
import argparse
import json
from pathlib import Path
import sys
import time


def train_step(model, optimizer, observation, target, rng, trainable_filter):
    import flax.nnx as nnx
    import optax
    loss, grads = nnx.value_and_grad(lambda net: net.compute_loss(rng, observation, target, train=True),
                                    argnums=nnx.DiffState(0, trainable_filter))(model)
    params = nnx.state(model, trainable_filter)
    updates, optimizer = optax.adamw(2.5e-5).update(grads, optimizer, params)
    nnx.update(model, optax.apply_updates(params, updates))
    return loss, optimizer


def evaluation_step(model, observation, target):
    """One forward pass for both validation CE and expected value; JIT at call site."""
    import jax
    import jax.numpy as jnp
    from .model import normalize_value_target_shape, target_to_twohot, dist_to_value
    from .observation import preprocess_observation

    observation = preprocess_observation(jax.random.key(0), observation, train=False,
                                         image_keys=list(observation.images))
    logits = model(observation, train=False).astype(jnp.float32)
    target = normalize_value_target_shape(target)
    distribution = target_to_twohot(target) if target.ndim == 1 else target
    loss = -jnp.sum(distribution.astype(jnp.float32) * jax.nn.log_softmax(logits, axis=-1), axis=-1)
    return jnp.mean(loss), dist_to_value(logits)


def main(args):
    started = time.monotonic()

    def stage(message):
        print(f"[{time.monotonic() - started:8.1f}s] {message}", flush=True)

    stage("Importing value runtime (JAX / Gemma)...")
    import flax.nnx as nnx
    import jax
    import numpy as np
    import optax
    from tqdm.auto import tqdm
    from .config import ValueModelConfig
    from .model import ValueModel
    from .data import ValueDataset
    from .weights import restore, save, validate_and_merge

    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.batch_size, args.steps, args.val_interval, args.save_interval) < 1:
        raise ValueError("Training counts must be positive")
    stage(f"Devices: {jax.devices()}")
    stage(f"Loading training data: {args.data}")
    train = ValueDataset(args.data, args.tokenizer, split="train")
    stage(f"Loading validation data (train frames: {len(train.rows)})")
    validation = ValueDataset(args.data, args.tokenizer, split="validation")
    stage(f"Data ready: train={len(train.rows)} frames, validation={len(validation.rows)} frames")
    stage("Initializing SigLIP + Gemma value model...")
    model = ValueModel(ValueModelConfig(), rngs=nnx.Rngs(args.seed))
    parameters = nnx.state(model)
    stage(f"Restoring weights: {args.checkpoint}")
    loaded = restore(args.checkpoint)["params"]
    parameters.replace_by_pure_dict(validate_and_merge(parameters.to_pure_dict(), loaded))
    nnx.update(model, parameters)
    stage("Weights loaded; preparing optimizer and train step")
    def trainable(path, value):
        # NNX filters receive VariableState wrappers; nnx.Param is checked by All.
        if args.freeze_mode == "none":
            return True
        if str(path[0]) == "img":
            return False
        return args.freeze_mode != "all_backbones" or str(path[0]) != "llm"
    trainable_filter = nnx.All(nnx.Param, trainable)
    if not nnx.state(model, trainable_filter).flat_state():
        raise ValueError("No trainable value parameters")
    print("trainable_parameters=" + str(sum(value.value.size for value in
          nnx.state(model, trainable_filter).flat_state().values())), flush=True)
    optimizer = optax.adamw(2.5e-5).init(nnx.state(model, trainable_filter))
    run_step = nnx.jit(lambda net, opt, obs, target, key: train_step(net, opt, obs, target, key, trainable_filter))
    run_evaluation = nnx.jit(evaluation_step)
    args.output.mkdir(parents=True)
    (args.output / "run.json").write_text(json.dumps({
        "data": str(args.data.resolve()), "dataset_version": train.manifest["dataset_version"],
        "initialization_checkpoint": str(args.checkpoint.resolve()),
        "initialization_mode": "weights-only, fixed initial value checkpoint across rounds", "tokenizer": str(args.tokenizer.resolve()), "seed": args.seed,
        "freeze_mode": args.freeze_mode, "task_scales": train.manifest["task_scales"],
        "validation_max_batches": args.validation_max_batches,
        "label_recipe": "paper-v1", "backbone": "SigLIP-So400m + Gemma3-270M",
        "state_encoding": "raw six-joint positions, .3g text; arm degrees, gripper 0-100",
        "reference_commit": "a44350b4a99f1d7f6137635ab3e894a5f72f5bda"}, indent=2))
    rng = np.random.default_rng(args.seed)
    stage("Starting training; the first step includes image decoding and JIT compilation.")
    with tqdm(total=args.steps, desc="Value train", unit="step", dynamic_ncols=True) as progress:
        for step in range(1, args.steps + 1):
            if step == 1:
                tqdm.write("Loading the first image batch...", file=sys.stdout)
            observation, target = train.batch(rng.integers(len(train.rows), size=args.batch_size))
            if step == 1:
                tqdm.write("Running the first train step (JIT compilation may take time)...", file=sys.stdout)
            loss, optimizer = run_step(model, optimizer, observation, target, jax.random.key(args.seed + step))
            loss_value = float(loss)  # Synchronize GPU before counting a completed step.
            if not np.isfinite(loss_value):
                raise ValueError("Nonfinite value loss")
            progress.set_postfix(loss=f"{loss_value:.6f}", refresh=False)
            progress.update(1)
            if step == 1 or step % 100 == 0:
                tqdm.write(f"step={step}/{args.steps} train_loss={loss_value:.6f}", file=sys.stdout)
            if step % args.val_interval == 0 or step == args.steps:
                losses, errors = [], []
                batch_count = (len(validation.rows) + args.batch_size - 1) // args.batch_size
                if args.validation_max_batches:
                    batch_count = min(batch_count, args.validation_max_batches)
                tqdm.write(f"step={step}: validating {batch_count} batches...", file=sys.stdout)
                with tqdm(total=batch_count, desc="Validation", unit="batch", leave=False,
                          dynamic_ncols=True) as validation_progress:
                    for offset in range(0, len(validation.rows), args.batch_size):
                        if args.validation_max_batches and offset >= args.validation_max_batches * args.batch_size:
                            break
                        read_started = time.monotonic()
                        obs, target = validation.batch(range(offset, min(offset + args.batch_size, len(validation.rows))))
                        read_seconds = time.monotonic() - read_started
                        if offset == 0:
                            tqdm.write("Validation first batch includes JIT warm-up if this shape is new.", file=sys.stdout)
                        inference_started = time.monotonic()
                        val_loss, value = run_evaluation(model, obs, target)
                        value = np.asarray(value)
                        loss_value_validation = float(val_loss)
                        inference_seconds = time.monotonic() - inference_started
                        losses.extend([loss_value_validation] * len(target))
                        errors.extend(np.abs(value - target).tolist())
                        validation_progress.set_postfix(read=f"{read_seconds:.3f}s", infer=f"{inference_seconds:.3f}s", refresh=False)
                        validation_progress.update(1)
                tqdm.write(f"validation_loss={np.mean(losses):.6f} validation_mae={np.mean(errors):.6f}", file=sys.stdout)
            if step % args.save_interval == 0 or step == args.steps:
                destination = args.output / f"step_{step:08d}"
                tqdm.write(f"step={step}: saving checkpoint to {destination}...", file=sys.stdout)
                save(destination, nnx.state(model).to_pure_dict(), step, optimizer)
                tqdm.write(f"step={step}: checkpoint saved", file=sys.stdout)
    stage("Training completed")
    print("device_memory_stats=" + str(jax.devices()[0].memory_stats()), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Full value-model checkpoint, not policy params")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-mode", choices=["all_backbones", "siglip_only", "none"], default="all_backbones")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--val-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--validation-max-batches", type=int, default=0, help="0 evaluates all; positive values are smoke-test only")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def cli(argv=None):
    main(parse_args(argv))


if __name__ == "__main__":
    cli()
