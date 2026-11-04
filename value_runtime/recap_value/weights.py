"""Strict value-model weights and atomic, single-device checkpoints."""
from pathlib import Path
import flax.traverse_util
import jax
import numpy as np
import orbax.checkpoint as ocp


def restore(path, *, ema=False):
    with ocp.PyTreeCheckpointer() as checkpointer:
        absolute = str(Path(path).resolve())
        metadata = checkpointer.metadata(absolute)
        arguments = jax.tree.map(lambda _: ocp.ArrayRestoreArgs(
            restore_type=np.ndarray, sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0])), metadata)
        data = checkpointer.restore(absolute, args=ocp.args.PyTreeRestore(item=metadata, restore_args=arguments))
    key = "ema_params" if ema else "params"
    if key not in data:
        raise ValueError(f"Checkpoint missing {key}")
    flat = flax.traverse_util.flatten_dict(data[key])
    if flat and all(key[-1] == "value" for key in flat):
        flat = {key[:-1]: value for key, value in flat.items()}
    data["params"] = flax.traverse_util.unflatten_dict(flat)
    return data


def validate_and_merge(reference, loaded):
    ref = flax.traverse_util.flatten_dict(reference)
    got = flax.traverse_util.flatten_dict(loaded)
    # Bundle contains a SigLIP classifier; the value model uses patch tokens only.
    for key in (("img", "head", "kernel"), ("img", "head", "bias")):
        if key not in ref:
            got.pop(key, None)
    if set(ref) != set(got):
        raise ValueError(f"Value parameter keys mismatch: missing={set(ref)-set(got)}, extra={set(got)-set(ref)}")
    for key in ref:
        if ref[key].shape != got[key].shape:
            raise ValueError(f"Value parameter shape mismatch: {key}")
    return jax.tree.map(lambda r, v: np.asarray(v, dtype=r.dtype), reference,
                        flax.traverse_util.unflatten_dict(got))


def save(path, params, step, opt_state=None):
    payload = {"params": jax.device_get(params), "step": np.array(step)}
    if opt_state is not None:
        payload["opt_state"] = jax.device_get(opt_state)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(str(Path(path).resolve()), payload)
