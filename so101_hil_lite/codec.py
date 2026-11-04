"""NumPy msgpack wire format used by openpi-client (Apache-2.0).

Kept locally so Evo-RL's NumPy 2 environment need not install openpi-client,
which currently pins NumPy below 2.
"""

from functools import partial

import msgpack
import numpy as np


def _pack(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = partial(msgpack.Packer, default=_pack)
packb = partial(msgpack.packb, default=_pack)
unpackb = partial(msgpack.unpackb, object_hook=_unpack)
