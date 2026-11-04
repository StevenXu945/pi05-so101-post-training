import dataclasses


@dataclasses.dataclass(frozen=True)
class ValueModelConfig:
    """Independent SigLIP So400m + Gemma3-270M configuration; no policy registry changes."""
    dtype: str = "bfloat16"
    siglip_variant: str = "So400m/14"
    embed_dim: int = 640
    vocab_size: int = 262144
    max_token_len: int = 96

    @property
    def gemma_config(self):
        from gemma.gm.nn._gemma import Gemma3_270M
        return Gemma3_270M().config
