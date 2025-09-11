"""Basic NN building blocks."""

from typing import Callable

import jax.numpy as jnp
from flax import linen as nn


class FullyConnected(nn.Module):
    """Fully connected Neural Network."""

    hidden_sizes: tuple[int, ...]
    activation: Callable
    use_bias: bool = True
    last_layer_activation: Callable | None = None
    blockid: str | None = None
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """Forward pass."""
        if self.blockid is None:
            blockid = ""
        else:
            blockid = f"{self.blockid}_"
        for i, hidden_size in enumerate(self.hidden_sizes):
            x = nn.Dense(
                features=hidden_size,
                dtype=self.dtype,
                use_bias=self.use_bias,
                name=f"{blockid}layer{i}",
            )(x)
            if i < len(self.hidden_sizes) - 1:
                x = self.activation(x)
            else:
                if self.last_layer_activation is not None:
                    x = self.last_layer_activation(x)
        return x


class MaskedMultiHeadSelfAttention(nn.Module):
    """Masked Multi-Head Attention Module."""

    n_heads: int
    qkv_dim: int
    bias: bool
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = False):
        """Forward Pass."""
        _, _, C = x.shape
        mask = nn.make_causal_mask(x[:, :, 0], dtype=jnp.bool_)
        out = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.qkv_dim,
            dtype=self.dtype,
            use_bias=self.bias,
            deterministic=deterministic,
            out_features=C,
        )(x, mask=mask)
        return out


class TokenEmbedding(nn.Module):
    """Embedding Layer for Tokens with optional Positional Encoding."""

    vocab_size: int
    emb_size: int
    dtype: jnp.dtype
    pos_size: int | None = None

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """Forward Pass."""
        embed = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.emb_size,
            dtype=self.dtype,
            name="Embedding",
        )(x)
        if self.pos_size:
            pos = jnp.arange(x.shape[1]).reshape(1, -1)
            pos = nn.Embed(
                num_embeddings=self.pos_size,
                features=self.emb_size,
                dtype=self.dtype,
                name="PositionEmbedding",
            )(pos)
            embed += pos
        return embed
