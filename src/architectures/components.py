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
    
class TransformerBlock(nn.Module):
    """A single Transformer block (Pre-Norm)."""

    num_heads: int
    feed_forward_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Applies one layer of Pre-Norm Self-Attention and MLP."""
        # Attention Block (Pre-Norm + Residual)
        x_norm_attn = nn.LayerNorm()(x)
        attn_out = nn.SelfAttention(num_heads=self.num_heads)(x_norm_attn)
        x = x + attn_out

        # FFN Block (Pre-Norm + Residual)
        x_norm_ffn = nn.LayerNorm()(x)

        ffn_out = FullyConnected(
            hidden_sizes=(self.feed_forward_dim, x.shape[-1]),
            activation=nn.gelu,
            last_layer_activation=None,
        )(x_norm_ffn)

        x = x + ffn_out

        return x


class Transformer(nn.Module):
    """A stack of Transformer blocks."""

    depth: int
    num_heads: int
    feed_forward_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Applies residual normalized attention layers to input."""
        for _ in range(self.depth):
            x = TransformerBlock(
                num_heads=self.num_heads, feed_forward_dim=self.feed_forward_dim
            )(x, train=train)
        return x
