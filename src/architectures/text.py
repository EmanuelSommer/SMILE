"""GPT Implementation in Flax based on https://github.com/karpathy/nanoGPT."""

import flax.linen as nn
import jax
import jax.numpy as jnp

import src.config.models.text as cfg
from src.architectures.components import (
    FullyConnected,
    MaskedMultiHeadSelfAttention,
    TokenEmbedding,
)


class GPT(nn.Module):
    """High-Level Implementation of GPT in Flax."""

    config: cfg.GPTConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True):
        """Forward Pass."""
        B, T = x.shape
        assert T <= self.config.context_len, "Cannot forward, model block size exceeded"
        # Embedding Tokens
        dtype = self.config.dtype.flax_dtype
        emb = TokenEmbedding(
            vocab_size=self.config.vocab_size,
            emb_size=self.config.emb_size,
            dtype=dtype,
            pos_size=self.config.context_len,
        )(x)
        # Dropout
        x = nn.Dropout(self.config.dropout, deterministic=not train)(emb)
        # Transformer Blocks
        for _ in range(self.config.n_blocks):
            x = Block(
                n_heads=self.config.n_heads,
                mlp_hidden_size=4 * self.config.emb_size,
                dropout=self.config.dropout,
                bias=self.config.bias,
                dtype=dtype,
            )(x, deterministic=not train)

        x = nn.LayerNorm(dtype=dtype, use_bias=self.config.bias)(x)
        logits = nn.Dense(
            self.config.vocab_size,
            dtype=dtype,
            use_bias=False,
            name="DenseLogits",
        )(x)

        return logits


class Block(nn.Module):
    """Single GPT Block."""

    n_heads: int
    mlp_hidden_size: int
    dropout: float
    bias: bool
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = False):
        """Forward Pass."""
        _, _, C = x.shape
        x = nn.LayerNorm(dtype=self.dtype, use_bias=self.bias)(x)
        # Skip Connection
        x += nn.Dropout(self.dropout, deterministic=deterministic)(
            MaskedMultiHeadSelfAttention(
                n_heads=self.n_heads,
                qkv_dim=C,
                bias=self.bias,
                dtype=self.dtype,
            )(x, deterministic=deterministic)
        )
        x = nn.LayerNorm(dtype=self.dtype, use_bias=self.bias)(x)
        # Skip Connection
        x += nn.Dropout(self.dropout, deterministic=deterministic)(
            FullyConnected(
                hidden_sizes=[self.mlp_hidden_size, x.shape[-1]],
                activation=nn.gelu,
                use_bias=self.bias,
                last_layer_activation=None,
                blockid="FFN",
                dtype=self.dtype,
            )(x)
        )
        return x