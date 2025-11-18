"""Basic ConvNN building blocks and models."""

from typing import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn

import src.config.models.image as cfg
from src.architectures.components import (
    Transformer,
)


class FRN(nn.Module):
    """Filter Response Normalization (FRN) with Thresholded ReLU activation."""

    eps_init: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Applies Filter Response Normalization with Thresholded ReLU."""
        eps = self.param("eps", lambda key, shape: jnp.full(shape, self.eps_init), (1,))
        gamma = self.param("gamma", nn.initializers.ones, (1, 1, 1, x.shape[-1]))
        beta = self.param("beta", nn.initializers.zeros, (1, 1, 1, x.shape[-1]))
        tau = self.param("tau", nn.initializers.zeros, (1, 1, 1, x.shape[-1]))

        # Compute mean squared activation per channel
        nu2 = jnp.mean(jnp.square(x), axis=(1, 2), keepdims=True)
        x = x * jax.lax.rsqrt(nu2 + jnp.abs(eps))

        return jnp.maximum(gamma * x + beta, tau)


class LeNet(nn.Module):
    """Implementation of LeNet."""

    config: cfg.LeNetConfig

    def setup(self):
        """Initialize the fully connected neural network."""
        self.core = LeNetCore(
            activation=self.config.activation.flax_activation,
            out_dim=self.config.out_dim,
            use_bias=self.config.use_bias,
        )

    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Forward pass."""
        return self.core(x)


class LeNetCore(nn.Module):
    """Core Implementation of LeNet."""

    activation: Callable
    out_dim: int
    use_bias: bool

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """Forward pass.

        Args:
            x (jnp.ndarray): The input data of
            shape (batch_size, channels, height, width).
        """
        x = nn.Conv(
            features=6, kernel_size=(5, 5), strides=(1, 1), padding=2, name="conv1"
        )(x)
        x = self.activation(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = nn.Conv(
            features=16, kernel_size=(5, 5), strides=(1, 1), padding=0, name="conv2"
        )(x)
        x = self.activation(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=120, use_bias=self.use_bias, name="fc1")(x)
        x = self.activation(x)
        x = nn.Dense(features=84, use_bias=self.use_bias, name="fc2")(x)
        x = self.activation(x)
        x = nn.Dense(features=self.out_dim, use_bias=self.use_bias, name="fc3")(x)
        return x

class BasicBlockFRN(nn.Module):
    """Basic ResNet block with two convolutional layers and FRN normalization."""

    features: int
    stride: int = 1
    use_downsample: bool = False
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Forward pass for the BasicBlock.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            train: Whether the model is in training mode.

        Returns:
            Output tensor of the same shape as input.
        """
        identity = x

        # First convolution
        x = nn.Conv(
            self.features, (3, 3), strides=self.stride, padding="SAME", use_bias=False
        )(x)
        x = FRN()(x)
        # x = self.activation(x)

        # Second convolution
        x = nn.Conv(self.features, (3, 3), strides=1, padding="SAME", use_bias=False)(x)

        # Downsample if required
        if self.use_downsample:
            identity = nn.Conv(
                self.features, (1, 1), strides=self.stride, use_bias=False
            )(identity)
            identity = FRN()(identity)

        x += identity
        x = FRN()(x)
        return x


class ResNetCore(nn.Module):
    """Core implementation of ResNet."""

    layers: list[int]
    num_classes: int
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Forward pass through ResNet.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            train: Whether the model is in training mode.

        Returns:
            Output tensor of shape (batch_size, num_classes).
        """
        # TODO 7x7 + stride 2 for ImageNet
        x = nn.Conv(64, (3, 3), strides=1, padding="SAME", use_bias=False)(x)
        x = FRN()(x)
        # x = self.activation(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")

        # Residual layers
        x = self._make_layer(64, self.layers[0], stride=1, train=train)(x)
        x = self._make_layer(128, self.layers[1], stride=2, train=train)(x)
        x = self._make_layer(256, self.layers[2], stride=2, train=train)(x)
        x = self._make_layer(512, self.layers[3], stride=2, train=train)(x)

        x = jnp.mean(x, axis=(1, 2))  # pool over width and height
        x = nn.Dense(self.num_classes)(x)
        return x

    def _make_layer(
        self, features: int, blocks: int, stride: int, train: bool
    ) -> nn.Sequential:
        """Creates a layer of residual blocks.

        Args:
            features: Number of output features for the blocks.
            blocks: Number of blocks in the layer.
            stride: Stride for the first block in the layer.

        Returns:
            A sequential module containing the residual blocks.
        """

        def layer_fn(x: jnp.ndarray) -> jnp.ndarray:
            # First block with stride
            x = BasicBlockFRN(features, stride, use_downsample=(stride != 1))(
                x, train=train
            )
            # Remaining blocks
            for _ in range(1, blocks):
                x = BasicBlockFRN(features)(x, train=train)
            return x

        return layer_fn


class ResNet7Core(nn.Module):
    """Core implementation of ResNet7."""

    num_classes: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Forward pass through ResNet7.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            train: Whether the model is in training mode.

        Returns:
            Output tensor of shape (batch_size, num_classes).
        """
        x = nn.Conv(32, kernel_size=(3, 3), padding="SAME")(x)
        x = FRN()(x)

        x = nn.Conv(64, kernel_size=(3, 3), padding="SAME")(x)
        x = FRN()(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        residual = x

        x = nn.Conv(64, kernel_size=(3, 3), padding="SAME")(x)
        x = FRN()(x)

        # x = nn.Conv(128, kernel_size=(3, 3), padding="SAME")(x)
        # x = FRN()(x)
        # x = x + residual

        x = nn.Conv(128, kernel_size=(3, 3), padding="SAME")(x)
        x = FRN()(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(128, kernel_size=(3, 3), padding="SAME")(x)
        x = FRN()(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        residual = x

        # x = nn.Conv(512, kernel_size=(3, 3), padding="SAME")(x)
        # x = FRN()(x)
        # x = nn.relu(x)

        x = nn.Conv(128, kernel_size=(3, 3), padding="SAME")(x)
        x = FRN()(x)
        x = x + residual

        x = jnp.mean(x, axis=(1, 2))  # Global average pooling
        x = nn.Dense(self.num_classes)(x)

        return x


class ResNet7(nn.Module):
    """ResNet7 implementation."""

    config: cfg.ResNetConfig

    def setup(self):
        """Set up the ResNet7 model."""
        self.core = ResNet7Core(
            num_classes=self.config.out_dim,
        )

    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Forward pass through ResNet7."""
        return self.core(x, train)


class ResNet(nn.Module):
    """ResNet implementation.

    Inspired by https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    """

    config: cfg.ResNetConfig

    def setup(self):
        """Set up the ResNet18 model."""
        self.core = ResNetCore(
            layers=self.config.layers,
            num_classes=self.config.out_dim,
            activation=self.config.activation.flax_activation,
        )

    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Forward pass through ResNet18.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            train: Whether the model is in training mode.

        Returns:
            Output tensor of shape (batch_size, num_classes).
        """
        return self.core(x, train)


class ViTCore(nn.Module):
    """Core implementation of Vision Transformer (Inspired from https://github.com/vballoli/vit-flax)."""

    patch_size: int
    dim: int
    depth: int
    num_heads: int
    transformer_fc_dim: int
    classifier_fc_dim: int
    out_dim: int
    initializer: Callable = nn.initializers.normal(stddev=1.0)

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Applies the Vision Transformer to the input tensor.
        Args:
            x (jnp.ndarray): Input tensor image, shape (batch, height, width, channels).
            train (bool): Whether the model is in training mode (for dropout, etc.).
        """
        b, h, w, c = x.shape

        # 1. Patch + Position Embedding

        # Patch embedding using a single Conv layer
        patch_embed = nn.Conv(
            features=self.dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            name="patch_embedding",
        )(x)

        # Output of Conv is (batch, num_patches_h, num_patches_w, dim)
        # We want (batch, num_patches, dim)
        num_patches = patch_embed.shape[1] * patch_embed.shape[2]

        # Reshape to (batch, num_patches, dim)
        patch_embed = patch_embed.reshape((b, num_patches, self.dim))

        # CLS token
        cls_token = self.param("class_tokens", self.initializer, (1, 1, self.dim))
        cls_token = jnp.tile(cls_token, (b, 1, 1))  # Broadcast to batch size

        # Positional embedding
        pos_embedding = self.param(
            "pos_embedding", self.initializer, (1, num_patches + 1, self.dim)
        )

        # Prepend CLS token and add positional embedding
        x = jnp.concatenate([cls_token, patch_embed], axis=1)
        x += pos_embedding

        # 2. Transformer Encoder
        x = Transformer(
            depth=self.depth,
            num_heads=self.num_heads,
            feed_forward_dim=self.transformer_fc_dim,
        )(x, train=train)

        # 3. Classifier Head
        x = x[:, 0]

        x = nn.Dense(features=self.classifier_fc_dim, name="mlp_fc1")(x)
        x = nn.gelu(x)
        x = nn.Dense(features=self.out_dim, name="mlp_output")(x)

        return x


class ViT(nn.Module):
    """Vision Transformer (ViT) model."""

    config: cfg.ViTConfig

    def setup(self):
        """Initialize the ViT core model."""
        self.core = ViTCore(
            patch_size=self.config.patch_size,
            dim=self.config.dim,
            depth=self.config.depth,
            num_heads=self.config.num_heads,
            transformer_fc_dim=self.config.transformer_fc_dim,
            classifier_fc_dim=self.config.classifier_fc_dim,
            out_dim=self.config.out_dim,
        )

    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Forward pass."""
        return self.core(x, train=train)