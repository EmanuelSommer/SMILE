"""Basic ConvNN building blocks and models."""

from typing import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn

import src.config.models.image as cfg


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
