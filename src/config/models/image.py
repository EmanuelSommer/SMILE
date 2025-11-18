"""Configuration for models processing image data."""

from dataclasses import field

from src.config.models.base import Activation, ModelConfig


class LeNetConfig(ModelConfig):
    """LeNet Model Configuration."""

    model: str = "LeNet"
    activation: Activation = field(
        default=Activation.SIGMOID,
        metadata={"description": "Activation func. for hidden layers"},
    )
    out_dim: int = field(
        default=10,
        metadata={"description": "Output dimension of the model"},
    )
    use_bias: bool = field(
        default=True,
        metadata={"description": "Whether to include bias terms"},
    )


class ResNetConfig(ModelConfig):
    """ResNet Model Configuration."""

    model: str = "ResNet"
    activation: Activation = field(
        default=Activation.RELU,
        metadata={"description": "Activation func. for hidden layers"},
    )
    out_dim: int = field(
        default=10,
        metadata={"description": "Output dimension of the model"},
    )
    layers: list[int] = field(
        default_factory=lambda: [2, 2, 2, 2],
        metadata={"description": "Number of layers in each block (default: ResNet18)"},
    )


class ResNet7Config(ModelConfig):
    """ResNet7 Model Configuration."""

    model: str = "ResNet7"
    activation: Activation = field(
        default=Activation.RELU,
        metadata={"description": "Activation func. for hidden layers"},
    )
    out_dim: int = field(
        default=10,
        metadata={"description": "Output dimension of the model"},
    )

class ViTConfig(ModelConfig):
    """Vision Transformer (ViT) Model Configuration."""

    model: str = "ViT"
    patch_size: int = field(
        default=8,
        metadata={"description": "Size of image patches"},
    )
    dim: int = field(
        default=64,
        metadata={"description": "Embedding dimension"},
    )
    depth: int = field(
        default=3,
        metadata={"description": "Number of transformer blocks"},
    )
    num_heads: int = field(
        default=4,
        metadata={"description": "Number of attention heads"},
    )
    transformer_fc_dim: int = field(
        default=2048,
        metadata={
            "description": "Hidden dimension of transformer feed-forward network"
        },
    )
    classifier_fc_dim: int = field(
        default=2048,
        metadata={"description": "Hidden dimension of classifier network"},
    )
    out_dim: int = field(
        default=10,
        metadata={"description": "Number of output classes"},
    )