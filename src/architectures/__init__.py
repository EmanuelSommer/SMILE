"""Module for Registering new models."""

from src.architectures.image import LeNet, ResNet, ResNet7
from src.architectures.tabular import FCN
from src.architectures.text import GPT

__all__ = [
    "LeNet",
    "ResNet",
    "ResNet7",
    "FCN",
    "GPT",
]
