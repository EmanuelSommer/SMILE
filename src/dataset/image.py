"""DataLoader Implementations for Image Data."""

import logging
import jax
import jax.numpy as jnp

from src.config.data import DataConfig, DatasetType, Source
from src.dataset.base import BaseLoader
from src.types import PRNGKey

logger = logging.getLogger(__name__)


class ImageLoader(BaseLoader):
    """DataLoader for Image data."""

    def __init__(self, config: DataConfig, rng_key: PRNGKey, n_chains: int):
        """__init__ method for the ImageLoader class."""
        assert config.data_type == DatasetType.IMAGE
        super().__init__(config=config, rng_key=rng_key, n_chains=n_chains)

        # data augmentation (to be improved)
        import dm_pix as pix

        def basic_transform(key, img):
            """Transform the image expects either HWC or CHW format."""
            # Compile individual transformations with jit
            random_flip_lr = jax.jit(
                lambda k, x: pix.random_flip_left_right(k, x, probability=0.5)
            )
            # random_flip_ud = jax.jit(
            #     lambda k, x: pix.random_flip_up_down(k, x, probability=0.1)
            # )

            # Apply flips
            key, subkey = jax.random.split(key)
            img = random_flip_lr(subkey, img)

            # key, subkey = jax.random.split(key)
            # img = random_flip_ud(subkey, img)

            # Apply cropping
            key, subkey = jax.random.split(key)
            should_crop = jax.random.uniform(subkey) < 0.95

            h, w = img.shape[:2]
            min_crop_h = max(int(h * 0.7), 1)
            min_crop_w = max(int(w * 0.7), 1)

            def crop_fn(img, key):
                original_dtype = img.dtype
                random_crop_jit = jax.jit(
                    lambda k, x: pix.random_crop(
                        k, x, (min_crop_h, min_crop_w) + x.shape[2:]
                    )
                )
                cropped = random_crop_jit(key, img)
                resized = jax.image.resize(cropped, img.shape, method="bilinear")
                return resized.astype(original_dtype)

            # Conditionally apply cropping
            key, subkey = jax.random.split(key)
            img = jax.lax.cond(should_crop, lambda: crop_fn(img, subkey), lambda: img)
            # img = crop_fn(img, subkey)
            return img

        def identity_transform(key, img):
            """Identity transform."""
            return img

        self._transform = (
            jax.jit(basic_transform)
            if config.augment == "basic"
            else jax.jit(identity_transform)
        )

    def load_data(self):
        """Load the dataset from different sources."""
        if self.config.source == Source.TORCHVISION:
            data_x, data_y, test_size = _get_torchvision_data(
                self._name, self._dir, self.config.normalize
            )
            non_test_x, non_test_y = self.shuffle_arrays(
                data_x[:-test_size], data_y[:-test_size]
            )
            data_x = jnp.concatenate(
                [
                    non_test_x,
                    data_x[-test_size:],
                ],
                axis=0,
            )
            data_y = jnp.concatenate(
                [
                    non_test_y,
                    data_y[-test_size:],
                ],
                axis=0,
            )

            # append loading_permutations for not permuted test data
            self.loading_permutation = jnp.concatenate(
                [self.loading_permutation, jnp.arange(data_x.shape[0])[-test_size:]],
                axis=0,
            )

            # Limit the number of datapoints
            data_x = data_x[: self.config.datapoint_limit]
            data_y = data_y[: self.config.datapoint_limit]

            if self.config.flatten:
                data_x = jax.vmap(jnp.ravel)(data_x)

            return data_x, data_y

        raise NotImplementedError(
            f"Source {self.config.source} is not supported for image data."
        )

    def augment(
        self,
        data: jnp.ndarray,
        rng_key: PRNGKey | None = None,
    ):
        """Apply augmentations to the data."""
        if len(data.shape) == 4:
            rng_keys = jax.random.split(rng_key, data.shape[0])
            data = jax.vmap(self._transform, in_axes=(0))(rng_keys, data)
        else:  # TODO implement for NDHWC!
            raise NotImplementedError(
                f"Augmentation not implemented for data with shape {data.shape}."
            )
        return data


def _get_torchvision_data(
    name: str, dir: str, normalize: bool
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    """Get torchvision datasets."""
    import os

    import numpy as np
    import torch
    from torchvision import datasets, transforms

    def _build_data_from_iterable(dataset) -> tuple[np.ndarray, np.ndarray]:
        """Iterate through a torch dataset and stack the results."""
        logger.info(f"Loading and transforming data for {name}...")
        data_x_list = []
        data_y_list = []
        # Use torch.utils.data.DataLoader for efficient loading
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=512, shuffle=False, num_workers=0
        )
        for imgs, labels in loader:
            data_x_list.append(imgs.numpy())  # imgs are (N, C, H, W)
            data_y_list.append(labels.numpy())

        data_x = np.concatenate(data_x_list, axis=0)
        data_y = np.concatenate(data_y_list, axis=0)
        logger.info("...done.")
        return data_x, data_y

    match name:
        case "mnist":
            if normalize:
                mnist_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                )
            else:
                mnist_transform = transforms.Compose(
                    [
                        transforms.PILToTensor(),
                    ]
                )

            d_train = datasets.MNIST(
                dir, train=True, download=True, transform=mnist_transform
            )
            d_test = datasets.MNIST(
                dir, train=False, download=True, transform=mnist_transform
            )
        case "fashion_mnist":
            if normalize:
                fashion_mnist_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                )
            else:
                fashion_mnist_transform = transforms.Compose(
                    [
                        transforms.PILToTensor(),
                    ]
                )
            d_train = datasets.FashionMNIST(
                dir, train=True, download=True, transform=fashion_mnist_transform
            )
            d_test = datasets.FashionMNIST(
                dir, train=False, download=True, transform=fashion_mnist_transform
            )
        case "cifar10":
            if normalize:
                cifar10_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                        ),
                    ]
                )
            else:
                cifar10_transform = transforms.Compose(
                    [
                        transforms.PILToTensor(),
                    ]
                )
            d_train = datasets.CIFAR10(
                dir,
                train=True,
                download=True,
                transform=cifar10_transform,
            )
            d_test = datasets.CIFAR10(
                dir,
                train=False,
                download=True,
                transform=cifar10_transform,
            )
        case "cifar100":
            # Using CIFAR-100 stats, similar to CIFAR-10
            if normalize:
                cifar100_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]
                        ),
                    ]
                )
            else:
                cifar100_transform = transforms.Compose(
                    [
                        transforms.PILToTensor(),
                    ]
                )
            d_train = datasets.CIFAR100(
                dir, train=True, download=True, transform=cifar100_transform
            )
            d_test = datasets.CIFAR100(
                dir, train=False, download=True, transform=cifar100_transform
            )
        case "imagenette":
            IMG_SIZE = (160, 160)
            if normalize:
                imagenet_transform = transforms.Compose(
                    [
                        transforms.Resize(IMG_SIZE),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        ),
                    ]
                )
            else:
                imagenet_transform = transforms.Compose(
                    [
                        transforms.Resize(IMG_SIZE),
                        transforms.ToTensor(),
                    ]
                )
            train_path = os.path.join(dir, "imagenette2", "train")
            val_path = os.path.join(dir, "imagenette2", "val")
            download_data = not (
                os.path.exists(train_path) and os.path.exists(val_path)
            )
            if download_data:
                logger.info(
                    f"Imagenette data not found in {dir}/imagenette2. Downloading..."
                )

            d_train = datasets.Imagenette(
                dir,
                split="train",
                download=download_data,
                transform=imagenet_transform,
            )
            d_test = datasets.Imagenette(
                dir,
                split="val",
                download=False,
                transform=imagenet_transform,
            )
        case _:
            raise NotImplementedError(f"Dataset {name} is not supported.")

    train_x, train_y = _build_data_from_iterable(d_train)
    test_x, test_y = _build_data_from_iterable(d_test)

    test_size = test_x.shape[0]

    data_x = jnp.concatenate([train_x, test_x], axis=0)  # (N, C, H, W)
    data_y = jnp.concatenate([train_y, test_y], axis=0)

    # Transpose from (N, C, H, W) to (N, H, W, C) for the rest of the code
    data_x = data_x.transpose((0, 2, 3, 1))

    return data_x, data_y, test_size
