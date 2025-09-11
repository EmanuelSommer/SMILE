"""DataLoader Implementations for Image Data."""

import jax
import jax.numpy as jnp

from src.config.data import DataConfig, DatasetType, Source
from src.dataset.base import BaseLoader
from src.types import PRNGKey


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
            data_x, data_y, test_size = _get_torchvision_data(self._name, self._dir)
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

            # normalize
            if self.config.normalize:
                data_x = data_x / 255.0

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


def _get_torchvision_data(name: str, dir: str) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    """Get torchvision datasets."""
    from torchvision import datasets, transforms

    match name:
        case "mnist":
            d_train = datasets.MNIST(
                dir, train=True, download=True, transform=transforms.ToTensor()
            )
            d_test = datasets.MNIST(
                dir, train=False, download=True, transform=transforms.ToTensor()
            )
        case "fashion_mnist":
            d_train = datasets.FashionMNIST(
                dir, train=True, download=True, transform=transforms.ToTensor()
            )
            d_test = datasets.FashionMNIST(
                dir, train=False, download=True, transform=transforms.ToTensor()
            )
            # retrieve original test dataset with `test_split: 0.14285`
        case "cifar10":
            d_train = datasets.CIFAR10(
                dir,
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        # CIFAR10 mean and std from
                        # https://github.com/kuangliu/pytorch-cifar/issues/19
                        transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                        ),
                    ]
                ),
            )
            d_test = datasets.CIFAR10(
                dir,
                train=False,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        # CIFAR10 mean and std from
                        # https://github.com/kuangliu/pytorch-cifar/issues/19
                        transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                        ),
                    ]
                ),
            )
            # retrieve original test dataset with `test_split: 0.16666`
        case "cifar100":
            d_train = datasets.CIFAR100(
                dir, train=True, download=True, transform=transforms.ToTensor()
            )
            d_test = datasets.CIFAR100(
                dir, train=False, download=True, transform=transforms.ToTensor()
            )
        case _:
            raise NotImplementedError(f"Dataset {name} is not supported.")

    test_size = d_test.data.shape[0]
    data_x = jnp.concatenate(
        [jnp.array(d_train.data), jnp.array(d_test.data)],
        axis=0,
    )
    if len(data_x.shape) == 3:
        data_x = data_x[:, None, ...]
        data_x = data_x.transpose((0, 2, 3, 1))  # ensure NHWC / NDHWC

    data_y = jnp.concatenate(
        [jnp.array(d_train.targets), jnp.array(d_test.targets)],
        axis=0,
    )
    return data_x, data_y, test_size
