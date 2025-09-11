"""DataLoader Implementations for Text Data."""

import re
from typing import TYPE_CHECKING, Optional

import jax
import jax.numpy as jnp

from src.config.data import DataConfig, DatasetType, Source
from src.dataset.base import BaseLoader
from src.types import PRNGKey

if TYPE_CHECKING:
    from src.dataset.tokenizers import Tokenizer


class TextLoader(BaseLoader):
    """DataLoader for text data.

    This class loads and processes text data for language modeling tasks.
    It handles tokenization, context window creation, and batch generation.

    Example usage:
        from src.dataset.text import TextLoader
        from src.config.core import Config
        from src.dataset.tokenizers import SingleCharTokenizer
        import jax

        cfg = Config.from_yaml("scripts/nanogpt/cfg.yaml")
        tokenizer = SingleCharTokenizer()
        loader = TextLoader(
            config=cfg.data,
            rng_key=jax.random.key(cfg.rng),
            n_chains=2,
            tokenizer=tokenizer,
            context=20,
            omit_freq=100
        )

        # Get shapes of first batch
        [(a.shape, b.shape) for i,(a,b) in enumerate(loader.iter("test", batch_size=10)) if i==0]

        # Decode tokens to see the actual text
        test = [(a, b) for i,(a,b) in enumerate(loader.iter("test", batch_size=10)) if i==0]
        tokenizer.decode(test[0][0][1, 4].tolist()), tokenizer.decode([test[0][1][1, 4].tolist()])
    """

    def __init__(
        self,
        config: DataConfig,
        rng_key: PRNGKey,
        n_chains: int,
        tokenizer: "Tokenizer",
        context: int,
        omit_freq: int,
    ):
        """Initialze the TextLoader class.

        Args:
            config: The configuration for the data.
            rng_key: The random number generator key.
            n_chains: Number of chains the dataloader is used for.
            tokenizer: The tokenizer to use.
            context: The context size for the data.
            omit_freq: The frequency to omit.
        """
        assert config.data_type == DatasetType.TEXT
        self.tokenizer = tokenizer
        self.context = context
        self.omit_freq = omit_freq
        if not config.features:
            raise ValueError("Feature column mus be specified in the config.")
        self._feature = config.features[0]
        super().__init__(config=config, rng_key=rng_key, n_chains=n_chains)

    def load_data(self):
        """Load text data from different sources."""
        if self.config.source == Source.HUGGINGFACE:
            from datasets import DatasetDict, concatenate_datasets, load_dataset

            path = self.config.path.split("/")
            data = (
                load_dataset(path=path[-2], name="/".join(path[-1:]))
                if len(path) > 1
                else load_dataset(path=path[0])
            )
            if isinstance(data, DatasetDict):
                data = concatenate_datasets(
                    dsets=[data[key] for key in data.keys() if key != "unsupervised"]
                )

            assert (
                self.config.target_column in data.column_names
            ), f"Target column {self.config.target_column} not found in the dataset."
            if not self.config.features:
                raise ValueError("Feature column mus be specified in the config.")

            assert (
                self._feature in data.column_names
            ), f"Feature column {self._feature} not found in the dataset."

            # Omit rare chars
            if self.omit_freq:
                corpus = "".join(data[self._feature])
                omit = [c for c in set(corpus) if corpus.count(c) < self.omit_freq]
                data = data.map(
                    lambda x: self._remove_chars(x, omit), batched=True, batch_size=None
                )

            # Tokenize
            if self.tokenizer.needs_training:
                self.tokenizer.train(text="".join(data[self._feature]))
            data = data.map(self._tokenize, batched=True, batch_size=None)
            data.set_format(type="jax")

            x: jnp.ndarray = data[self._feature]
            y: jnp.ndarray = data[self.config.target_column]

            # no shuffle for text data
            x = x[: self.config.datapoint_limit]
            y = y[: self.config.datapoint_limit]

            return x, y

        if self.config.source == Source.LOCAL:
            if self.config.path.endswith(".txt"):
                with open(self.config.path, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                raise NotImplementedError(
                    "Loading from non-txt files is not supported."
                )

            # Clean text: remove excess whitespace
            # text = re.sub(r"\s+", " ", text).strip()

            # Omit rare characters if specified
            if self.omit_freq:
                char_counts = {c: text.count(c) for c in set(text)}
                omit = [c for c, count in char_counts.items() if count < self.omit_freq]
                pattern = rf'({"|".join(re.escape(v) for v in omit)})'
                text = re.sub(pattern, "", text)

            # Tokenize the entire text
            if self.tokenizer.needs_training:
                self.tokenizer.train(text=text)

            # Convert to token IDs
            tokens = self.tokenizer.encode(text)
            data = jnp.array(tokens, dtype=jnp.int32)

            # Create a single dimensional array of tokens that can be sliced
            # This format works with the get_batch method shown in the prompt
            x = data
            # For compatibility with the existing interface, create a dummy y
            y = jnp.ones(data.shape[0], dtype=jnp.int32)  # Minimal dummy target

            # Apply datapoint limit if specified
            if self.config.datapoint_limit and self.config.datapoint_limit < len(x):
                x = x[: self.config.datapoint_limit]

            return x, y

        raise NotImplementedError(
            f"Source {self.config.source} is not supported at this time."
        )

    def _iter(
        self,
        data: tuple[jnp.ndarray, jnp.ndarray],
        batch_size: Optional[int] = None,
        chains: Optional[jax.Array] = None,
        shuffle: bool = True,
        progress: bool = False,
    ):
        """Generate batches of context windows with next token targets.

        Args:
            data: Tuple containing the token sequence as the first element.
            batch_size: Size of each batch. Required.
            chains: For which chains to load the data.
            shuffle: Whether to use different random samples across chains.
            progress: Whether to show a progress bar.

        Yields:
            Batches of context windows (x) and next token targets (y).
        """
        from functools import partial

        from tqdm import tqdm

        batch_size = batch_size or 128

        if chains is None:
            chains = jnp.arange(self.n_chains)

        token_data = data[0]  # Get the token sequence
        if len(token_data) <= self.context + 1:
            raise ValueError(
                f"Data length ({len(token_data)}) must be greater than context size + 1 ({self.context + 1})"
            )

        # Define the function to get random context windows
        @partial(jax.jit, static_argnums=(2, 3))
        def get_batch(data, rng, batch_size, block_size):
            ix = jax.random.randint(
                rng, shape=(batch_size, 1), minval=0, maxval=len(data) - block_size - 1
            )
            slice_vmap = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))
            x = slice_vmap(data, ix, (block_size,))
            y = slice_vmap(data, ix + 1, (block_size,))
            # y = y[:, -1] # for strict next token prediction
            return x, y

        # Generate batches
        n_batches = max(1, len(token_data) // (batch_size * self.context))
        iterator = range(n_batches)
        if progress:
            iterator = tqdm(iterator, desc="Data Batch")

        for i in iterator:
            # Get a different batch for each chain if shuffle is True
            if shuffle:
                chain_batches = [
                    get_batch(token_data, key, batch_size, self.context)
                    for key in self.chainwise_key(chains=chains)
                ]
                x_batches = jnp.stack([batch[0] for batch in chain_batches])
                y_batches = jnp.stack([batch[1] for batch in chain_batches])
            else:
                # Use the same batch for all chains
                rng_key = self.chainwise_key(chains=chains)[0]
                x_batch, y_batch = get_batch(
                    token_data, rng_key, batch_size, self.context
                )
                x_batches = jnp.stack([x_batch for _ in chains])
                y_batches = jnp.stack([y_batch for _ in chains])

            yield x_batches, y_batches

    def _tokenize(self, element: dict[str, list]):
        """Tokenize the features in the dataset."""
        if self.context:
            element[self._feature] = self.tokenizer.encode_batch_with_padding(
                element[self._feature], self.context
            )
        else:
            element[self._feature] = self.tokenizer.encode_batch(element[self._feature])
        return element

    def _remove_chars(self, element: dict[str, list[str]], omit: list[str]):
        """Remove characters from the dataset."""
        pattern = rf'({"|".join(re.escape(v) for v in omit)})'
        element[self._feature] = [
            re.sub(pattern, "", t) for t in element[self._feature]
        ]
        return element

    @property
    def data_train_len(self):
        data = self.data_train[0]
        if len(data.shape) > 1:
            return len(data)
        return len(data) // self.context

    @property
    def data_valid_len(self):
        data = self.data_valid[0]
        if len(data.shape) > 1:
            return len(data)
        return len(data) // self.context

    @property
    def data_test_len(self):
        data = self.data_test[0]
        if len(data.shape) > 1:
            return len(data)
        return len(data) // self.context
