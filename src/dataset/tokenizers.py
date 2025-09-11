"""Utilities for Datasets like tokenizers, helper functions, etc."""

import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Iterable

import tiktoken
from tokenizers import Tokenizer as HuggingfaceTokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


class SpecialToken(str, Enum):
    """Special tokens for tokenizers."""

    PAD = "<|PAD|>"
    CLS = "<|CLS|>"

    @classmethod
    def to_list(cls) -> list[str]:
        """Return a list of special tokens."""
        return [token.value for token in cls]


class Tokenizer(ABC):
    """Base class for tokenizers."""

    @abstractmethod
    def encode(self, string: str) -> list[int]:
        """Encode a string into tokens."""
        pass

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        """Decode tokens into a string."""
        pass

    @property
    @abstractmethod
    def padding_token_id(self) -> int:
        """Padding token id."""
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Vocabulary size."""
        pass

    def pad(self, tokens: list[int], length: int) -> list[int]:
        """Pad tokens to a given length."""
        assert length > 0, "length must be a positive integer."
        if len(tokens) >= length:
            return tokens[:length]
        return tokens + [self.padding_token_id] * (length - len(tokens))

    def encode_with_padding(self, text: str, length: int) -> list[int]:
        """Encode a string into tokens with padding."""
        return self.pad(self.encode(text), length)

    def encode_batch_with_padding(
        self, texts: Iterable[str], length: int
    ) -> list[list[int]]:
        """Encode a batch of strings into tokens with padding."""
        return [self.encode_with_padding(text, length) for text in texts]

    @property
    def needs_training(self) -> bool:
        """Check if the tokenizer needs training."""
        return self.vocab_size == 0

    def train(self, text: Iterable[str] | str) -> None:
        """Train the tokenizer."""
        Warning("Tokenizer is pretrained.")

    def encode_batch(self, strings: Iterable[str]) -> list[list[int]]:
        """Batch encode a list of strings into tokens."""
        return [self.encode(string) for string in strings]


class CustomBPETokenizer(Tokenizer):
    """Custom Byte-Pair Encoding Tokenizer."""

    def __init__(self, pad_len: int | None = None, max_vocab_size: int = 500) -> None:
        """Initialize the tokenizer."""
        self.tokenizer = HuggingfaceTokenizer(BPE(continuing_subword_prefix="##"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.enable_padding(pad_token=SpecialToken.PAD, length=pad_len)
        self.max_vocab_size = max_vocab_size

    def train(self, text: Iterable[str] | str) -> None:
        """Train the tokenizer."""
        if isinstance(text, str):
            text = [text]
        trainer = BpeTrainer(
            vocab_size=self.max_vocab_size,
            special_tokens=[SpecialToken.PAD.value],
            continuing_subword_prefix="##",
        )
        self.tokenizer.train_from_iterator(text, trainer=trainer)

    def encode(self, string: str) -> list[int]:
        """Encode a string into tokens."""
        return self.tokenizer.encode(string).ids

    def decode(self, tokens: list[int]) -> str:
        """Decode tokens into a string."""
        decoded = self.tokenizer.decode(tokens)
        # Handle continuing subword prefix by removing '##' and space before it
        decoded = re.sub(r"\s*##", "", decoded)
        # Remove whitespace before special tokens
        for token in SpecialToken.to_list():
            decoded = re.sub(rf"\s+({re.escape(token)})", r"\1", decoded)
        return decoded

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @property
    def padding_token_id(self) -> int:
        """Padding token id."""
        return self.tokenizer.token_to_id(SpecialToken.PAD.value)


class BPETokenizer(Tokenizer):
    """Byte-Pair Encoding Tokenizer."""

    def __init__(self, model: str = "gpt-2") -> None:
        """Use openai tiktoken tokenizer."""
        self.tokenizer = tiktoken.encoding_for_model(model)

    def encode(self, string: str) -> list[int]:
        """Encode a string into tokens."""
        return self.tokenizer.encode(string)

    def decode(self, tokens: list[int]) -> str:
        """Decode tokens into a string."""
        return self.tokenizer.decode(tokens)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.n_vocab

    @property
    def padding_token_id(self) -> int:
        """Padding token id."""
        return -1


class BertTokenizer(BPETokenizer):
    """BERT Tokenizer: BPE Tokenizer used in GPT-2 with special tokens."""

    def __init__(self, model: str = "gpt-2") -> None:
        """Initialize BERT Tokenizer."""
        super().__init__(model)

        special_tokens = {
            token: self.tokenizer.n_vocab + i
            for i, token in enumerate(SpecialToken.to_list())
        }
        self.tokenizer = tiktoken.Encoding(
            name="gpt2_bert",
            pat_str=self.tokenizer._pat_str,
            mergeable_ranks=self.tokenizer._mergeable_ranks,
            special_tokens={**self.tokenizer._special_tokens, **special_tokens},
        )

    @property
    def padding_token_id(self) -> int:
        return self.encode(SpecialToken.PAD.value)[0]

    def encode(self, string: str) -> list[int]:
        """Encode a string into tokens with special tokens."""
        return self.tokenizer.encode(
            string, allowed_special=set(SpecialToken.to_list())
        )


class SingleCharTokenizer(Tokenizer):
    """Tokenizer for single character tokens."""

    def __init__(self) -> None:
        """Initialize SingleCharTokenizer given some text."""
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}
        self.vocab: list[str] = []

    def train(self, text: Iterable[str] | str) -> None:
        """Train the tokenizer."""
        if issubclass(text.__class__, Iterable):
            text = "".join(text)
        self.vocab = sorted(set(text)) + [SpecialToken.PAD.value]
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}

    def encode(self, string: str) -> list[int]:
        """Encode a string into tokens."""
        # Remove special tokens
        string = re.sub(
            rf'({"|".join(re.escape(v) for v in SpecialToken.to_list())})', "", string
        )
        return [self.stoi[c] for c in string]

    def decode(self, tokens: list[int]) -> str:
        """Decode tokens into a string."""
        return "".join(self.itos[t] for t in tokens)

    @property
    def padding_token_id(self) -> int:
        return self.vocab.index(SpecialToken.PAD.value)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


tokenizers: dict[str, type[Tokenizer]] = {
    "BPE": BPETokenizer,
    "BERT": BertTokenizer,
    "SingleChar": SingleCharTokenizer,
    "CustomBPE": CustomBPETokenizer,
}
