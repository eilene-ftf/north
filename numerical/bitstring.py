"""Bitstring encoding of numbers, using role-filler pairs."""

import typing
import sys

import nengo_spa as spa
import numpy as np

from embeddings import random

__all__ = ["Bitstring", "encode"]


class Bitstring(spa.SemanticPointer):
    """Thin wrapper around Semantic Pointers, specifically with methods
    supporting the twos-complement encoding of integers.
    """

    def __init__(
        self,
        data: np.ndarray,
        vocab: spa.Vocabulary,
        width: int,
        name: typing.Optional[str] = None,
    ) -> None:
        super(Bitstring, self).__init__(data, vocab=vocab, name=name)
        self.width = width

    def badd(self, other: "Bitstring") -> "Bitstring":
        raise NotImplementedError()

    def bsub(self, other: "Bitstring") -> "Bitstring":
        raise NotImplementedError()

    def bmul(self, other: "Bitstring") -> "Bitstring":
        raise NotImplementedError()

    def bdiv(self, other: "Bitstring") -> "Bitstring":
        raise NotImplementedError()

    def lshift(self, other: "Bitstring") -> "Bitstring":
        raise NotImplementedError()

    def rshift(self, other: "Bitstring") -> "Bitstring":
        raise NotImplementedError()

    def lt(self, other: "Bitstring") -> "Bitstring":
        raise NotImplementedError()

    def gt(self, other: "Bitstring") -> "Bitstring":
        raise NotImplementedError()

    def band(self, other: "Bitstring") -> "Bitstring":
        """Bit-wise binary and between two Bitstrings.

        Args:
            other Bitstring: the other argument.

        Returns:
            `self XOR other`.
        """
        if not self.check_same_width(other):
            raise ValueError(
                f"Expected same width, got lhs {self.width} rhs {other.width}"
            )

        sbits = self.get_bits()
        obits = other.get_bits()
        return bitwise_and(
            sbits,
            obits,
            b_0=self.vocab["BIT_0"],
            b_1=self.vocab["BIT_1"],
        )

    def bor(self, other: "Bitstring") -> "Bitstring":
        raise NotImplementedError()

    def bxor(self, other: "Bitstring") -> "Bitstring":
        raise NotImplementedError()

    def bnot(self) -> "Bitstring":
        raise NotImplementedError()

    def check_same_width(self, other: "Bitstring") -> bool:
        return self.width == other.width

    def get_bits(self) -> list[spa.SemanticPointer]:
        width = self.width
        width_markers = []
        for i in range(width):
            width_markers.append(self.vocab[f"S_{i}"])

        base_bits = [self.vocab["BIT_0"], self.vocab["BIT_1"]]
        bits_mat = np.array([ptr.v for ptr in base_bits])
        bits = []
        for marker in width_markers:
            bit = self.bind(~marker)
            sims = bits_mat @ bit.v
            max_sim_idx = np.argmax(sims)
            clean_bit = base_bits[max_sim_idx]
            bits.append(clean_bit)
        return bits


def encode(
    n: int,
    vocabulary: spa.Vocabulary,
    width: int = 8,
) -> Bitstring:
    """Encode an integer into a `Bitstring` high-dimensional vector, of width
    `width`.

    WARNING: reserves the following names in the vocabulary:
    + `f"S_{i}" for i in range(width)`,
    + `BIT_0` and `BIT_1`.

    Args:
        n int: The integer to encode.
        vocabulary spa.Vocabulary: The SPA vocabulary to use.
        width int: Optional, the bit width of the encoding. Default: `8`.

    Returns:
        The `Bitstring` encoding.

    Example:
    ```python
    import nengo_spa as spa
    import numbers as num

    dim = 256
    vocab_keys = ["S_0", "S_1", "S_2", "S_3", "S_4", "S_5", "S_6", "S_7"]
    vocab = spa.Vocabulary(dim)
    vocab.populate(";".join(vocab_keys))

    b_one = num.encode(2, vocab)
    # => (S_0 * BIT_0) + (S_1 * BIT_0) + (S_2 * BIT_0) + (S_3 * BIT_0) + (S_4 * BIT_0) + (S_5 * BIT_0) + (S_6 * BIT_1) + (S_7 * BIT_0)
    # representing: 0000010
    ```
    """

    for i in range(width):
        if f"S_{i}" not in vocabulary:
            s_i = spa.SemanticPointer(
                random(1, vocabulary.dimensions).squeeze(),
                vocab=vocabulary,
                name=f"S_{i}",
            )
            vocabulary.add(f"S_{i}", s_i)
    if "BIT_0" not in vocabulary:
        bit_0 = spa.SemanticPointer(
            random(1, vocabulary.dimensions).squeeze(),
            vocab=vocabulary,
            name="BIT_0",
        )
        vocabulary.add("BIT_0", bit_0)
    if "BIT_1" not in vocabulary:
        bit_1 = spa.SemanticPointer(
            random(1, vocabulary.dimensions).squeeze(),
            vocab=vocabulary,
            name="BIT_0",
        )
        vocabulary.add("BIT_1", bit_1)

    n_bin = bin(n).replace("b", "")
    print(f"{n_bin = }", file=sys.stderr)
    if len(n_bin) < width:
        for _ in range(width - len(n_bin)):
            n_bin = "0" + n_bin

    brep = np.zeros(vocabulary.dimensions)
    for i, b in enumerate(n_bin):
        if b == "0":
            brep += (vocabulary[f"S_{i}"] * vocabulary["BIT_0"]).v
        elif i == "b":
            continue
        else:
            brep += (vocabulary[f"S_{i}"] * vocabulary["BIT_1"]).v

    brep_name = f"BINARY_{n}"
    sp = Bitstring(brep, vocab=vocabulary, name=brep_name, width=width)
    vocabulary.add(brep_name, sp)
    return sp


def bitwise_and(
    lhs_bits: list[spa.SemanticPointer],
    rhs_bits: list[spa.SemanticPointer],
    b_0: spa.SemanticPointer,
    b_1: spa.SemanticPointer,
) -> Bitstring:
    """Bit-wise"""
    raise NotImplementedError()
