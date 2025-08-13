"""Bitstring encoding of numbers, using role-filler pairs."""

import typing

import nengo_spa as spa
import numpy as np
from embeddings import random


class Bitstring(spa.SemanticPointer):
    """Thin wrapper around Semantic Pointers, specifically with methods
    supporting the twos-complement encoding of integers.
    """

    def __init__(
        self,
        data: np.ndarray,
        vocabulary: spa.Vocabulary,
        name: typing.Optional[str] = None,
    ) -> None:
        super(Bitstring, self).__init__(data, vocab=vocabulary, name=name)

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

    def lt(self, other: "Bitstring") -> float:
        raise NotImplementedError()

    def gt(self, other: "Bitstring") -> float:
        raise NotImplementedError()


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

    n_bin = bin(n)
    if len(n_bin) < width:
        for _ in range(width - len(n_bin)):
            n_bin = "0" + n_bin

    brep = np.zeros(vocabulary.dimensions)
    for i, b in enumerate(n_bin):
        if b == "0":
            brep += (vocabulary[f"S_{i}"] * vocabulary["BIT_0"])
        else:
            brep += (vocabulary[f"S_{i}"] * vocabulary["BIT_1"])
    
    brep_name = f"BINARY_{n}"
    sp = spa.SemanticPointer(brep, vocab=vocabulary, name=brep_name)
    vocabulary.add(brep_name, sp)
    return sp