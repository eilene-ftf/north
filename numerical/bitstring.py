"""Bitstring encoding of numbers, using role-filler pairs."""

import typing
import sys
import functools

import nengo_spa as spa  # type: ignore
import numpy as np


__all__ = ["Bitstring", "encode"]


def random(
    num_vectors: int, dim: int, dtype=float, rng=np.random.default_rng()
) -> np.ndarray:
    """Create randomly sampled matrix of ``(num_vectors, dim)``.

    Args:
        num_vectors int: The number of vectors to sample.
        dim: int: The dimensionality of the vectors.
        dtype float: Optional, defaults to ``float``.
        rng: Optional, defaults to ``np.random.default_rng()``.

    Returns:
        ``(num_vectors, dim)`` randomly sampled matrix from from the normalized
        ``np.random.normal(dim, sd=1/np.sqrt(dim)``.
    """
    sd = 1.0 / np.sqrt(dim)
    vs = rng.normal(scale=sd, size=(num_vectors, dim)).astype(dtype)
    norms = np.linalg.vector_norm(vs, axis=1, keepdims=True)
    vs /= norms
    return vs


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
        """Bit-wise *and* between two `Bitstring`'s.

        Args:
            other Bitstring: the other argument.

        Returns:
            `self & other`.
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
            vocab=self.vocab,
            width=self.width,
            lname=self.name,
            rname=other.name,
        )

    def bor(self, other: "Bitstring") -> "Bitstring":
        """Bit-wise *or* between two `Bitstring`'s.

        Args:
            other Bitstring: the other argument.

        Returns:
            `self | other`.
        """
        if not self.check_same_width(other):
            raise ValueError(
                f"Expected same width, got lhs {self.width} rhs {other.width}"
            )

        sbits = self.get_bits()
        obits = other.get_bits()
        return bitwise_or(
            sbits,
            obits,
            vocab=self.vocab,
            width=self.width,
            lname=self.name,
            rname=other.name,
        )

    def bxor(self, other: "Bitstring") -> "Bitstring":
        """Bit-wise *or* between two `Bitstring`'s.

        Args:
            other Bitstring: the other argument.

        Returns:
            `self ^ other`.
        """
        sbits = self.get_bits()
        obits = other.get_bits()
        return bitwise_xor(
            sbits,
            obits,
            vocab=self.vocab,
            width=self.width,
            lname=self.name,
            rname=other.name,
        )

    def bnot(self) -> "Bitstring":
        """Bit-wise *negation* of a `Bitstring`."""
        sbits = self.get_bits()
        return bitwise_not(sbits, vocab=self.vocab, width=self.width, name=self.name)

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

    def decode(self) -> int:
        """Decode a ``Bitstring`` to an int."""
        bits = self.get_bits()
        base_bits = [self.vocab["BIT_0"], self.vocab["BIT_1"]]
        bits_mat = np.array([ptr.v for ptr in base_bits])
        decoded_bits = []
        for bit in bits:
            sims = bits_mat @ bit.v
            max_sim_idx = np.argmax(sims)
            decoded_bits.append(str(max_sim_idx))
        decoded_str = "".join(decoded_bits)
        decoded_str = "0b" + decoded_str
        return int(decoded_str, base=2)


def encode(
    n: int,
    vocabulary: spa.Vocabulary,
    width: int = 8,
    rng=np.random.default_rng(),
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
                random(1, vocabulary.dimensions, rng=rng).squeeze(),
                vocab=vocabulary,
                name=f"S_{i}",
            )
            vocabulary.add(f"S_{i}", s_i)
    if "BIT_0" not in vocabulary:
        bit_0 = spa.SemanticPointer(
            random(1, vocabulary.dimensions, rng=rng).squeeze(),
            vocab=vocabulary,
            name="BIT_0",
        )
        vocabulary.add("BIT_0", bit_0)
    if "BIT_1" not in vocabulary:
        bit_1 = spa.SemanticPointer(
            random(1, vocabulary.dimensions, rng=rng).squeeze(),
            vocab=vocabulary,
            name="BIT_0",
        )
        vocabulary.add("BIT_1", bit_1)

    n_bin = bin(n).removeprefix("0b")
    if len(n_bin) < width:
        for _ in range(width - len(n_bin)):
            n_bin = "0" + n_bin
    print(n_bin)

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
    if brep_name not in vocabulary:
        vocabulary.add(brep_name, sp)
    return sp


def from_list(
    sps: list[spa.SemanticPointer],
    vocab: spa.Vocabulary,
    width: int,
    name: typing.Optional[str] = None,
) -> Bitstring:
    """Convert a list of semantic pointers into a Bitstring of with width ``width``.

    Args:
        sps list[spa.SemanticPointer]: The list of semantic pointers.
        vocab spa.Vocabulary: The vocabulary used.
        width int: The width of the integer representation.
        name Optional[str]: The new name for the Bitstring. Default: ``None``.

    Returns:
        The Bitstring encoding.
    """
    if width < len(sps):
        print(
            "WARNING: decreasing width of representation may lead to loss of information",
            file=sys.stderr,
        )

    width_markers = []
    for i in range(width):
        width_markers.append(vocab[f"S_{i}"])

    brep = np.zeros(vocab.dimensions)
    for i, marker in enumerate(width_markers):
        if i < len(sps):
            brep += (marker * sps[i]).v
        else:
            brep += (marker * vocab["B_0"]).v

    brep_name = f"BINARY_{name}" if name else None
    sp = Bitstring(brep, vocab=vocab, name=brep_name, width=width)
    if brep_name and brep_name not in vocab:
        vocab.add(brep_name, sp)
    return sp


def bitwise_and(
    lhs_bits: list[spa.SemanticPointer],
    rhs_bits: list[spa.SemanticPointer],
    vocab: spa.Vocabulary,
    width: int,
    theta: float = 0.8,
    lname: typing.Optional[str] = None,
    rname: typing.Optional[str] = None,
) -> Bitstring:
    """Bit-wise conjunction between two lists of semantic pointers.

    Args:
        lhs_bits, rhs_bits (list[spa.SemanticPointer]):
            The bit pairs of the left and right hand side of the operation.
        vocab spa.Vocabulary: The vocabulary
        width int: The width of the encoding
        theta float: Optional, for approximate comparison. Default: ``0.8``.
        lname Optional[str]: Optional name of the left-hand side.
        rname Optional[str]: Optional name of the right-hand side.

    Returns:
        The component-wise conjunction of the two represented values as a new
        `Bitstring`.
    """
    bs = []
    for lhs, rhs in zip(lhs_bits, rhs_bits):
        if lhs.compare(vocab["BIT_1"]) > theta and rhs.compare(vocab["BIT_1"]) > theta:
            bs.append(vocab["BIT_1"])
        else:
            bs.append(vocab["BIT_0"])

    if lname is not None and rname is not None:
        new_name = str(
            int(lname.removeprefix("BINARY_")) & int(rname.removeprefix("BINARY_"))
        )
        return from_list(bs, vocab, width, name=new_name)
    else:
        return from_list(bs, vocab, width)


def bitwise_or(
    lhs_bits: list[spa.SemanticPointer],
    rhs_bits: list[spa.SemanticPointer],
    vocab: spa.Vocabulary,
    width: int,
    theta: float = 0.8,
    lname: typing.Optional[str] = None,
    rname: typing.Optional[str] = None,
) -> Bitstring:
    """Bit-wise disjunction between two lists of semantic pointers.

    Args:
        lhs_bits, rhs_bits (list[spa.SemanticPointer]):
            The bit pairs of the left and right hand side of the operation.
        vocab spa.Vocabulary: The vocabulary
        width int: The width of the encoding
        theta float: Optional, for approximate comparison. Default: ``0.8``.
        lname Optional[str]: Optional name of the left-hand side.
        rname Optional[str]: Optional name of the right-hand side.

    Returns:
        The component-wise disjunction of the two represented values as a new
        `Bitstring`.
    """
    bs = []
    for lhs, rhs in zip(lhs_bits, rhs_bits):
        if lhs.compare(vocab["BIT_1"]) > theta or rhs.compare(vocab["BIT_1"]) > theta:
            bs.append(vocab["BIT_1"])
        else:
            bs.append(vocab["BIT_0"])

    if lname is not None and rname is not None:
        new_name = str(
            int(lname.removeprefix("BINARY_")) & int(rname.removeprefix("BINARY_"))
        )
        return from_list(bs, vocab, width, name=new_name)
    else:
        return from_list(bs, vocab, width)


def bitwise_xor(
    lhs_bits: list[spa.SemanticPointer],
    rhs_bits: list[spa.SemanticPointer],
    vocab: spa.Vocabulary,
    width: int,
    theta: float = 0.8,
    lname: typing.Optional[str] = None,
    rname: typing.Optional[str] = None,
) -> Bitstring:
    """Bit-wise xor between two lists of semantic pointers.

    Args:
        lhs_bits, rhs_bits (list[spa.SemanticPointer]):
            The bit pairs of the left and right hand side of the operation.
        vocab spa.Vocabulary: The vocabulary
        width int: The width of the encoding
        theta float: Optional, for approximate comparison. Default: ``0.8``.
        lname Optional[str]: Optional name of the left-hand side.
        rname Optional[str]: Optional name of the right-hand side.

    Returns:
        The component-wise xor of the two represented values as a new
        `Bitstring`.
    """
    bs = []
    for lhs, rhs in zip(lhs_bits, rhs_bits):
        if (
            lhs.compare(vocab["BIT_1"]) > theta
            and (not rhs.compare(vocab["BIT_1"]) > theta)
        ) or (
            (not lhs.compare(vocab["BIT_1"]) > theta)
            and rhs.compare(vocab["BIT_1"]) > theta
        ):
            bs.append(vocab["BIT_1"])
        else:
            bs.append(vocab["BIT_0"])

    if lname is not None and rname is not None:
        new_name = str(
            int(lname.removeprefix("BINARY_")) & int(rname.removeprefix("BINARY_"))
        )
        return from_list(bs, vocab, width, name=new_name)
    else:
        return from_list(bs, vocab, width)


def bitwise_not(
    bits: list[spa.SemanticPointer],
    vocab: spa.Vocabulary,
    width: int,
    theta: float = 0.8,
    name: typing.Optional[str] = None,
) -> Bitstring:
    """Bit-wise negation between two lists of semantic pointers.

    Args:
        bits (list[spa.SemanticPointer]): Bitstring to invert.
        vocab spa.Vocabulary: The vocabulary
        width int: The width of the encoding
        theta float: Optional, for approximate comparison. Default: ``0.8``.
        lname Optional[str]: Optional name of the left-hand side.
        rname Optional[str]: Optional name of the right-hand side.

    Returns:
        The component-wise xor of the two represented values as a new
        `Bitstring`.
    """
    bs = []
    for bit in bits:
        if bit.compare(vocab["BIT_1"]) > theta:
            bs.append(vocab["BIT_0"])
        else:
            bs.append(vocab["BIT_1"])
    if name is not None:
        new_name = str(-~int(name.removeprefix("BINARY_")))
        return from_list(bs, vocab, width, name=new_name)
    else:
        return from_list(bs, vocab, width)
