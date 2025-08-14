"""Spiking neural implementation of `Bitstring.bor` operations as a `spa.Network`'s."""

import nengo
import nengo_spa as spa
import numpy as np
from numpy.fft import fft, ifft
import functools

from ..bitstring import *

__all__ = ["BOr"]


__all__ = ["BNot"]


def invert(x: np.ndarray) -> np.ndarray:
    return x[np.r_[0, x.size - 1 : 0 : -1]]


def cconv(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return ifft(fft(x) * fft(y)).real


def get_bit_and_flip(
    x: np.ndarray,
    position: int,
    vocab: spa.Vocabulary,
) -> np.ndarray:
    """Get the bit at position `position`, and pass it through
    cleanup with `b_0` and `b_1`.

    Args:
        x np.ndarray:
            The `Bitstring` representation.
        position int:
            The position/index of the bit you wish to retrieve.
        b_0 spa.SemanticPointer:
            The SP for `BIT_0`.
        b_1 spa.SemanticPointer:
            The SP for `BIT_1`.

    Returns:
        The bit flipped at `position`, and rebound with the marker.
    """
    marker = vocab[f"S_{position}"].v
    b_0 = vocab["BIT_0"].v
    b_1 = vocab["BIT_1"].v
    bits = [b_0, b_1]
    bits_arr = np.array([bits[0], bits[1]])

    bit = cconv(x, invert(marker))
    sims = bits_arr @ bit
    max_sim_idx = np.argmin(sims)

    flip_bit = bits[max_sim_idx]
    flip_bit = cconv(flip_bit, marker)
    return flip_bit


class BNot(spa.Network):
    """Spiking implementation of bit-wise NOT, mirroring the functionality of
    of `.bitstring.bnot`.

    Args:
        vocab spa.Vocabulary:
            The vocabulary to use or interpret the vector.
        neurons_per_dimension int:
            Optional, number of neurons to use in each dimension. Default: `200`.
        width int:
            The `Bitstring` width of each of the two parameters.
        **kwargs dict:
            Keyword arguments passed through to `spa.Network`.

    Attributes:
        input nengo.Node:
            Input vector.
        output nengo.Node:
            The output node.
        width int:
            The `Bitstring` width.
    """

    def __init__(
        self,
        vocab: spa.Vocabulary,
        width: int,
        neurons_per_dimension: int = 200,
        **kwargs,
    ) -> None:
        super(BNot, self).__init__(**kwargs)

        self.vocab = vocab
        self.width = width
        self.neurons_per_dimension = neurons_per_dimension

        with self:
            input = nengo.Node(size_in=vocab.dimensions, label="input")

            in_ens = nengo.Ensemble(self.neurons_per_dimension, self.vocab.dimensions)
            nengo.Connection(input, in_ens)

            bits = []
            resymbolizes = []
            for i in range(self.width):
                bit = nengo.Node(size_in=vocab.dimensions, label=f"bit_at_{i}")
                get_bit_and_flip_at_i = functools.partial(
                    get_bit_and_flip,
                    position=i,
                    vocab=self.vocab,
                )
                nengo.Connection(in_ens, bit, function=get_bit_and_flip_at_i)
                bits.append(bit)
                resymbolize = spa.State(
                    vocab=self.vocab,
                    neurons_per_dimension=self.neurons_per_dimension,
                    label=f"resymbolize_at_{i}",
                )
                nengo.Connection(bit, resymbolize.input)
                resymbolizes.append(resymbolize)
            result = spa.Superposition(n_inputs=len(resymbolizes), vocab=self.vocab)

            for i, res in enumerate(resymbolizes):
                resymbolizes[i] >> result.inputs[i]

            output = result.output

        self.input = input
        self.output = output

        self.declare_input(self.input, self.vocab)
        self.declare_output(self.output, self.vocab)
