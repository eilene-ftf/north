"""Spiking neural implementation of `Bitstring.band` operations as a `spa.Network`'s."""

import nengo
import nengo_spa as spa
import numpy as np

from .bitstring import *

__all__ = ["BAnd"]

def implement_and(
    neurons_per_dimension: int, vocab: spa.Vocabulary
) -> tuple[nengo.Network, tuple[nengo.Node, nengo.Node], nengo.Node]:
    raise NotImplementedError()


class BAnd(spa.Network):
    """Spiking implementation of bit-wise AND, mirroring the functionality of
    of `.bitstring.band`.

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
        input_left nengo.Node:
            Left input vector.
        input_right nengo.Node:
            Right input vector.
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
        super(BAnd, self).__init__(**kwargs)

        self.vocab = vocab
        self.width = width
        self.neurons_per_dimension = neurons_per_dimension

        with self:
            self.binding_net, inputs, output = implement_and(
                self.neurons_per_dimension,
                self.vocab,
            )

        self.input_left = inputs[0]
        self.input_right = inputs[1]
        self.output = output

        self.declare_input(self.input_left, self.vocab)
        self.declare_input(self.input_right, self.vocab)
        self.declare_output(self.output, self.vocab)
