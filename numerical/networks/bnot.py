"""Spiking neural implementation of `Bitstring.bnot` operations as a `spa.Network`'s."""

import nengo
import nengo_spa as spa
import numpy as np

from .bitstring import *

__all__ = ["BNot"]


def implement_not(
    neurons_per_dimension: int, vocab: spa.Vocabulary
) -> tuple[nengo.Network, nengo.Node, nengo.Node]:
    raise NotImplementedError()


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
            self.binding_net, input, output = implement_not(
                self.neurons_per_dimension,
                self.vocab,
            )

        self.input = input
        self.output = output

        self.declare_input(self.input, self.vocab)
        self.declare_output(self.output, self.vocab)
