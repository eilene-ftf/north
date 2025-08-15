"""Algebraic implementation of bit-wise operations. For more information
about the specific operations, see ``..bitsring.Bitstring``.
"""

import functools

import nengo  # type: ignore
import nengo_spa as spa  # type: ignore
import numpy as np


class BXor(spa.Network):
    """Algebraic implementation of bit-wise XOR.

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
        self, vocab: spa.Vocabulary, neurons_per_dimension: int, width: int, **kwargs
    ) -> None:
        super(BXor, self).__init__(**kwargs)

        self.vocab = vocab
        self.neurons_per_dimension = neurons_per_dimension
        self.width = width

        with self:
            input_left = nengo.Node(
                size_in=self.vocab.dimensions, seed=self.seed, label="input_left"
            )
            input_right = nengo.Node(
                size_in=self.vocab.dimensions, seed=self.seed, label="input_right"
            )

            # TODO: implement logic

            output = nengo.Node(
                size_in=self.vocab.dimensions, seed=self.seed, label="output"
            )

        self.input_left = input_left
        self.input_right = input_right
        self.output = output

        self.declare_input(self.input_left, self.vocab)
        self.declare_input(self.input_right, self.vocab)
        self.declare_output(self.output, self.vocab)


class BAnd(spa.Network):
    """Algebraic implementation of bit-wise AND.

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
        self, vocab: spa.Vocabulary, neurons_per_dimension: int, width: int, **kwargs
    ) -> None:
        super(BAnd, self).__init__(**kwargs)

        self.vocab = vocab
        self.neurons_per_dimension = neurons_per_dimension
        self.width = width

        with self:
            input_left = nengo.Node(
                size_in=self.vocab.dimensions, seed=self.seed, label="input_left"
            )
            input_right = nengo.Node(
                size_in=self.vocab.dimensions, seed=self.seed, label="input_right"
            )

            # TODO: implement logic

            output = nengo.Node(
                size_in=self.vocab.dimensions, seed=self.seed, label="output"
            )

        self.input_left = input_left
        self.input_right = input_right
        self.output = output

        self.declare_input(self.input_left, self.vocab)
        self.declare_input(self.input_right, self.vocab)
        self.declare_output(self.output, self.vocab)


class BOr(spa.Network):
    """Algebraic implementaiton of bit-wise OR.

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
        self, vocab: spa.Vocabulary, neurons_per_dimension: int, width: int, **kwargs
    ) -> None:
        super(BOr, self).__init__(**kwargs)

        self.vocab = vocab
        self.neurons_per_dimension = neurons_per_dimension
        self.width = width

        with self:
            input_left = nengo.Node(
                size_in=self.vocab.dimensions, seed=self.seed, label="input_left"
            )
            input_right = nengo.Node(
                size_in=self.vocab.dimensions, seed=self.seed, label="input_right"
            )

            # TODO: implement logic

            output = nengo.Node(
                size_in=self.vocab.dimensions, seed=self.seed, label="output"
            )

        self.input_left = input_left
        self.input_right = input_right
        self.output = output

        self.declare_input(self.input_left, self.vocab)
        self.declare_input(self.input_right, self.vocab)
        self.declare_output(self.output, self.vocab)


class BNot(spa.Network):
    """Algebriac implementation of bit-wise flipping.

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
            Left input vector.
        output nengo.Node:
            The output node.
        width int:
            The `Bitstring` width.
    """

    def __init__(
        self, vocab: spa.Vocabulary, neurons_per_dimension: int, width: int, **kwargs
    ) -> None:
        super(BNot, self).__init__(**kwargs)

        self.vocab = vocab
        self.neurons_per_dimension = neurons_per_dimension
        self.width = width

        with self:
            input = nengo.Node(
                size_in=self.vocab.dimensions, seed=self.seed, label="input"
            )

            # TODO: implement logic

            output = nengo.Node(
                size_in=self.vocab.dimensions, seed=self.seed, label="output"
            )

        self.input = input
        self.output = output

        self.declare_input(self.input, self.vocab)
        self.declare_output(self.output, self.vocab)
