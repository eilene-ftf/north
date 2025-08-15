"""Algebraic implementation of bit-wise operations. For more information
about the specific operations, see ``..bitsring.Bitstring``.
"""

import functools

import nengo  # type: ignore
import nengo_spa as spa  # type: ignore
import numpy as np

from embeddings import random

from ..bitstring import Bitstring

__all__ = ["BXor", "BAnd", "BNot", "BOr"]

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
        self,
        vocab: spa.Vocabulary,
        width: int,
        neurons_per_dimension: int = 200,
        **kwargs,
    ) -> None:
        super(BXor, self).__init__(**kwargs)

        self.vocab = vocab

        self.neurons_per_dimension = neurons_per_dimension
        self.width = width

        if "S_LEFT" not in self.vocab:
            v = random(1, self.vocab.dimensions).squeeze()
            sp = spa.SemanticPointer(
                v, vocab=self.vocab, algebra=self.vocab.algebra, name="S_LEFT"
            )
            self.vocab.add("S_LEFT", sp)

        if "S_RIGHT" not in self.vocab:
            v = random(1, self.vocab.dimensions).squeeze()
            sp = spa.SemanticPointer(
                v, vocab=self.vocab, algebra=self.vocab.algebra, name="S_RIGHT"
            )
            self.vocab.add("S_RIGHT", sp)

        with self:
            input_left = nengo.Node(size_in=self.vocab.dimensions, label="input_left")
            input_right = nengo.Node(size_in=self.vocab.dimensions, label="input_right")

            output = nengo.Node(size_in=self.vocab.dimensions, label="output")

            cconvl = spa.Bind(
                vocab=self.vocab, neurons_per_dimension=self.neurons_per_dimension
            )
            cconvr = spa.Bind(
                vocab=self.vocab, neurons_per_dimension=self.neurons_per_dimension
            )
            spose = spa.Superposition(
                2, vocab=self.vocab, neurons_per_dimension=self.neurons_per_dimension
            )

            nengo.Connection(input_left, cconvl.input_left)
            self.vocab["S_LEFT"] >> cconvl.input_right

            nengo.Connection(input_right, cconvr.input_left)
            self.vocab["S_RIGHT"] >> cconvr.input_right

            cconvl.output >> spose.inputs[0]
            cconvr.output >> spose.inputs[1]

            # TODO: implement logic
            fn = spa.Transcode(
                function=lambda _, x: self.bxor(x),
                input_vocab=self.vocab,
                output_vocab=self.vocab,
            )
            spose.output >> fn.input
            nengo.Connection(fn.output, output)

        self.input_left = input_left
        self.input_right = input_right
        self.output = output

        self.declare_input(self.input_left, self.vocab)
        self.declare_input(self.input_right, self.vocab)
        self.declare_output(self.output, self.vocab)

    def bxor(self, x: spa.SemanticPointer) -> spa.SemanticPointer:
        lhs = Bitstring(
            x.bind(~self.vocab["S_LEFT"]), vocab=self.vocab, width=self.width
        )
        rhs = Bitstring(
            x.bind(~self.vocab["S_RIGHT"]), vocab=self.vocab, width=self.width
        )
        return lhs.band(rhs)


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

    Example:
    ```python
    import nengo
    import nengo_spa as spa
    import numerical as bit
    from numerical.networks.algebraic import BAnd

    dim = 256
    vocab_keys = ["S_0", "S_1", "S_2", "S_3", "S_4", "S_5", "S_6", "S_7"]
    vocab = spa.Vocabulary(dim)
    vocab.populate(".normalized(); ".join(vocab_keys))
    theta = 0.8
    rng = np.random.default_rng(0)
    width = 8

    b_five = bit.encode(5, vocab, width=width)
    b_four = bit.encode(4, vocab, width=width)

    with spa.Network(seed=0) as model:
        input_left = spa.Transcode(lambda _: "BINARY_5", output_vocab=vocab)
        input_right = spa.Transcode(lambda _: "BINARY_4", output_vocab=vocab)
        bit_flip = BAnd(vocab=vocab, width=8, seed=0)
        input_left >> bit_flip.input_left
        input_right >> bit_flip.input_right
        out_state = spa.State(vocab=vocab)
        bit_flip.output >> out_state
    ```
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

        self.neurons_per_dimension = neurons_per_dimension
        self.width = width

        if "S_LEFT" not in self.vocab:
            v = random(1, self.vocab.dimensions).squeeze()
            sp = spa.SemanticPointer(
                v, vocab=self.vocab, algebra=self.vocab.algebra, name="S_LEFT"
            )
            self.vocab.add("S_LEFT", sp)

        if "S_RIGHT" not in self.vocab:
            v = random(1, self.vocab.dimensions).squeeze()
            sp = spa.SemanticPointer(
                v, vocab=self.vocab, algebra=self.vocab.algebra, name="S_RIGHT"
            )
            self.vocab.add("S_RIGHT", sp)

        with self:
            input_left = nengo.Node(size_in=self.vocab.dimensions, label="input_left")
            input_right = nengo.Node(size_in=self.vocab.dimensions, label="input_right")

            output = nengo.Node(size_in=self.vocab.dimensions, label="output")

            cconvl = spa.Bind(
                vocab=self.vocab, neurons_per_dimension=self.neurons_per_dimension
            )
            cconvr = spa.Bind(
                vocab=self.vocab, neurons_per_dimension=self.neurons_per_dimension
            )
            spose = spa.Superposition(
                2, vocab=self.vocab, neurons_per_dimension=self.neurons_per_dimension
            )

            nengo.Connection(input_left, cconvl.input_left)
            self.vocab["S_LEFT"] >> cconvl.input_right

            nengo.Connection(input_right, cconvr.input_left)
            self.vocab["S_RIGHT"] >> cconvr.input_right

            cconvl.output >> spose.inputs[0]
            cconvr.output >> spose.inputs[1]

            # TODO: implement logic
            fn = spa.Transcode(
                function=lambda _, x: self.band(x),
                input_vocab=self.vocab,
                output_vocab=self.vocab,
            )
            spose.output >> fn.input
            nengo.Connection(fn.output, output)

        self.input_left = input_left
        self.input_right = input_right
        self.output = output

        self.declare_input(self.input_left, self.vocab)
        self.declare_input(self.input_right, self.vocab)
        self.declare_output(self.output, self.vocab)

    def band(self, x: spa.SemanticPointer) -> spa.SemanticPointer:
        lhs = Bitstring(
            x.bind(~self.vocab["S_LEFT"]), vocab=self.vocab, width=self.width
        )
        rhs = Bitstring(
            x.bind(~self.vocab["S_RIGHT"]), vocab=self.vocab, width=self.width
        )
        return lhs.band(rhs)


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
        self,
        vocab: spa.Vocabulary,
        width: int,
        neurons_per_dimension: int = 200,
        **kwargs,
    ) -> None:
        super(BOr, self).__init__(**kwargs)

        self.vocab = vocab

        self.neurons_per_dimension = neurons_per_dimension
        self.width = width

        if "S_LEFT" not in self.vocab:
            v = random(1, self.vocab.dimensions).squeeze()
            sp = spa.SemanticPointer(
                v, vocab=self.vocab, algebra=self.vocab.algebra, name="S_LEFT"
            )
            self.vocab.add("S_LEFT", sp)

        if "S_RIGHT" not in self.vocab:
            v = random(1, self.vocab.dimensions).squeeze()
            sp = spa.SemanticPointer(
                v, vocab=self.vocab, algebra=self.vocab.algebra, name="S_RIGHT"
            )
            self.vocab.add("S_RIGHT", sp)

        with self:
            input_left = nengo.Node(size_in=self.vocab.dimensions, label="input_left")
            input_right = nengo.Node(size_in=self.vocab.dimensions, label="input_right")

            output = nengo.Node(size_in=self.vocab.dimensions, label="output")

            cconvl = spa.Bind(
                vocab=self.vocab, neurons_per_dimension=self.neurons_per_dimension
            )
            cconvr = spa.Bind(
                vocab=self.vocab, neurons_per_dimension=self.neurons_per_dimension
            )
            spose = spa.Superposition(
                2, vocab=self.vocab, neurons_per_dimension=self.neurons_per_dimension
            )

            nengo.Connection(input_left, cconvl.input_left)
            self.vocab["S_LEFT"] >> cconvl.input_right

            nengo.Connection(input_right, cconvr.input_left)
            self.vocab["S_RIGHT"] >> cconvr.input_right

            cconvl.output >> spose.inputs[0]
            cconvr.output >> spose.inputs[1]

            # TODO: implement logic
            fn = spa.Transcode(
                function=lambda _, x: self.bor(x),
                input_vocab=self.vocab,
                output_vocab=self.vocab,
            )
            spose.output >> fn.input
            nengo.Connection(fn.output, output)

        self.input_left = input_left
        self.input_right = input_right
        self.output = output

        self.declare_input(self.input_left, self.vocab)
        self.declare_input(self.input_right, self.vocab)
        self.declare_output(self.output, self.vocab)

    def bor(self, x: spa.SemanticPointer) -> spa.SemanticPointer:
        lhs = Bitstring(
            x.bind(~self.vocab["S_LEFT"]), vocab=self.vocab, width=self.width
        )
        rhs = Bitstring(
            x.bind(~self.vocab["S_RIGHT"]), vocab=self.vocab, width=self.width
        )
        return lhs.band(rhs)


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

    Examples:
    ```python
    import nengo
    import nengo_spa as spa
    import numerical as bit
    from numerical.networks.algebraic import BNot

    dim = 256
    vocab_keys = ["S_0", "S_1", "S_2", "S_3", "S_4", "S_5", "S_6", "S_7"]
    vocab = spa.Vocabulary(dim)
    vocab.populate(".normalized(); ".join(vocab_keys))
    theta = 0.8
    rng = np.random.default_rng(0)
    width = 8

    b_zero = bit.encode(0, vocab, width=width)
    b_flip = bit.encode(255, vocab, width=width)
    b_two = bit.encode(2, vocab, width=width)

    with spa.Network(seed=0) as model:
        input = spa.Transcode(lambda _: "BINARY_255", output_vocab=vocab)
        bit_flip = BNot(vocab=vocab, width=8, seed=0)
        input >> bit_flip.input
        out_state = spa.State(vocab=vocab)
        bit_flip.output >> out_state
    ```
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
        self.neurons_per_dimension = neurons_per_dimension
        self.width = width

        with self:
            input = nengo.Node(size_in=self.vocab.dimensions, label="input")

            bitflip = spa.Transcode(
                function=lambda t, x: Bitstring.bnot(
                    Bitstring(x, self.vocab, self.width)
                ),
                input_vocab=self.vocab,
                output_vocab=self.vocab,
            )
            nengo.Connection(input, bitflip.input)

            output = nengo.Node(size_in=self.vocab.dimensions, label="output")

            nengo.Connection(bitflip.output, output)

        self.input = input
        self.output = output

        self.declare_input(self.input, self.vocab)
        self.declare_output(self.output, self.vocab)
