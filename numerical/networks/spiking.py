"""Spiking neural implementation of `Bitstring.band` operations as a `spa.Network`'s."""

import functools

import nengo  # type: ignore
import nengo_spa as spa  # type: ignore
import numpy as np
from numpy.fft import fft, ifft

__all__ = ["BAnd", "BOr", "BNot", "BXor"]

AND_MAPPING = {
    "BIT_1 * BIT_1": "BIT_1",
    "BIT_0 * BIT_1": "BIT_0",
    "BIT_0 * BIT_0": "BIT_0",
}


XOR_MAPPING = {
    "BIT_1 * BIT_1": "BIT_0",
    "BIT_1 * BIT_0": "BIT_1",
    "BIT_0 * BIT_0": "BIT_1",
}

OR_MAPPING = {
    "BIT_1 * BIT_1": "BIT_1",
    "BIT_1 * BIT_0": "BIT_1",
    "BIT_0 * BIT_0": "BIT_1",
}


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

    Example:
    ```python
    dim = 256
    vocab_keys = ["S_0", "S_1", "S_2", "S_3", "S_4", "S_5", "S_6", "S_7"]
    vocab = spa.Vocabulary(dim)
    vocab.populate(".normalized(); ".join(vocab_keys))
    theta = 0.8
    rng = np.random.default_rng(0)
    width = 8

    binary_5 = bit.encode(5, vocab, width=8)
    binary_8 = bit.encode(8, vocab, width=8)
    binary_0 = bit.encode(0, vocab, width=8)

    with spa.Network() as model:
        lin = spa.Transcode(lambda _: "BINARY_5", output_vocab=vocab)
        rin = spa.Transcode(lambda _: "BINARY_8", output_vocab=vocab)
        band = BAnd(vocab=vocab, width=8, seed=0)
        lin >> band.input_left
        rin >> band.input_right
        out_state = spa.State(vocab=vocab)
        band.output >> out_state
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
        self.width = width
        self.neurons_per_dimension = neurons_per_dimension

        with self:
            input_left = nengo.Node(size_in=self.vocab.dimensions, label="input_left")
            input_right = nengo.Node(size_in=self.vocab.dimensions, label="input_right")

            bits = []
            for w in range(width):
                marker = f"S_{w}"
                cconvl = spa.modules.Bind(
                    vocab=self.vocab, neurons_per_dimension=self.neurons_per_dimension
                )
                cconvr = spa.modules.Bind(
                    vocab=self.vocab, neurons_per_dimension=self.neurons_per_dimension
                )
                nengo.Connection(input_left, cconvl.input_left)
                (~self.vocab[marker]).connect_to(cconvl.input_right)

                nengo.Connection(input_right, cconvr.input_left)
                (~self.vocab[marker]).connect_to(cconvr.input_right)

                lbit = cconvl.output
                rbit = cconvr.output
                bits.append((lbit, rbit))

            flips = []
            for i, (lbit, rbit) in enumerate(bits):
                assoc = spa.WTAAssocMem(
                    threshold=0.2,
                    input_vocab=self.vocab,
                    output_vocab=self.vocab,
                    mapping=AND_MAPPING,
                    seed=self.seed,
                )
                cconvflip = spa.modules.Bind(
                    vocab=self.vocab, neurons_per_dimension=self.neurons_per_dimension
                )
                nengo.Connection(lbit, cconvflip.input_left)
                nengo.Connection(rbit, cconvflip.input_right)
                nengo.Connection(cconvflip.output, assoc.input)

                marker = f"S_{i}"
                flip = spa.State(vocab=self.vocab)
                assoc.output * self.vocab[marker] >> flip
                flips.append(flip)

            superposition = spa.Superposition(
                n_inputs=len(flips),
                vocab=self.vocab,
                neurons_per_dimension=self.neurons_per_dimension,
                seed=self.seed,
            )
            for i, f in enumerate(flips):
                f >> superposition.inputs[i]

            output = superposition.output

        self.input_left = input_left
        self.input_right = input_right
        self.output = output

        self.declare_input(self.input_left, self.vocab)
        self.declare_input(self.input_right, self.vocab)
        self.declare_output(self.output, self.vocab)


class BXor(spa.Network):
    """Spiking implementation of bit-wise XOR, mirroring the functionality of
    of `.bitstring.bxor`.

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
        self.width = width
        self.neurons_per_dimension = neurons_per_dimension

        with self:
            input_left = nengo.Node(size_in=self.vocab.dimensions, label="input_left")
            input_right = nengo.Node(size_in=self.vocab.dimensions, label="input_right")

            bits = []
            for w in range(width):
                marker = f"S_{w}"
                cconvl = spa.modules.Bind(
                    vocab=self.vocab, neurons_per_dimension=self.neurons_per_dimension
                )
                cconvr = spa.modules.Bind(
                    vocab=self.vocab, neurons_per_dimension=self.neurons_per_dimension
                )
                nengo.Connection(input_left, cconvl.input_left)
                (~self.vocab[marker]).connect_to(cconvl.input_right)

                nengo.Connection(input_right, cconvr.input_left)
                (~self.vocab[marker]).connect_to(cconvr.input_right)

                lbit = cconvl.output
                rbit = cconvr.output
                bits.append((lbit, rbit))

            flips = []
            for i, (lbit, rbit) in enumerate(bits):
                assoc = spa.WTAAssocMem(
                    threshold=0.2,
                    input_vocab=self.vocab,
                    output_vocab=self.vocab,
                    mapping=XOR_MAPPING,
                    seed=self.seed,
                )
                cconvflip = spa.modules.Bind(
                    vocab=self.vocab, neurons_per_dimension=self.neurons_per_dimension
                )
                nengo.Connection(lbit, cconvflip.input_left)
                nengo.Connection(rbit, cconvflip.input_right)
                nengo.Connection(cconvflip.output, assoc.input)

                marker = f"S_{i}"
                flip = spa.State(vocab=self.vocab)
                assoc.output * self.vocab[marker] >> flip
                flips.append(flip)

            superposition = spa.Superposition(
                n_inputs=len(flips),
                vocab=self.vocab,
                neurons_per_dimension=self.neurons_per_dimension,
                seed=self.seed,
            )
            for i, f in enumerate(flips):
                f >> superposition.inputs[i]

            output = superposition.output

        self.input_left = input_left
        self.input_right = input_right
        self.output = output

        self.declare_input(self.input_left, self.vocab)
        self.declare_input(self.input_right, self.vocab)
        self.declare_output(self.output, self.vocab)


def implement_or(
    neurons_per_dimension: int, vocab: spa.Vocabulary
) -> tuple[nengo.Network, tuple[nengo.Node, nengo.Node], nengo.Node]:
    raise NotImplementedError()


class BOr(spa.Network):
    """Spiking implementation of bit-wise XOR, mirroring the functionality of
    of `.bitstring.bxor`.

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
        self.width = width
        self.neurons_per_dimension = neurons_per_dimension

        with self:
            input_left = nengo.Node(size_in=self.vocab.dimensions, label="input_left")
            input_right = nengo.Node(size_in=self.vocab.dimensions, label="input_right")

            bits = []
            for w in range(width):
                marker = f"S_{w}"
                cconvl = spa.modules.Bind(
                    vocab=self.vocab, neurons_per_dimension=self.neurons_per_dimension
                )
                cconvr = spa.modules.Bind(
                    vocab=self.vocab, neurons_per_dimension=self.neurons_per_dimension
                )
                nengo.Connection(input_left, cconvl.input_left)
                (~self.vocab[marker]).connect_to(cconvl.input_right)

                nengo.Connection(input_right, cconvr.input_left)
                (~self.vocab[marker]).connect_to(cconvr.input_right)

                lbit = cconvl.output
                rbit = cconvr.output
                bits.append((lbit, rbit))

            flips = []
            for i, (lbit, rbit) in enumerate(bits):
                assoc = spa.WTAAssocMem(
                    threshold=0.2,
                    input_vocab=self.vocab,
                    output_vocab=self.vocab,
                    mapping=XOR_MAPPING,
                    seed=self.seed,
                )
                cconvflip = spa.modules.Bind(
                    vocab=self.vocab, neurons_per_dimension=self.neurons_per_dimension
                )
                nengo.Connection(lbit, cconvflip.input_left)
                nengo.Connection(rbit, cconvflip.input_right)
                nengo.Connection(cconvflip.output, assoc.input)

                marker = f"S_{i}"
                flip = spa.State(vocab=self.vocab)
                assoc.output * self.vocab[marker] >> flip
                flips.append(flip)

            superposition = spa.Superposition(
                n_inputs=len(flips),
                vocab=self.vocab,
                neurons_per_dimension=self.neurons_per_dimension,
                seed=self.seed,
            )
            for i, f in enumerate(flips):
                f >> superposition.inputs[i]

            output = superposition.output

        self.input_left = input_left
        self.input_right = input_right
        self.output = output

        self.declare_input(self.input_left, self.vocab)
        self.declare_input(self.input_right, self.vocab)
        self.declare_output(self.output, self.vocab)


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
        vocab spa.Vocabulary:
            The SPA vocabulary.

    Returns:
        The bit flipped at `position`, and re-bound with the marker.
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

    Example:
    ```python
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
