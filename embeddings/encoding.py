"""Encoding functions converting source-level representations to high-dimensional
vectors. For more information about the vector class `HRR`, see
`hrr.py`.
"""

import sys
import typing
from collections import UserDict
from dataclasses import dataclass
from pathlib import Path

import nengo_spa as spa
import numpy as np
import numpy.fft as fft

from .lex import Word, WordType, wordtype2str

import numerical as num

__all__ = [
    "HeteroAssoc",
    "AutoAssoc",
    "random",
    "Codebook",
    "EncodingEnvironment",
    "encode",
    "savez",
    "cconv",
    "invert",
    "get_name",
]

_FileLike = typing.Union[str, Path]


class HeteroAssoc:
    """Hetero-associative memory for real-valued vectors.

    This hetero-associative memory differs from the implementation in the
    machine, as it provides methods for storing the memory state
    using the functionality of [`numpy.savez`](https://numpy.org/doc/stable/reference/generated/numpy.savez.html).

    Attributes:
        dim_A int: The dimensionality of vectors stored in the address matrix (``self.A``).
        dim_P int: The dimensionality of vectors stored in the pattern matrix (``self.P``).
        capacity int: The current capacity of the matrices, this is
            the ``N`` in their dimensionality. Dynamically sized.
        stored_traces int: The number of non-zero vectors stored in the
            address and pattern matrices.
        A np.ndarray: A `(N, dim_A)` matrix storing *addresses*.
        P np.ndarray: P `(N, dim_P)` matrix storing *patterns*.
    """

    def __init__(
        self, dim_A: int, dim_P: int | None = None, initial_capacity: int = 100
    ) -> None:
        self.dim_A = dim_A
        if dim_P is not None:
            self.dim_P = dim_P
        else:
            self.dim_P = self.dim_A
        self.capacity = initial_capacity
        self.stored_traces = 0

        self.A = np.zeros((self.capacity, self.dim_A))
        self.P = np.zeros((self.capacity, self.dim_P))

    def write(self, x: np.ndarray, y: np.ndarray) -> None:
        """Associate `(x, y)` in memory, returning the values.

        Args:
            x np.ndarray: ``(self.dim_A,)`` (or convertable) vector to store
                as the address.
            y np.ndarray: ``(self.dim_P,)`` (or convertable) vector to store
                as the address.
        """
        if len(x.shape) > 1:
            x = x.squeeze()
        if len(y.shape) > 1:
            y = y.squeeze()

        # Check to see if the value is already stored in memory
        sims_A = self.A @ x
        sims_A_max_idx = np.argmax(sims_A)
        threshold = 0.8
        if sims_A[sims_A_max_idx] > threshold:
            self.A[sims_A_max_idx, :] = y
            return

        # Otherwise, store the trace as a new row
        if self.stored_traces >= self.capacity:
            self.A = np.concatenate([self.A, np.zeros((self.capacity, self.dim_A))])  # type: ignore
            self.P = np.concatenate([self.P, np.zeros((self.capacity, self.dim_P))])  # type: ignore
            self.capacity *= 2
        self.A[self.stored_traces, :] = x
        self.P[self.stored_traces, :] = y
        self.stored_traces += 1

    def memorize(self, x: np.ndarray, y: np.ndarray) -> None:
        """Alias for ``self.write``."""
        self.write(x, y)

    def read(self, x: np.ndarray) -> np.ndarray:
        """Read `x` from memory, returning the `x` and the resulting value.

        Args:
            x np.ndarray: ``(self.dim_A,)`` (or convertible) vector to use as an address.

        Returns:
            The recalled pattern from ``x``.
        """

        if len(x.shape) > 1:
            x = x.squeeze()

        similarities = self.A @ x
        max_sim_idx = np.argmax(np.abs(similarities))
        recalled_pattern = self.P[max_sim_idx]
        return recalled_pattern

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.read(x)

    def savez(self, file: _FileLike, allow_pickle: bool = True) -> None:
        """Save the contents of the associative memory to a file

        The data which is saved to the file is in the following format:
        ```
        {
        stored_traces: ..., # (1,) array containing the number of traces
        dim_A: ..., # (1,) array containing the dimensionality of the traces in `A`
        dim_P, ... # (1,) array containing the dimensionality of the traces in `P`
        A = self.A, # (stored_traces, dim_A) matrix containing the stored addresses
        P = self.P, # (stored_traces, dim_P) matrix containing the stored patterns
        }
        ```

        Args:
            file _FileLike: A ``str``, ``file``, or ``pathlib.Path`` file-like
                object.
            allow_pickle bool: Optional, defaults to true. Passed to ``np.savez``.

        Example:
        ```
        from forth import HeteroAssoc
        from tempfile import TemporaryFile

        assoc = Assoc(dim_A=10, dim_P=10, init_capacity=10)
        ... # add items to assoc memory

        outfile = TemporaryFile()
        assoc.savez(outfile)
        _ = outfile.seek(0) # Only needed to simulate closing and reopening the file
        npzfile = np.load(outfile)
        print(npz.files) # ["stored_traces", "dim_A", "dim_P", "A", "P"]
        A = npzfile["A"]
        ```
        """

        np.savez(
            file,
            stored_traces=np.array(self.stored_traces),
            dim_A=np.array(self.dim_A),
            dim_P=np.array(self.dim_P),
            A=self.A,
            P=self.P,
            allow_pickle=allow_pickle,
        )


class AutoAssoc:
    """Auto-associative memory used for cleanup in encoding and decoding calls.

    Like ``HeteroAssoc``, this differs from the one used in the machine
    as it allows for storage in a file using [`numpy.savez`](https://numpy.org/doc/stable/reference/generated/numpy.savez.html>).

    Attributes:
        dim int: The dimensionality of vectors stored in the weight matrix.
        capacity int: The current first dimension of the weight matrix.
        stored_traces int: The number of non-zero row vectors stored in the weight matrix.
        W np.ndarray: ``(stored_traces, dim)`` matrix storing vectors for auto-association.
    """

    def __init__(self, dim: int, init_capacity: int = 20) -> None:
        self.dim = dim
        self.capacity = init_capacity
        self.stored_traces = 0
        self.W = np.zeros(shape=(init_capacity, dim))

    def write(self, x: np.ndarray) -> None:
        """Write a value to memory.

        Args:
            x np.ndarray: ``(self.dim,)`` (or convertible) vector to store in the weights.
        """
        if len(x.shape) > 1:
            x = x.squeeze()

        # Check if the value is already in the matrix
        sims = self.W @ x
        sims_max_idx = np.argmax(sims)
        threshold = 0.8
        if sims[sims_max_idx] > threshold:
            return

        # Otherwise, store it
        if self.stored_traces >= self.capacity:
            self.W = np.concatenate([self.W, np.zeros((self.capacity, self.dim))])  # type: ignore
            self.capacity *= 2
        self.W[self.stored_traces, :] = x
        self.stored_traces += 1

    def memorize(self, x: np.ndarray) -> None:
        """See ``forth.encoding.AutoAssoc.write``."""
        self.write(x)

    def read(self, x: np.ndarray) -> np.ndarray:
        """Read a value from memory, returning the value and its recalled form.

        Args:
            x np.ndarray: ``(self.dim,)`` (or convertible) vector to recall from the weights.

        Returns:
            The recalled vector which is ``(self.dim,)``.
        """
        similarities = self.W @ x
        max_sim_idx = np.argmax(np.abs(similarities))
        recalled = self.W[max_sim_idx]
        return recalled

    def recall(self, x: np.ndarray) -> np.ndarray:
        """See ``forth.encoding.AutoAssoc.read``."""
        return self.read(x)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.read(x)

    def savez(self, file: _FileLike, allow_pickle: bool = True) -> None:
        """Save the contents of the associative memory to a file

        The data which is saved to the file is in the following format:
        ```
        {
        stored_traces: ..., # (1,) array containing the number of traces
        dim: ... # (1,) array containing the dimensionality of the traces in the weight matrix
        W: self.W # (stored_traces, dim) weight matrix.
        }
        ```

        Args:
            file _FileLike: A ``str``, ``file``, or ``pathlib.Path`` file-like
                object.
            allow_picke bool: Optional, defaults to true. Passed to ``np.savez``.

        Example:
        ```
        from forth import AutoAssoc
        from tempfile import TemporaryFile

        cleanup = Cleanup(dim=100)
        ... # add items to memory

        outfile = TemporaryFile()
        assoc.savez(outfile)
        _ = outfile.seek(0) # Only needed to simulate closing and reopening the file
        npzfile = np.load(outfile)
        print(npzfiles.files) # ["stored_traces", "dim", "W"]
        W = npzfile["W"]
        ```
        """

        np.savez(
            file,
            dim=np.array(self.dim),
            stored_traces=np.array(self.stored_traces),
            W=self.W,
            allow_pickle=allow_pickle,
        )


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


class Codebook(UserDict):
    """Thin dictionary wrapper for codebooks.

    Attributes:
        dim int: The dimensionality of the vectors to store.
    """

    _initialized = False

    def __init__(self, symbols: list[str], dim: int) -> None:
        super(Codebook, self).__init__()
        self.dim = dim
        for symbol in symbols:
            self.data[symbol] = random(1, dim).squeeze()

    def savez(self, file: _FileLike) -> None:
        """Save codebook using [`numpy.savez`](https://numpy.org/doc/stable/reference/generated/numpy.savez.html>).

        The format of the data is each ``(key, value)`` pair from ``self.data.items()``.

        Args:
            file _FileLike: A ``str``, or ``pathlib.Path`` to save the codebook to.

        Example:
        ```
        from forth import Codebook:
        from tempfile import TemporaryFile

        codebook = Codebook(["foo", "bar", "baz"], dim=100)

        outfile = TemporaryFile()
        assoc.savez(outfile)
        _ = outfile.seek(0) # Only needed to simulate closing and reopening the file
        npzfile = np.load(outfile)
        print(npz.files) # ["foo", "bar", "baz"]
        foo = npzfile["foo"]
        ```
        """
        np.savez(file, **self.data)

    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> None:
        self._initialized = True

        # Names for words
        for value in WordType._member_names_:
            self.data[value] = random(1, self.dim).squeeze()

        # Structural items
        self.data["R_LEFT"] = random(1, self.dim).squeeze()
        self.data["R_RIGHT"] = random(1, self.dim).squeeze()
        self.data["R_PHI"] = random(1, self.dim).squeeze()
        self.data["T_NIL"] = random(1, self.dim).squeeze()


@dataclass
class EncodingEnvironment:
    """Dataclass which holds the codebook, associative memory, and cleanup
    memory for encoding.

    See ``Codebook``, ``HeteroAssoc``, and ``AutoAssoc`` for more information.
    """

    codebook: Codebook
    assoc_mem: HeteroAssoc
    cleanup_mem: AutoAssoc

    def savez(self, to_save: _FileLike) -> None:
        """Serialize the encoding environment.

        Given the path ``to_save``, we create new files ``<to_save>/codebook.npz``,
        ``<to_save>/assoc_mem.npz``, and ``<to_save>/cleanup_mem.npz``. Each
        of these contains the serialized forms of the embeddings.

        Args:
            to_save _FileLike: ``pathlib.Path``, `str`, or file-like object
                which is the directory to save the embeddings to.
        """
        to_save = Path(to_save)
        codebook_file = to_save / "codebook"
        assoc_mem_file = to_save / "assoc_mem"
        cleanup_mem_file = to_save / "cleanup_mem"

        self.codebook.savez(codebook_file)
        self.assoc_mem.savez(assoc_mem_file)
        self.cleanup_mem.savez(cleanup_mem_file)


def cconv(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return fft.ifft(fft.fft(x) * fft.fft(y)).real


def invert(x: np.ndarray) -> np.ndarray:
    return x[np.r_[0, x.size - 1 : 0 : -1]]


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    norm = np.linalg.vector_norm(x) * np.linalg.vector_norm(y)
    norm = max(norm, 1e-8)
    return float((x.dot(y)) / norm)


def make_cons(
    lhs: np.ndarray,
    rhs: np.ndarray,
    enc_env: EncodingEnvironment,
    numeric: bool = False,
    theta: float = 0.8,
) -> np.ndarray:
    """Create a new linked list encoding out of ``lhs`` and ``rhs``.

    Args:
        lhs np.ndarray: High-dimensional vector encoding.
        rhs np.ndarray: High-dimensional vector encoding.
        enc_env EncodingEnvironment: The encoding environment.

    Returns:
        A linked-list representation of ``lhs`` and ``rhs``.
    """
    enc_env.cleanup_mem.write(lhs)
    enc_env.cleanup_mem.write(rhs)

    lhs_name = f"Pointer_to_{get_name(lhs, enc_env.codebook)}"
    if cosine_similarity(lhs, enc_env.codebook["T_NIL"]) > theta:
        lhs_ptr = lhs
    elif lhs_name in enc_env.codebook:
        lhs_ptr = enc_env.codebook[lhs_name]
    else:
        lhs_ptr = random(1, enc_env.codebook.dim).squeeze()
        enc_env.cleanup_mem.write(lhs_ptr)
        enc_env.codebook[lhs_name] = lhs_ptr
        enc_env.assoc_mem.write(lhs_ptr, lhs)

    rhs_name = f"Pointer_to_{get_name(rhs, enc_env.codebook)}"
    if cosine_similarity(rhs, enc_env.codebook["T_NIL"]) > theta:
        rhs_ptr = rhs
    elif rhs_name in enc_env.codebook:
        rhs_ptr = enc_env.codebook[rhs_name]
    else:
        rhs_ptr = random(1, enc_env.codebook.dim).squeeze()
        enc_env.cleanup_mem.write(rhs_ptr)
        enc_env.codebook[rhs_name] = rhs_ptr
        enc_env.assoc_mem.write(rhs_ptr, rhs)

    rfpair = (
        cconv(lhs_ptr, enc_env.codebook["R_LEFT"])
        + cconv(rhs_ptr, enc_env.codebook["R_RIGHT"])
        + enc_env.codebook["R_PHI"]
    )
    norm = np.linalg.vector_norm(rfpair)
    norm = max(norm, 1e-8)
    rfpair /= norm
    if numeric:
        old_name = lhs_name.replace("Pointer_to_", "").replace("NUMBER_", "")
        if old_name == "T_NIL":
            old_name = "0"

        name = f"NUMBER_{int(old_name) + 1}"
    else:
        name = f"LS_{lhs_name.replace('Pointer_to_', '')}_{rhs_name.replace('Pointer_to_', '')}"
    enc_env.codebook[name] = rfpair
    return rfpair


def cons(xs: list[np.ndarray], enc_env: EncodingEnvironment) -> np.ndarray:
    base = enc_env.codebook["T_NIL"]
    for x in reversed(xs):
        base = make_cons(x, base, enc_env)
    return base


def get_name(x: np.ndarray, codebook: Codebook, theta: float = 0.4) -> str:
    keys, values = zip(*codebook.items())
    weights = np.array(values)
    sims = weights @ x
    sims_max_idx = np.argmax(sims)
    if sims[sims_max_idx] > theta:
        return keys[sims_max_idx]
    else:
        raise KeyError("Unable to retrieve name")


def encode_number(cont: str, enc_env: EncodingEnvironment) -> np.ndarray:
    """Encode a number using the list encoding.

    Args:
        cont str: The string contents.
        enc_env EncodingEnvironment: The encoding environment.

    Returns:
        The list encoding of the number.
    """
    n = int(cont)
    if n < 0:
        raise ValueError("Can only encode values >=0")
    print(f"ENCODE_NUMBER: {n}", file=sys.stderr)

    num = enc_env.codebook["T_NIL"]
    for _ in range(n):
        num = make_cons(num, enc_env.codebook["T_NIL"], enc_env=enc_env, numeric=True)
    enc_env.codebook[get_name(num, enc_env.codebook)] = num
    return num


def encode_binary_number(
    cont: str, enc_env: EncodingEnvironment, vocab: spa.Vocabulary, width: int
) -> np.ndarray:
    """Encode an integer using ``Bitstrings``.

    Args:
        cont str:
            The contents of the lexed item.
        enc_env EncodingEnvironment:
            The encoding environment.
        vocab spa.Vocabulary:
            The vocabulary to use in encoding
        width int:
            The ``Bitstring`` width.

    Returns:
        Modifies `enc_env` and `vocab` in-place. Returns the encoded
        form of `cont` using `Bitstring`.
    """

    parsed = int(cont)
    bs = num.encode(parsed, vocab, width=width)

    for i in range(width):
        if f"S_{i}" not in enc_env.codebook:
            enc_env.codebook[f"S_{i}"] = vocab[f"S_{i}"].v

    if "BIT_0" not in enc_env.codebook:
        enc_env.codebook["BIT_0"] = vocab["BIT_0"]
    if "BIT_1" not in enc_env.codebook:
        enc_env.codebook["BIT_1"] = vocab["BIT_1"]

    if bs.name not in enc_env.codebook:
        enc_env.codebook[bs.name] = bs.v

    return bs.v


def encode(
    words: list[Word],
    enc_env: EncodingEnvironment,
    integer_encoding_scheme: typing.Literal["binary", "list"] = "list",
    vocab: typing.Optional[spa.Vocabulary] = None,
    width: int = 8,
) -> np.ndarray:
    """Encode a list of ``Word``'s into a high-dimensional vector.

    Args:
        words list[Word]: The list of words to encode.
        enc_env EncodingEnvironment: The encoding environment dataclass.
        vocab typing.Optional[spa.Vocabulary]:
            Optional vocabulary used in encoding binary strings.
        width int:
            Optional width used in encoding binary strings. Defaults to ``8``.

    Returns:
        The list of words encoded as a linked-list.
    """
    if not enc_env.codebook.is_initialized():
        enc_env.codebook.initialize()

    encoded_words = []
    for word in words:
        wordtype = word.tag
        if wordtype == WordType.NUMBER:
            if integer_encoding_scheme == "list":
                encoded_words.append(encode_number(word.cont, enc_env))
            elif integer_encoding_scheme == "binary" and vocab is None:
                raise ValueError("Vocabulary cannot be NONE if you do binary encoding")
            else:
                encoded_words.append(
                    encode_binary_number(word.cont, enc_env, vocab, width)
                )
        elif wordtype == WordType.IDENT:
            cont = word.cont
            if cont not in enc_env.codebook:
                enc_env.codebook[cont] = random(1, enc_env.codebook.dim).squeeze()
            encoded_value = enc_env.codebook[cont]
            encoded_words.append(encoded_value)
        elif wordtype == WordType.EOF:
            break
        else:
            encoded_value = enc_env.codebook[wordtype2str(wordtype)]
            encoded_words.append(encoded_value)

    return cons(encoded_words, enc_env)


def savez(
    file: _FileLike, encoded_representation: np.ndarray, enc_env: EncodingEnvironment
) -> None:
    """Serialize the results of an encoding.

    Args:
        file _FileLike: File-like object.
        encoded_representation np.ndarray: ``(d,)`` high-dimensional vector.
        enc_env EncodingEnvironment: The encoding environment.
    """

    path = Path(file)
    if not path.exists():
        path.mkdir()
    encoded_representation_path = path / "embeddings"
    np.savez(encoded_representation_path, embeddings=encoded_representation)
    enc_env.savez(file)
