"""Decode the high-dimensional vector embedding of FORTH source-level code."""

import numpy as np

from .encoding import (
    EncodingEnvironment,
    cosine_similarity,
    invert,
    cconv,
)

__all__ = ["decode"]


def car(x: np.ndarray, enc_env: EncodingEnvironment) -> np.ndarray:
    """Retrieve the ``enc_env.codebook["___lhs"]`` element from the pointer.

    Args:
        x np.ndarray: High-dimensional vector associated with tuple.
        enc_env EncodingEnvironment: The encoding environment.

    Returns:
        The element bound with ``enc_env.codebook["___lhs"]``.
    """
    rfpair = enc_env.assoc_mem.read(x)
    lhs = cconv(rfpair, invert(enc_env.codebook["___lhs"]))
    return enc_env.cleanup_mem.read(lhs)


def cdr(x: np.ndarray, enc_env: EncodingEnvironment) -> np.ndarray:
    """Retrieve the ``enc_env.codebook["___rhs"]`` element from the pointer.

    Args:
        x np.ndarray: High-dimensional vector associated with tuple.
        enc_env EncodingEnvironment: The encoding environment.

    Returns:
        The element bound with ``enc_env.codebook["___rhs"]``.
    """
    rfpair = enc_env.assoc_mem.read(x)
    rhs = cconv(rfpair, invert(enc_env.codebook["___rhs"]))
    return enc_env.cleanup_mem.read(rhs)


def decode_word(
    x: np.ndarray, enc_env: EncodingEnvironment, theta: float = 0.2
) -> str | None:
    """Decode a high-dimensional vector embedding.

    Args:
        x np.ndarray: High-dimensional vector embedding of a word.
        enc_env EncodingEnvironment: The encoding environment.
        theta float: Optional, similarity threshold. Default: 0.2

    Returns:
        The string corresponding to the item, or `None` if the similarity
        is less than ``theta``.
    """
    keys, codes = zip(*enc_env.codebook.items())
    values = np.array(codes)
    sims = values @ x
    sims_max_idx = np.argmax(sims)
    if sims[sims_max_idx] > theta:
        decoded_value = keys[sims_max_idx]
        if decoded_value == "EOF":
            return None
        else:
            return decoded_value
    else:
        return None


def decode(x: np.ndarray, enc_env: EncodingEnvironment) -> list[str]:
    """Decode a high-dimensional vector embedding of the FORTH language.

    Args:
        x np.ndarray: High-dimensional vector embedding of the word list.
        enc_env EncodingEnvironment: The encoding environment

    Returns:
        Decoded embeddings as a list of strings.
    """
    car_ = car(x, enc_env)
    cdr_ = cdr(x, enc_env)
    word_rec = []
    head = decode_word(car_, enc_env)
    while head is not None:
        word_rec.append(head)
        car_ = car(cdr_, enc_env)
        cdr_ = cdr(cdr_, enc_env)
        head = decode_word(car_, enc_env)
    return word_rec
