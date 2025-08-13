import pytest

import nengo_spa as spa
import numerical as num

def test_bin_encode() -> None:
    dim = 256
    vocab_keys = [
        "S_0",
        "S_1",
        "S_2",
        "S_3",
        "S_4",
        "S_5",
        "S_6",
        "S_7",
        "BIT_0",
        "BIT_1",
    ]
    vocab = spa.Vocabulary(dim)
    vocab.populate(";".join(vocab_keys))
    theta = 0.8

    b_zero = num.encode(0, vocab)
    man_zero = (
        (vocab.S_0 * vocab.BIT_0)
        + (vocab.S_1 * vocab.BIT_0)
        + (vocab.S_2 * vocab.BIT_0)
        + (vocab.S_3 * vocab.BIT_0)
        + (vocab.S_4 * vocab.BIT_0)
        + (vocab.S_5 * vocab.BIT_0)
        + (vocab.S_6 * vocab.BIT_0)
        + (vocab.S_7 * vocab.BIT_0)
    )

    assert b_zero.compare(man_zero) > theta, "Encoded representation and manual representation should be the same."
    


def test_bin_add() -> None:
    pass


def test_bin_sub() -> None:
    pass


def test_bin_mul() -> None:
    pass


def test_bin_div() -> None:
    pass


def test_bin_lshift() -> None:
    pass


def test_bin_rshift() -> None:
    pass


def test_bin_lt() -> None:
    pass


def test_bin_gt() -> None:
    pass


def test_bin_and() -> None:
    pass


def test_bin_or() -> None:
    pass


def test_bin_xor() -> None:
    pass


def test_bin_to_bits() -> None:
    pass


# def test_from_list_bits() -> None:
    # dim = 256
    # vocab_keys = ["S_0", "S_1", "S_2", "S_3", "S_4", "S_5", "S_6", "S_7"]
    # vocab = spa.Vocabulary(dim)
    # vocab.populate(";".join(vocab_keys))

    # b_one = num.encode(1, vocab)
