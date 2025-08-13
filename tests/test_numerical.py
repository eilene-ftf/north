import nengo_spa as spa
import pytest

import numpy as np
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
    theta = 0.99
    rng = np.random.default_rng(0)

    b_zero = num.encode(0, vocab, rng=rng)
    man_zero = (
        (vocab["S_0"] * vocab["BIT_0"])
        + (vocab["S_1"] * vocab["BIT_0"])
        + (vocab["S_2"] * vocab["BIT_0"])
        + (vocab["S_3"] * vocab["BIT_0"])
        + (vocab["S_4"] * vocab["BIT_0"])
        + (vocab["S_5"] * vocab["BIT_0"])
        + (vocab["S_6"] * vocab["BIT_0"])
        + (vocab["S_7"] * vocab["BIT_0"])
    )

    assert b_zero.compare(man_zero) > theta, (
        "Encoded representation and manual representation should be the same."
    )

    b_one = num.encode(1, vocab, rng=rng)
    man_one = (
        (vocab["S_0"] * vocab["BIT_0"])
        + (vocab["S_1"] * vocab["BIT_0"])
        + (vocab["S_2"] * vocab["BIT_0"])
        + (vocab["S_3"] * vocab["BIT_0"])
        + (vocab["S_4"] * vocab["BIT_0"])
        + (vocab["S_5"] * vocab["BIT_0"])
        + (vocab["S_6"] * vocab["BIT_0"])
        + (vocab["S_7"] * vocab["BIT_1"])
    )

    assert b_one.compare(man_one) > theta, (
        "Encoded representation and manual representation should be the same."
    )

    b_two = num.encode(2, vocab, rng=rng)
    man_two = (
        (vocab["S_0"] * vocab["BIT_0"])
        + (vocab["S_1"] * vocab["BIT_0"])
        + (vocab["S_2"] * vocab["BIT_0"])
        + (vocab["S_3"] * vocab["BIT_0"])
        + (vocab["S_4"] * vocab["BIT_0"])
        + (vocab["S_5"] * vocab["BIT_0"])
        + (vocab["S_6"] * vocab["BIT_1"])
        + (vocab["S_7"] * vocab["BIT_0"])
    )

    assert b_two.compare(man_two) > theta, (
        "Encoded representation and manual representation should be the same."
    )

    b_three = num.encode(3, vocab, rng=rng)
    man_three = (
        (vocab["S_0"] * vocab["BIT_0"])
        + (vocab["S_1"] * vocab["BIT_0"])
        + (vocab["S_2"] * vocab["BIT_0"])
        + (vocab["S_3"] * vocab["BIT_0"])
        + (vocab["S_4"] * vocab["BIT_0"])
        + (vocab["S_5"] * vocab["BIT_0"])
        + (vocab["S_6"] * vocab["BIT_1"])
        + (vocab["S_7"] * vocab["BIT_1"])
    )

    assert b_three.compare(man_three) > theta, (
        "Encoded representation and manual representation should be the same."
    )


# def test_bin_add() -> None:
#     pass


# def test_bin_sub() -> None:
#     pass


# def test_bin_mul() -> None:
#     pass


# def test_bin_div() -> None:
#     pass


# def test_bin_lshift() -> None:
#     pass


# def test_bin_rshift() -> None:
#     pass


# def test_bin_lt() -> None:
#     pass


# def test_bin_gt() -> None:
#     pass


def test_bin_and() -> None:
    dim = 256
    vocab_keys = ["S_0", "S_1", "S_2", "S_3", "S_4", "S_5", "S_6", "S_7"]
    vocab = spa.Vocabulary(dim)
    vocab.populate(";".join(vocab_keys))
    theta = 0.8
    rng = np.random.default_rng(0)

    b_five = num.encode(5, vocab, width=8, rng=rng) 
    b_four = num.encode(4, vocab, width=8, rng=rng)

    should_be_five = b_five.band(b_five)
    assert b_five.compare(should_be_five) > theta
    assert b_four.compare(b_five.band(b_four)) > theta
    
    b_8 = num.encode(8, vocab, rng=rng)
    b_3 = num.encode(3, vocab, rng=rng)
    b_0 = num.encode(0, vocab, rng=rng)

    assert b_0.compare(b_8.band(b_3)) > theta


# def test_bin_or() -> None:
#     pass


# def test_bin_xor() -> None:
#     pass


# def test_bin_to_bits() -> None:
#     pass


def test_from_list_bits() -> None:
    dim = 256
    vocab_keys = ["S_0", "S_1", "S_2", "S_3", "S_4", "S_5", "S_6", "S_7"]
    vocab = spa.Vocabulary(dim)
    vocab.populate(";".join(vocab_keys))
    theta = 0.8
    rng = np.random.default_rng(0)

    b_one = num.encode(1, vocab, width=8, rng=rng)

    man_bits = [
        vocab["BIT_0"],
        vocab["BIT_0"],
        vocab["BIT_0"],
        vocab["BIT_0"],
        vocab["BIT_0"],
        vocab["BIT_0"],
        vocab["BIT_0"],
        vocab["BIT_1"],
    ]
    enc_bits = b_one.get_bits()

    for enc_bit, man_bit in zip(enc_bits, man_bits):
        assert enc_bit.compare(man_bit) > theta

    b_two = num.encode(2, vocab, width=8, rng=rng)

    man_bits = [
        vocab["BIT_0"],
        vocab["BIT_0"],
        vocab["BIT_0"],
        vocab["BIT_0"],
        vocab["BIT_0"],
        vocab["BIT_0"],
        vocab["BIT_1"],
        vocab["BIT_0"],
    ]
    enc_bits = b_two.get_bits()

    for enc_bit, man_bit in zip(enc_bits, man_bits):
        assert enc_bit.compare(man_bit) > theta

    b_three = num.encode(3, vocab, width=8, rng=rng)

    man_bits = [
        vocab["BIT_0"],
        vocab["BIT_0"],
        vocab["BIT_0"],
        vocab["BIT_0"],
        vocab["BIT_0"],
        vocab["BIT_0"],
        vocab["BIT_1"],
        vocab["BIT_1"],
    ]
    enc_bits = b_three.get_bits()

    for enc_bit, man_bit in zip(enc_bits, man_bits):
        assert enc_bit.compare(man_bit) > theta
