import matplotlib.pyplot as plt
import nengo  # type: ignore
import nengo_spa as spa  # type: ignore
import numpy as np
import pytest

import numerical as num
import numerical.networks as numnet


def test_bnot() -> None:
    dim = 256
    vocab_keys = ["S_0", "S_1", "S_2", "S_3", "S_4", "S_5", "S_6", "S_7"]
    vocab = spa.Vocabulary(dim)
    vocab.populate(";".join(vocab_keys))
    theta = 0.8
    rng = np.random.default_rng(0)
    width = 8

    b_zero = num.encode(0, vocab, width=width)

    with numnet.BNot(vocab, 8) as model:
        pass

    with nengo.Simulator(model) as sim:
        sim.run(1)
