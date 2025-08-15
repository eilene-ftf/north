"""Algebraic implementation of bit-wise operations. For more information
about the specific operations, see ``..bitsring.Bitstring``.
"""

import functools

import nengo # type: ignore
import nengo_spa as spa # type: ignore
import numpy as np


class BXor(spa.Network):
    """Algebraic implementation of bit-wise XOR."""

    pass


class BAnd(spa.Network):
    """Algebraic implementation of bit-wise AND."""


class BOr(spa.Network):
    """Algebraic implementaiton of bit-wise OR."""


class BNot(spa.Network):
    """Algebriac implementation of bit-wise flipping."""
