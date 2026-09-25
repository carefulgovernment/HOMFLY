"""Quantum dimensions and framing factors, evaluated in a field.

Conventions (see docs/CONVENTIONS.md):  dim_q R = prod_{boxes} {A q^c}/{q^h},
{x} = x - 1/x, c = content, h = hook length;  for A = q^N this is the
U_q(sl_N) quantum dimension.  The framing (twist) eigenvalue is
theta_R = A^{|R|} q^{2 kappa_R}.
"""
from __future__ import annotations

from ..algebra.fields import BadPoint
from .partitions import boxes, conjugate, kappa


def qdim(R, A, q):
    R = tuple(R)
    lt = conjugate(R)
    num = A * 0 + 1
    den = A * 0 + 1
    for (i, j) in boxes(R):
        c = j - i
        h = (R[i] - j - 1) + (lt[j] - i - 1) + 1
        x = A * q ** c
        num = num * (x - x ** -1)
        den = den * (q ** h - q ** -h)
    if den == 0:
        raise BadPoint("q is a small root of unity")
    return num / den


def theta(R, A, q):
    return A ** sum(R) * q ** (2 * kappa(R))
