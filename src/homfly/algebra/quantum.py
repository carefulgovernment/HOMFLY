"""Quantum numbers and related elementary functions, evaluated in a field."""
from __future__ import annotations

from .fields import BadPoint


def qnum(n, q):
    """[n]_q = (q^n - q^-n)/(q - q^-1)."""
    den = q - q ** -1
    if den == 0:
        raise BadPoint("q = +-1")
    return (q ** n - q ** -n) / den


def qnum_A(A, q, shift=0):
    """{A q^shift} / {q}  with {x} = x - 1/x.  (A = q^N gives [N + shift].)"""
    den = q - q ** -1
    if den == 0:
        raise BadPoint("q = +-1")
    x = A * q ** shift
    return (x - x ** -1) / den


def curly(x):
    """{x} = x - x^{-1}."""
    return x - x ** -1
