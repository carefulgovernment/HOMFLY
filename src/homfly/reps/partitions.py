"""Young diagrams (partitions) and combinatorics needed by all methods.

A partition is a tuple of positive integers in weakly decreasing order,
e.g. ``(3, 1)`` for the diagram [3,1].  Boxes are ``(i, j)`` with row ``i`` and
column ``j`` starting from 0; the content is ``j - i``.
"""
from __future__ import annotations

from functools import lru_cache


def P(*parts):
    """Normalise ``P(2,1)``, ``P([2,1])``, ``P("21")`` to a partition tuple."""
    if len(parts) == 1 and not isinstance(parts[0], int):
        parts = parts[0]
        if isinstance(parts, str):
            parts = [int(c) for c in parts.replace(",", "").replace("[", "").replace("]", "")]
    return tuple(sorted((int(x) for x in parts if x), reverse=True))


def size(la):
    return sum(la)


@lru_cache(maxsize=None)
def partitions(n, max_part=None):
    if max_part is None:
        max_part = n
    if n == 0:
        return ((),)
    out = []
    for k in range(min(n, max_part), 0, -1):
        for rest in partitions(n - k, k):
            out.append((k,) + rest)
    return tuple(out)


def conjugate(la):
    if not la:
        return ()
    return tuple(sum(1 for x in la if x > j) for j in range(la[0]))


def boxes(la):
    return [(i, j) for i, r in enumerate(la) for j in range(r)]


def content(box):
    i, j = box
    return j - i


def kappa(la):
    """kappa_R = sum of contents = nu(R^T) - nu(R)."""
    return sum(j - i for i, j in boxes(la))


def hook(la, box):
    i, j = box
    lt = conjugate(la)
    return (la[i] - j - 1) + (lt[j] - i - 1) + 1


def addable(la):
    out = []
    for i in range(len(la) + 1):
        r = la[i] if i < len(la) else 0
        if i == 0 or la[i - 1] > r:
            out.append((i, r))
    return out


def removable(la):
    return [(i, la[i] - 1) for i in range(len(la)) if i == len(la) - 1 or la[i] > la[i + 1]]


def add_box(la, i):
    l = list(la) + [0]
    l[i] += 1
    return P(l)


def remove_box(la, i):
    l = list(la)
    l[i] -= 1
    return P(l)


@lru_cache(maxsize=None)
def standard_tableaux(la):
    """All standard Young tableaux of shape ``la`` as *paths*: tuple of the rows
    into which boxes 1, 2, ..., n are placed.  (Row sequence = Yamanouchi word.)"""
    la = P(la)
    if not la:
        return ((),)
    out = []
    for (i, j) in removable(la):
        for t in standard_tableaux(remove_box(la, i)):
            out.append(t + (i,))
    return tuple(out)


def num_standard_tableaux(la):
    from math import factorial
    n = size(la)
    h = 1
    for b in boxes(la):
        h *= hook(la, b)
    return factorial(n) // h


def is_rectangular(la):
    return len(set(la)) <= 1


def is_hook(la):
    return len(la) <= 1 or all(x == 1 for x in la[1:])


def tableau_contents(path):
    """Contents c(1..n) of the boxes of a tableau given as a row path."""
    rows = []
    cs = []
    for i in path:
        while len(rows) <= i:
            rows.append(0)
        cs.append(rows[i] - i)
        rows[i] += 1
    return cs


def fmt(la):
    return "[" + ",".join(map(str, la)) + "]"
