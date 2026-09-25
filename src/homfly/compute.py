"""High-level entry point.

    >>> from homfly import homfly
    >>> homfly("3_1")                    # fundamental, from the knot table
    >>> homfly("4_1", R=(2,))            # colored via cabling (or Racah tables)
    >>> homfly(TorusKnot(3, 4), R=(2, 1))  # Rosso--Jones

Every computation runs a method as a black box over GF(p) and reconstructs the
Laurent polynomial in (A, q) by interpolation + CRT (reconstruction.pipeline).
"""
from __future__ import annotations

from .knots.braid import Braid
from .knots.families import TorusKnot, TwoBridge, DoubleBraid
from .knots.table import load_table
from .methods import (Cabling, HeckeFundamental, NotApplicable, RossoJones)
from .reconstruction.pipeline import reconstruct_laurent, ReconstructionReport
from .reps.partitions import P


def resolve_knot(k):
    """Name from the table -> Braid; other descriptions pass through."""
    if isinstance(k, str):
        rec = load_table()[k]
        if rec.braid is None:
            raise ValueError("no braid for %s (unknot?)" % k)
        return rec.braid
    if isinstance(k, (list, tuple)) and all(isinstance(a, int) for a in k):
        return Braid.from_word(k)
    return k


def default_methods(racah_store=None):
    ms = [RossoJones(), HeckeFundamental()]
    if racah_store is not None:
        from .methods.rt_braid import RTBraid
        from .methods.arborescent import Arborescent
        ms += [RTBraid(racah_store), Arborescent(racah_store)]
    ms.append(Cabling())
    return ms


def choose_method(knot, R, methods=None):
    for m in methods or default_methods():
        if m.supports(knot, R):
            return m
    raise NotApplicable("no available method for %s in %s" % (knot, R))


def homfly(knot, R=(1,), method=None, racah_store=None, seed=0, report=None, **kw):
    """Reduced colored HOMFLY H_R(K; A, q) as a Laurent polynomial."""
    R = P(R)
    K = resolve_knot(knot)
    if isinstance(K, TorusKnot) and K.m == 1 or (isinstance(K, Braid) and K.strands == 1):
        from .algebra.laurent import Laurent
        return Laurent.const(1)
    if method is None:
        m = choose_method(K, R, default_methods(racah_store))
    elif isinstance(method, str):
        m = {x.name: x for x in default_methods(racah_store)}[method]
    else:
        m = method
    if isinstance(K, TorusKnot) and not m.supports(K, R):
        K = K.braid()
    rep = report if report is not None else ReconstructionReport()
    rep.method = m.name
    return reconstruct_laurent(m.black_box(K, R), steps=m.steps, seed=seed, report=rep, **kw)


__all__ = ["homfly", "TorusKnot", "TwoBridge", "DoubleBraid", "Braid"]
