"""Modular reconstruction pipeline.

    black box  H(A0, q0) in GF(p)            (any method in homfly.methods)
        |  dense / sparse interpolation       (interpolation.py)
        v
    H mod p  as {(i, j): c mod p}
        |  CRT over several primes            (crt.py)
        v
    H mod M  -> symmetric lift (integer coefficients)
             or rational reconstruction (rational coefficients)
        |  stabilisation + verification at fresh random points / fresh prime
        v
    Laurent in (A, q)

The black box is a callable ``evaluate(F, A0, q0) -> F-element``.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..algebra.fields import GF, BadPoint, primes_below
from ..algebra.laurent import Laurent, AQ
from .crt import crt_pair, symmetric_lift, rational_reconstruction
from .interpolation import dense_bivariate, detect_exponent_box

DEFAULT_PRIME_BOUND = 2 ** 62


@dataclass
class ReconstructionReport:
    primes: list = field(default_factory=list)
    A_range: tuple = None
    q_range: tuple = None
    evaluations: int = 0
    verified: bool = False


def reconstruct_laurent(evaluate, steps=(1, 1), bounds=None, rational=False,
                        prime_bound=DEFAULT_PRIME_BOUND, max_primes=20,
                        verify_points=3, seed=None, report=None, qsym=False):
    """Reconstruct a Laurent polynomial in (A, q) from a modular black box.

    Parameters
    ----------
    evaluate : callable(F, A0, q0) -> element of F
    steps    : exponent steps (2, 2) for knots (H is a polynomial in A^2, q^2)
    bounds   : ((Amin, Amax), (qmin, qmax)) or None for automatic detection
    rational : allow rational coefficients (Wang reconstruction) instead of
               integers (symmetric lift)
    """
    rng = random.Random(seed)
    rep = report if report is not None else ReconstructionReport()
    counter = [0]
    primes = primes_below(prime_bound, max_primes)

    def black_box(F):
        def f2(a, q):
            counter[0] += 1
            return evaluate(F, a, q)
        return f2

    residues, modulus = {}, 1
    previous = None
    for p in primes:
        F = GF(p)
        f2 = black_box(F)
        if bounds is None:
            bounds = detect_exponent_box(f2, F, steps=steps, rng=rng)
        rep.A_range, rep.q_range = bounds
        img = dense_bivariate(f2, F, bounds[0], bounds[1], steps=steps, rng=rng, qsym=qsym)
        rep.primes.append(p)
        keys = set(residues) | set(img)
        new = {}
        for k in keys:
            r, _ = crt_pair(residues.get(k, 0), modulus, int(img[k]) if k in img else 0, p)
            new[k] = r
        residues, modulus = new, modulus * p
        lifted = _lift(residues, modulus, rational)
        # early exit: coefficients usually fit one prime; a lift that is wrong
        # fails at random points of an independent prime with overwhelming
        # probability, so check it there instead of a full second interpolation
        if (lifted is not None and previous is None and len(primes) > 2
                and _verify(Laurent(lifted, AQ), evaluate, primes[-1], verify_points + 2, rng)):
            previous = lifted
        if lifted is not None and lifted == previous:
            poly = Laurent(lifted, AQ)
            fresh = primes[min(len(rep.primes), len(primes) - 1)]
            rep.verified = _verify(poly, evaluate, fresh, verify_points, rng)
            rep.evaluations = counter[0]
            if not rep.verified:
                raise RuntimeError("reconstruction failed verification (bounds too small?)")
            return poly
        previous = lifted
    raise RuntimeError("no stabilisation after %d primes" % len(primes))


def _lift(residues, modulus, rational):
    out = {}
    for k, r in residues.items():
        if rational:
            c = rational_reconstruction(r, modulus)
            if c is None:
                return None
        else:
            c = symmetric_lift(r, modulus)
        if c != 0:
            out[k] = c
    return out


def _verify(poly, evaluate, p, npts, rng):
    F = GF(p)
    ok = 0
    tries = 0
    while ok < npts and tries < 10 * npts:
        tries += 1
        a, q = F.random_element(rng), F.random_element(rng)
        try:
            v = evaluate(F, a, q)
        except BadPoint:
            continue
        if poly.evaluate([a, q], one=F.one) != v:
            return False
        ok += 1
    return ok == npts
