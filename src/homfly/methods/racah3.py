"""3-strand colored HOMFLY from the family-P inclusive Racah blocks
(racah.portable): every R with |R| <= 5 (plus whatever else is imported).

    H_R^{nat}(beta) = sum_{Q ∈ R^{⊗3}} dim_q(Q) Tr_Q(R_{i1}^{±1} R_{i2}^{±1} ...) / dim_q(R)

The blocks already carry the topological framing factor, so no writhe
correction is applied.  2-strand braids are Markov-stabilised (append sigma_2).
Family P is in our NATURAL convention (pinned by tests/test_racah3.py).
"""
from __future__ import annotations

from functools import lru_cache

from ..algebra.fields import BadPoint
from ..algebra.linalg import matmul
from ..conventions import natural_point
from ..knots.braid import Braid
from ..racah import portable
from ..reps.partitions import P
from ..reps.qdim import qdim
from .base import Method

_KEY = {1: "R1", 2: "R2", -1: "R1_inv", -2: "R2_inv"}


def three_strand_word(braid):
    if braid.strands == 3:
        return list(braid.word)
    if braid.strands == 2:
        return list(braid.word) + [2]
    if braid.strands == 1:
        return [1, 2]
    raise ValueError("not a <= 3-strand braid")


def natural_value(R, word, F, A, q, root=portable.DEFAULT_DIR):
    data = portable.load_inclusive(tuple(R), root)
    if getattr(F, "p", None) is not None:
        return _natural_value_modp(data, R, word, F, A, q)
    tot = F.zero
    for ch in data.channels:
        M = ch.evaluate(F, A, q)
        # multiply left to right: rho(beta) = rho(w1) rho(w2) ...
        prod = M[_KEY[word[0]]]
        for a in word[1:]:
            prod = matmul(prod, M[_KEY[a]])
        tr = prod[0][0]
        for i in range(1, len(prod)):
            tr = tr + prod[i][i]
        tot = tot + qdim(ch.Q, A, q) * tr
    return tot / qdim(R, A, q)


def _natural_value_np(data, R, word, F, A, q):
    import numpy as np
    p = F.p
    keys = tuple(sorted({_KEY[a] for a in word}))
    M = data.evaluate_np(p, int(A), int(q), keys)
    tot = 0
    for idx, ch in enumerate(data.channels):
        prod = M[_KEY[word[0]]][idx]
        for a in word[1:]:
            X = M[_KEY[a]][idx]
            prod = np.fmod(prod.astype(np.float64) @ X.astype(np.float64), p).astype(np.int64)
        tr = int(np.trace(prod)) % p
        if tr:
            tot += int(qdim(ch.Q, A, q)) * tr
    return F(tot) / qdim(R, A, q)


NP_BOUND = 2 ** 21


def _natural_value_modp(data, R, word, F, A, q):
    p = F.p
    if p < NP_BOUND:
        return _natural_value_np(data, R, word, F, A, q)
    Ai, qi = int(A), int(q)
    keys = sorted({_KEY[a] for a in word})
    tot = 0
    for ch in data.channels:
        M = ch.evaluate_modp(p, Ai, qi, keys)
        prod = M[_KEY[word[0]]]
        for a in word[1:]:
            prod = portable.matmul_mod(prod, M[_KEY[a]], p)
        tr = sum(prod[i][i] for i in range(len(prod)))
        tot += int(qdim(ch.Q, A, q)) * tr
    return F(tot) / qdim(R, A, q)


class Racah3Strand(Method):
    name = "racah-3strand"
    prime_bound = NP_BOUND        # numpy/BLAS fast path

    def __init__(self, root=portable.DEFAULT_DIR):
        self.root = root

    def supports(self, knot, R):
        return (isinstance(knot, Braid) and knot.strands <= 3 and knot.is_knot()
                and P(R) in portable.available(self.root))

    def evaluate(self, knot, R, F, A, q):
        A, q = natural_point(A, q)
        return natural_value(P(R), three_strand_word(knot), F, A, q, self.root)


class Racah3StrandU(Method):
    """3-strand knots from A-independent U_Q blocks (families G/F: [4,2],
    [2,2,1,1], [3,2,1]).  Natural convention with framing theta_R^{-w}
    (pinned against Rosso--Jones T[3,4], T[3,5] in tests/test_racah_data.py).
    Traces are cached per q, so the A-direction of the interpolation grid is
    almost free."""

    name = "racah-3strand-U"
    prime_bound = 2 ** 21

    def __init__(self, root=None):
        from ..racah import uform
        self.uform = uform
        self.root = root or uform.LARGE_DIR
        self._row = None

    def supports(self, knot, R):
        return (isinstance(knot, Braid) and knot.strands <= 3 and knot.is_knot()
                and P(R) in self.uform.available(self.root))

    def evaluate(self, knot, R, F, A, q):
        from ..reps.qdim import theta
        if getattr(F, "p", None) is None or F.p >= self.uform.PRIME_BOUND:
            raise ValueError("racah-3strand-U needs GF(p) with p < 2^21")
        A, q = natural_point(A, q)
        R = P(R)
        U = self.uform.load(R, self.root)
        word = three_strand_word(knot)
        p, a, qq = F.p, int(A), int(q)
        key = (R, tuple(word), p, qq)
        if self._row is None or self._row[0] != key:
            # per q: the traces and, per Q with t_Q != 0, its box contents and
            # inverse hook product; dim_q(Q) = prod_boxes [A q^c] / prod [q^h]
            t = U.traces(word, p, qq)
            terms = []
            for tq, Q in zip(t, U.Q):
                if tq:
                    hd = 1
                    for h in _hooks(Q):
                        hd = hd * (pow(qq, h, p) - pow(qq, -h, p)) % p
                    if hd == 0:
                        raise BadPoint("q is a small root of unity")
                    terms.append((tq * pow(hd, -1, p) % p, _contents(Q)))
            cs = [c for _, cc in terms for c in cc] or [0]
            qc = {c: pow(qq, c, p) for c in range(min(cs), max(cs) + 1)}
            self._row = (key, terms, qc)
        _, terms, qc = self._row
        g = {}
        for c, x in qc.items():
            y = a * x % p
            if y == 0:
                raise BadPoint("A q^c = 0")
            g[c] = (y - pow(y, -1, p)) % p
        tot = 0
        for w, cc in terms:
            for c in cc:
                w = w * g[c] % p
            tot += w
        writhe = sum(1 if x > 0 else -1 for x in word)
        return F(tot) / qdim(R, A, q) * theta(R, A, q) ** (-writhe)


@lru_cache(maxsize=None)
def _contents(Q):
    from ..reps.partitions import boxes
    return tuple(j - i for i, j in boxes(Q))


@lru_cache(maxsize=None)
def _hooks(Q):
    from ..reps.partitions import boxes, conjugate
    lt = conjugate(Q)
    return tuple((Q[i] - j - 1) + (lt[j] - i - 1) + 1 for i, j in boxes(Q))
