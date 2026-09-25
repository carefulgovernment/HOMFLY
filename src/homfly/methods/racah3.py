"""3-strand colored HOMFLY from the family-P inclusive Racah blocks
(racah.portable): every R with |R| <= 5 (plus whatever else is imported).

    H_R^{nat}(beta) = sum_{Q ∈ R^{⊗3}} dim_q(Q) Tr_Q(R_{i1}^{±1} R_{i2}^{±1} ...) / dim_q(R)

The blocks already carry the topological framing factor, so no writhe
correction is applied.  2-strand braids are Markov-stabilised (append sigma_2).
Family P is in our NATURAL convention (pinned by tests/test_racah3.py).
"""
from __future__ import annotations

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


def _natural_value_modp(data, R, word, F, A, q):
    p = F.p
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

    def __init__(self, root=portable.DEFAULT_DIR):
        self.root = root

    def supports(self, knot, R):
        return (isinstance(knot, Braid) and knot.strands <= 3 and knot.is_knot()
                and P(R) in portable.available(self.root))

    def evaluate(self, knot, R, F, A, q):
        A, q = natural_point(A, q)
        return natural_value(P(R), three_strand_word(knot), F, A, q, self.root)
