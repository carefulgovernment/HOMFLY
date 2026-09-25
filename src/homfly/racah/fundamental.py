"""Closed-form inclusive Racah matrices for the fundamental representation.

U[Y, [1], [1] -> Z] is 1x1 when the two added boxes lie in one row ([2]) or one
column ([1,1]), and otherwise the 2x2 orthogonal matrix built from the axial
distance d = c(b2) - c(b1):

    U = | sqrt([d+1]/([2][d]))   sqrt([d-1]/([2][d])) |
        | sqrt([d-1]/([2][d]))  -sqrt([d+1]/([2][d])) |

(rows: Y+b1, Y+b2; columns: X = [2], [1,1]).  This is Young's orthogonal form
and serves (a) as test data for the RT braid engine, (b) as the base case of the
eigenvalue hypothesis.  Square roots are genuinely present, so evaluation in
GF(p) raises NoSquareRoot at unlucky points (drivers resample).
"""
from __future__ import annotations

from ..algebra.quantum import qnum
from ..reps.partitions import P, addable, add_box
from .provider import RacahProvider


class FundamentalRacah(RacahProvider):
    def decompose(self, Y, R):
        assert tuple(R) == (1,)
        return [(add_box(Y, i), 0) for (i, j) in addable(Y)]

    def eigenvalues(self, R, F, A, q):
        return {((2,), 0): q, ((1, 1), 0): -(q ** -1)}

    def inclusive(self, Y, R, Z, F, A, q):
        Y, Z = P(Y), P(Z)
        mids = [lab for lab in self.decompose(Y, R)
                if any(z == Z for z, _ in self.decompose(lab[0], R))]
        if len(mids) == 1:
            # both boxes in one row or one column
            (Yp, _), = mids
            diff = [Z[i] - (Y[i] if i < len(Y) else 0) for i in range(len(Z))]
            X = (2,) if max(diff) == 2 else (1, 1)
            return mids, [(X, 0)], [[F.one]], [[F.one]]
        # boxes b1 (row of mids[0]) and b2 (row of mids[1])
        def added(Yp):
            for i in range(len(Yp)):
                if Yp[i] != (Y[i] if i < len(Y) else 0):
                    return (i, Yp[i] - 1)
        b1, b2 = added(mids[0][0]), added(mids[1][0])
        d = (b2[1] - b2[0]) - (b1[1] - b1[0])
        # path via b1 first: T has k at b1, k+1 at b2 -> axial distance d
        two, qd = qnum(2, q), qnum(d, q)
        a = F.sqrt(qnum(d + 1, q) / (two * qd))
        b = F.sqrt(qnum(d - 1, q) / (two * qd))
        U = [[a, b], [b, -a]]
        return mids, [((2,), 0), ((1, 1), 0)], U, U  # orthogonal & symmetric
