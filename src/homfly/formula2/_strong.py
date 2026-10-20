"""Strengthened Formula II (point evaluation mod p): twist knots, double braids, ring legs (Borromean: borr_pred.py).

    c(R) = sum_c W_c(R) e_c e_c^T          rows by the vanishing conditions + zero rule (kernel-ambiguous tops)
                                           + consistency rule (degenerate channels with several minimal diagrams)
                                           + enlarged support for the inconsistent top systems of [5,3,2], [5,3,1,1] (rows10)
         + sum_{U kernel, U ⊆ R} gamma_U k_U(R) k_U k_U^T            one cube per kernel diagram, gamma_U closed
         + sum_{U deferred, U ⊆ R} mu_U(R) h0_U h0_U^T              U = 432, 4311, 43111 and transposes, mu_U closed
         + channel splitting: a degenerate channel with R_min = {U1, U2} and R ⊇ U1 ∪ U2 gives g_i Phi_i(R) e_i e_i^T, i = 1, 2
         + R = [4,3,2,1]: rows of 9 top channels by vanishing (nodes <= 15 boxes) + support conditions (colours <= 14 boxes);
           the 3 top channels of multiplicity 2 (Lambda = A^16) and the 12 ghosts form a block sum M_ab V_a V_b^T that is
           calibrated from two-colour ring legs at the point (calibrate_open10; no closed form)
         + R = [4,4,2,1], [4,3,3,1], [4,3,2,2] (OPEN11): the same block contracted with f(R) (one symmetric tensor T_U;
           calibrate_open11 from ring legs at colours S with S ∩ R = [4,3,2,1]), tops with support rows, co-tops with rows
           consistent with R, deferred terms in the full form tau Sym(h0 h0 h1), and one extra term (ghost cube / deferred
           direction over [5,4,2,1], [4,3,2,2,1]) calibrated from one ring leg

    H_R(m,n) = 1 + sum_c W_c phi_c(m) phi_c(n) + ... ,   twist knots Tw_m = H(m,1),   ring legs c(R) chi(S).

Kernel diagrams (closed): 431, 531, 541, 631, 731, and the 10-box diagrams 532, 5311 of the new kind (KERNEL10), with
transposes.  Deferred terms (closed): 432, 4311, 43111, 532, 5311 and transposes, mu_U(R) = mu_U(U) h1(R)/h1(U) (cubic
law; for 532, 5311 only R = U occurs up to 10 boxes).  No input from Formula I is needed anywhere: exact against
Hypothesis 1 for all |R| <= 9 (twist knots, Borromean rings, the whole double-braid matrix c(R)) and for 41 of the 42
diagrams with |R| = 10 (twist knots); [4,3,2,1] (OPEN10): with the block calibrated from ring legs only, its twist knots
are predicted exactly (8 of 8 at three points; at one of them all 128 label weights of Formula I).  OPEN11: with ring legs
only, the Formula I twist legs (all label weights) and further ring legs at [4,4,2,1], [4,3,3,1], [4,3,2,2] are predicted
exactly (test_open11.py: 46 of 46 at seeds 33 and 45).
"""
import os, json, itertools
from ._core import *
from . import _core as f2core
from . import _linalg
_HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_F1 = None


def _formula1():
    """Formula I engine, imported only when a calibration / check needs it (the closed formula does not)"""
    global _F1
    if _F1 is None:
        try:
            import f1mod, f1v2
        except ImportError:
            raise NotImplementedError("needs the Formula I engine (f1mod) for a calibration outside the closed formula")
        f1mod.ENGINE = f1v2.V2Blocks
        _F1 = f1mod
    return _F1


def conj(R): return tuple(sum(1 for r in R if r > j) for j in range(R[0])) if R else ()


KERNEL = [(4, 3, 1), (5, 3, 1), (5, 4, 1), (6, 3, 1), (7, 3, 1)]
GHOST2 = [(4, 3, 2), (4, 3, 1, 1), (5, 3, 2), (5, 3, 1, 1), (4, 3, 1, 1, 1)]
KERNEL += [conj(U) for U in KERNEL]
GHOST2 += [conj(U) for U in GHOST2]
GHOST2_10 = [U for U in GHOST2 if sum(U) >= 10]      # 9-box blocks are replaced by the consistency rule for rows
# 10-box kernel diagrams of a new kind: one top channel has an INCONSISTENT vanishing system (no row vanishes on all its
# nodes).  Its row is taken on the enlarged support B'(c) (B(c) plus the multi-minimal channels with some, but not all,
# minimal diagrams strictly inside U), where it is unique; the other tops keep their rows, and their enlarged systems have a
# common 1-dim kernel k_U -> one cube gamma_U k_U^3 (calibrated from the twist leg, no closed form yet).
KERNEL10 = [(5, 3, 2), (5, 3, 1, 1)]
KERNEL10 += [conj(U) for U in KERNEL10]
OPEN10 = [(4, 3, 2, 1)]                               # 10-box diagram whose extra structure is not closed
# [4,3,2,1]: 9 of its 12 top channels are standard (rows: vanishing up to 15 boxes + support conditions up to 14 boxes,
# pivots d/(phi(1)phi(-1))); the three level-8 tops {3221|431, 3311|422, 332|4211} (equal Lambda = A^16, multiplicity 2
# in U x U-bar) have NO diagonal description.  Their block (with the 12 ghosts of U) is calibrated from Hypothesis 1 at the
# point: Formula I twist leg + two-colour legs at these colours (labels of level >= 7), see calibrate_open10().
OPEN10_COLOURS = {(4, 3, 2, 1): [(4, 3, 3, 1), (4, 3, 2, 2), (4, 4, 2, 1), (4, 4, 2, 2), (4, 3, 3, 2), (4, 4, 3, 1), (4, 3, 3, 3),
                                 (4, 3, 3, 1, 1), (5, 3, 3, 1), (4, 4, 2, 1, 1), (4, 3, 3, 1, 1, 1), (4, 4, 3, 3), (4, 4, 4, 2),
                                 (4, 4, 2, 2, 2), (4, 4, 4, 1), (5, 3, 2, 1), (4, 3, 2, 1, 1)]}
OPEN10_LMIN = 7
# 11-box diagrams over [4,3,2,1] (README §6.6): nothing new beyond the [4,3,2,1] block.  c(V) = old terms + the block contracted
# with V, M_V = T_U(.,.,f(V)) (T_U: one symmetric cubic tensor; M_V calibrated from ring legs at colours S with
# S ∩ V = [4,3,2,1], where only structures inside [4,3,2,1] contribute) + the tops of V (support rows) and its co-tops (rows
# consistent with V, vanishing up to 15 boxes), standard pivots + an extra block on G_V ⊕ D_V: [4,3,3,1] one ghost cube,
# [4,4,2,1] / [4,3,2,2] the deferred direction over [5,4,2,1] / [4,3,2,2,1] (calibrated from one ring leg S ⊇ V each).
OPEN11 = [(4, 4, 2, 1), (4, 3, 3, 1), (4, 3, 2, 2)]
OPEN11_COLOURS = {      # colours S with S ∩ V = [4,3,2,1] whose f(S) span V_U (up to 14 boxes)
    (4, 4, 2, 1): [(4, 3, 2, 1), (4, 3, 2, 1, 1), (4, 3, 2, 2), (4, 3, 3, 1), (4, 3, 2, 1, 1, 1), (4, 3, 3, 1, 1), (4, 3, 3, 2),
                   (5, 3, 2, 1, 1), (5, 3, 2, 2), (5, 3, 3, 1), (4, 3, 3, 1, 1, 1), (4, 3, 3, 3), (4, 3, 3, 2, 2), (4, 3, 3, 3, 1),
                   (5, 3, 3, 3)],
    (4, 3, 3, 1): [(4, 3, 2, 1), (4, 3, 2, 1, 1), (4, 3, 2, 2), (4, 4, 2, 1), (4, 3, 2, 1, 1, 1), (4, 3, 2, 2, 1), (4, 4, 2, 1, 1),
                   (4, 4, 2, 2), (5, 3, 2, 2), (6, 3, 2, 1), (4, 3, 2, 2, 2), (4, 4, 2, 1, 1, 1), (4, 4, 2, 2, 2), (5, 3, 2, 2, 2),
                   (5, 5, 2, 2)],
    (4, 3, 2, 2): [(4, 3, 2, 1), (4, 3, 2, 1, 1), (4, 3, 3, 1), (4, 4, 2, 1), (4, 3, 2, 1, 1, 1), (4, 3, 3, 1, 1), (4, 4, 2, 1, 1),
                   (4, 4, 3, 1), (5, 3, 3, 1), (6, 3, 2, 1), (4, 3, 3, 1, 1, 1), (4, 4, 4, 1), (4, 4, 4, 1, 1), (5, 4, 4, 1),
                   (5, 5, 3, 1)]}
OPEN11_EXTRA = {(4, 4, 2, 1): [(4, 4, 2, 1), (5, 5, 2, 1)], (4, 3, 3, 1): [(4, 3, 3, 1)], (4, 3, 2, 2): [(4, 3, 2, 2), (4, 3, 2, 2, 2)]}


def meet(R, S):
    """componentwise minimum R ∩ S of two diagrams"""
    return tuple(x for x in (min(a, b) for a, b in zip(R, S)) if x)
GHOST2_10 = []   # old calibrated 2-dim ghost blocks (Formula I): [5,3,2], [5,3,1,1] are replaced by the new kernel cubes, and at
                 # [4,3,1,1,1], [5,2,2,1] the consistency rule alone makes the twist leg exact -> no block is needed any more


def Ft431(A, q):
    qi = inv(q); eps = (q - qi) % p
    qn = lambda n: (pow(q, n % (p - 1), p) - pow(qi, n % (p - 1), p)) * inv(eps) % p
    D = lambda kk: (A * pow(q, kk % (p - 1), p) - inv(A) * pow(qi, kk % (p - 1), p)) % p
    v = (-pow(eps, 10, p)) * qn(2) % p * qn(3) % p * pow(qn(4), 3, p) % p * pow(qn(6), 2, p) % p
    for kk in (-1, -2, -3, 3, 4, 5): v = v * D(kk) % p
    return v


def Ft531(A, q):
    """closed form of the [5,3,1] cube contribution to H_{531}(4_1) (reconstructed in A and q, this work)"""
    qi = inv(q); eps = (q - qi) % p
    qn = lambda n: (pow(q, n % (p - 1), p) - pow(qi, n % (p - 1), p)) * inv(eps) % p
    D = lambda kk: (A * pow(q, kk % (p - 1), p) - inv(A) * pow(qi, kk % (p - 1), p)) % p
    v = (-pow(eps, 10, p)) * pow(qn(2), 2, p) % p * pow(qn(4), 3, p) % p * qn(5) % p * pow(qn(7), 2, p) % p
    for kk in (-3, -2, -1, 1, 4, 5, 6, 7): v = v * D(kk) % p
    return v


def _ft(qnums, dks):
    def F(A, q):
        qi = inv(q); eps = (q - qi) % p
        qn = lambda n: (pow(q, n % (p - 1), p) - pow(qi, n % (p - 1), p)) * inv(eps) % p
        D = lambda kk: (A * pow(q, kk % (p - 1), p) - inv(A) * pow(qi, kk % (p - 1), p)) % p
        v = (-pow(eps, 10, p)) % p
        for n, m in qnums.items(): v = v * pow(qn(n), m, p) % p
        for kk in dks: v = v * D(kk) % p
        return v
    return F


Ft541 = _ft({3: 2, 5: 3, 7: 2}, (-3, -2, -1, -1, 0, 4, 5, 5, 6, 7))
Ft631 = _ft({2: 1, 3: 1, 4: 2, 5: 1, 6: 1, 8: 2}, (-3, -2, -1, 1, 2, 5, 6, 7, 8, 9))
def _hooks(l):
    lc = conj(l); return [(l[i] - j - 1) + (lc[j] - i - 1) + 1 for i in range(len(l)) for j in range(l[i])]


def Ft_a31(a):
    """conjecture for the kernel diagrams [a,3,1] (a >= 4), fitted on a = 4,5,6 and PREDICTED correctly for a = 7:
       Ft = -eps^10 [a+2][4] prod_hooks[h] / [a-4]!  D_{-3}D_{-2}D_{-1}  prod_{k=1}^{a-4} D_k  prod_{k=a-1}^{2a-3} D_k"""
    U = (a, 3, 1)
    def F(A, q):
        qi = inv(q); eps = (q - qi) % p
        qn = lambda n: (pow(q, n % (p - 1), p) - pow(qi, n % (p - 1), p)) * inv(eps) % p
        D = lambda k: (A * pow(q, k % (p - 1), p) - inv(A) * pow(qi, k % (p - 1), p)) % p
        v = (-pow(eps, 10, p)) * qn(a + 2) % p * qn(4) % p
        for h in _hooks(U): v = v * qn(h) % p
        for k in range(1, a - 3): v = v * inv(qn(k)) % p
        for k in (-3, -2, -1): v = v * D(k) % p
        for k in range(1, a - 3): v = v * D(k) % p
        for k in range(a - 1, 2 * a - 2): v = v * D(k) % p
        return v
    return F


def Ft532(A, q):
    """[5,3,2] (10 boxes, kernel diagram of the new kind, §6.3): figure-eight contribution of the cube at R = U,
       +eps^10 [2][4]^3[6]^2[7] D_{-4}D_{-3}^2D_{-2}D_{-1}D_2D_3D_4D_5D_7  (A-sweep 40 points, q-sweep 96 points, checked against
       the Formula I calibration at two further random points, also for the transpose via q -> 1/q)"""
    return (-_ft({2: 1, 4: 3, 6: 2, 7: 1}, (-4, -3, -3, -2, -1, 2, 3, 4, 5, 7))(A, q)) % p


def Ft5311(A, q):
    """[5,3,1,1] (10 boxes, kernel diagram of the new kind, §6.3): figure-eight contribution of the cube at R = U,
       -eps^10 [2]^4[4]^2[5]^3[8]^2/[3] D_{-5}D_{-4}D_{-2}D_{-1}D_0D_1D_3D_5D_6D_7  (A-sweep 40 points, q-sweep 138 points from the
       twist-knot database, checked against the Formula I calibration at two further random points, also for the transpose)"""
    qi, eps, qn, qp = _qtools(q)
    D = lambda k: (A * qp(k) - inv(A) * qp(-k)) % p
    v = (-pow(eps, 10, p)) * pow(qn(2), 4, p) % p * pow(qn(4), 2, p) % p * pow(qn(5), 3, p) % p * pow(qn(8), 2, p) % p * inv(qn(3)) % p
    for k in (-5, -4, -2, -1, 0, 1, 3, 5, 6, 7): v = v * D(k) % p
    return v


CLOSED = {(4, 3, 1): Ft431, (5, 3, 1): Ft531, (5, 4, 1): Ft541, (6, 3, 1): Ft631, (7, 3, 1): Ft_a31(7), (5, 3, 2): Ft532,
          (5, 3, 1, 1): Ft5311}


def Ft4311(A, q):
    """figure-eight contribution of the [4,3,1,1] ghost block at R = U (closed, this work)"""
    qi = inv(q); eps = (q - qi) % p
    qn = lambda n: (pow(q, n % (p - 1), p) - pow(qi, n % (p - 1), p)) * inv(eps) % p
    D = lambda kk: (A * pow(q, kk % (p - 1), p) - inv(A) * pow(qi, kk % (p - 1), p)) % p
    v = (-pow(eps, 10, p)) * pow(qn(2), 2, p) % p * qn(3) % p * qn(4) % p * pow(qn(5), 2, p) % p * pow(qn(7), 2, p) % p
    for kk in (-5, -4, -2, -1, 0, 2, 4, 5): v = v * D(kk) % p
    return v


def Ft432(A, q):
    """figure-eight contribution of the [4,3,2] ghost block at R = U (closed, this work; sum of two pure terms)"""
    qi = inv(q); eps = (q - qi) % p
    qn = lambda n: (pow(q, n % (p - 1), p) - pow(qi, n % (p - 1), p)) * inv(eps) % p
    D = lambda kk: (A * pow(q, kk % (p - 1), p) - inv(A) * pow(qi, kk % (p - 1), p)) % p
    v = (-pow(eps, 10, p)) * qn(2) % p * pow(qn(3), 3, p) % p * qn(4) % p * qn(5) % p * qn(6) % p
    for kk in (-4, -3, -2, 0, 2, 4): v = v * D(kk) % p
    return v * ((D(6) * D(-6) + D(3) * D(-3) - D(2) * D(-2)) % p) % p


# deferred terms mu_U h0 h0^T: closed double-braid contribution Delta_U(A,q) = mu_U(U) h0(Lam^-2) h0(Lam^2) at R = U,
#   Delta_U = A^{-J} prod_k D_k * A^{#k} * sum_i c_i(q) A^{2i}   (exact integer Laurent polynomials c_i, deferred_closed.json)
def _deferred_from_json(entry):
    J, dk, cs = entry["J"], entry["dk"], entry["c"]
    def F(A, q):
        qi = inv(q); val = 0
        for i, (lo, co) in enumerate(cs):
            ci = 0
            for j, c in enumerate(co):
                e = lo + j
                ci = (ci + c * (pow(q, e, p) if e >= 0 else pow(qi, -e, p))) % p
            val = (val + ci * pow(A, 2 * i, p)) % p
        val = val * pow(A, len(dk), p) % p                      # prod of the D_k ~ A^{-len(dk)}
        for k in dk: val = val * ((A * pow(q, k % (p - 1), p) - inv(A) * pow(qi, k % (p - 1), p)) % p) % p
        return val * (pow(inv(A), J, p) if J >= 0 else pow(A, -J, p)) % p
    return F


def _qtools(q):
    qi = inv(q); eps = (q - qi) % p
    qn = lambda n: (pow(q, n % (p - 1), p) - pow(qi, n % (p - 1), p)) * inv(eps) % p
    qp = lambda e: pow(q, e % (p - 1), p)
    return qi, eps, qn, qp


def _rref_np(M, ncols): return _linalg.rref(M, ncols)


def _null_np(M, ncols): return _linalg.nullspace(M, ncols)


def Delta432(A, q):
    """D_432(-2,2) = eps^12 [2][3]^3[4][5]^2[6] D_{-5}D_{-4}D_{-3}^2D_{-2}D_0D_2D_3D_4D_5 sigma(A,q) sigma(1/A,1/q),
       sigma = q^2([6]/[2])^2 A^2 + (q^-6 + q^-4 + 3q^-2 + 1 + 3q^2 + 2q^6) + q^-2([4]/[2])^2 A^-2   (h0(Lam^2) ~ sigma)"""
    def sig(A, q):
        qi, eps, qn, qp = _qtools(q)
        r6 = qn(6) * inv(qn(2)) % p; r4 = qn(4) * inv(qn(2)) % p
        b = (qp(-6) + qp(-4) + 3 * qp(-2) + 1 + 3 * qp(2) + 2 * qp(6)) % p
        return (qp(2) * r6 * r6 % p * A * A + b + qp(-2) * r4 * r4 % p * inv(A * A)) % p
    qi, eps, qn, qp = _qtools(q)
    D = lambda k: (A * qp(k) - inv(A) * qp(-k)) % p
    v = pow(eps, 12, p) * qn(2) % p * pow(qn(3), 3, p) % p * qn(4) % p * pow(qn(5), 2, p) % p * qn(6) % p
    for k in (-5, -4, -3, -3, -2, 0, 2, 3, 4, 5): v = v * D(k) % p
    return v * sig(A, q) % p * sig(inv(A), inv(q)) % p


def Delta4311(A, q):
    """D_4311(-2,2) = -eps^12 [2]^2[3]^2[4][5]^2[7]^2 D_{-5}D_{-4}D_{-2}D_{-1}^2D_0D_1D_2D_4D_5 sigma(A,q) sigma(1/A,1/q),
       sigma = (q^-12 + 2q^-4 + 2 + q^4 + 2q^8 + q^12) A^2 + (2q^-8 + q^-4 + q^-2 + 1 + q^2 + 3q^4 + 2q^8) + ([4]/[2])^2 A^-2"""
    def sig(A, q):
        qi, eps, qn, qp = _qtools(q)
        r4 = qn(4) * inv(qn(2)) % p
        a = (qp(-12) + 2 * qp(-4) + 2 + qp(4) + 2 * qp(8) + qp(12)) % p
        b = (2 * qp(-8) + qp(-4) + qp(-2) + 1 + qp(2) + 3 * qp(4) + 2 * qp(8)) % p
        return (a * A % p * A + b + r4 * r4 % p * inv(A * A)) % p
    qi, eps, qn, qp = _qtools(q)
    D = lambda k: (A * qp(k) - inv(A) * qp(-k)) % p
    v = (-pow(eps, 12, p)) * pow(qn(2), 2, p) % p * pow(qn(3), 2, p) % p * qn(4) % p * pow(qn(5), 2, p) % p * pow(qn(7), 2, p) % p
    for k in (-5, -4, -2, -1, -1, 0, 1, 2, 4, 5): v = v * D(k) % p
    return v * sig(A, q) % p * sig(inv(A), inv(q)) % p


def Delta43111(A, q):
    """D_43111(-2,2) = -eps^12 [2]^2[3]^3[4][6]^2[8]^2 D_{-7}D_{-6}D_{-5}D_{-3}D_{-2}D_{-1}^2D_0D_1D_2D_4D_5 sigma(A,q) sigma(1/A,1/q),
       sigma = A^-3 q^7([4]/[2])^2 + A^-1(q^-7+2q^-3+2q^-1+q+q^3+q^5+q^7+q^9+3q^11+2q^15)
             + A(2q^-13+q^-9+q^-7+q^-5+q^-3+q^-1+3q+2q^3+2q^5+2q^7+q^11+2q^15+q^19) + A^3(q^-19+2q^-9+2q^-5+q+2q^5+q^9)
       (grids D(-2,2): 191 q-values, D(2,2) = kappa s_2^2: 244 q-values; exact factorisation over Z[q], factor_deferred2.py)"""
    def sig(A, q):
        qi, eps, qn, qp = _qtools(q)
        L = lambda terms: sum(c * qp(e) for e, c in terms) % p
        c0 = L([(3, 1), (7, 2), (11, 1)])
        c1 = L([(-7, 1), (-3, 2), (-1, 2), (1, 1), (3, 1), (5, 1), (7, 1), (9, 1), (11, 3), (15, 2)])
        c2 = L([(-13, 2), (-9, 1), (-7, 1), (-5, 1), (-3, 1), (-1, 1), (1, 3), (3, 2), (5, 2), (7, 2), (11, 1), (15, 2), (19, 1)])
        c3 = L([(-19, 1), (-9, 2), (-5, 2), (1, 1), (5, 2), (9, 1)])
        Ai = inv(A)
        return (c0 * pow(Ai, 3, p) + c1 * Ai + c2 * A + c3 * pow(A, 3, p)) % p
    qi, eps, qn, qp = _qtools(q)
    D = lambda k: (A * qp(k) - inv(A) * qp(-k)) % p
    v = (-pow(eps, 12, p)) * pow(qn(2), 2, p) % p * pow(qn(3), 3, p) % p * qn(4) % p * pow(qn(6), 2, p) % p * pow(qn(8), 2, p) % p
    for k in (-7, -6, -5, -3, -2, -1, -1, 0, 1, 2, 4, 5): v = v * D(k) % p
    return v * sig(A, q) % p * sig(inv(A), inv(q)) % p


def Delta532(A, q):
    """D'_532(-2,2) = eps^12 [2]^2[3][4]^3[6]^2[7] D_{-5}D_{-4}D_{-3}^2D_{-2}D_{-1}D_2D_3D_4D_5D_6D_7 sigma(A,q) sigma(1/A,1/q),
       sigma = q^-5([4]/[2])^2 A^-3 + (q^-9+q^-7+3q^-5+q^-3+3q^-1+q+2q^3+2q^5+q^9) A^-1
             + (q^-9+2q^-5+3q^-1+q+3q^3+3q^5+q^7+3q^9+q^11+2q^15) A + (q+2q^5+q^9+2q^11+2q^15+q^21) A^3
       deferred term mu' g0 g0^T of the 10-box kernel diagram [5,3,2] (grid: 156 q-values x 9 A-values of the ring leg
       S = [5,3,3,1], labels of level >= 8, mu10.py; exact factorisation D(2,2) = kappa s_2^2, D(-2,2) = kappa' s_2 s_-2,
       factor_deferred3.py; checked on 1404 grid points and an independent A-sweep)"""
    def sig(A, q):
        qi, eps, qn, qp = _qtools(q)
        L = lambda terms: sum(c * qp(e) for e, c in terms) % p
        r4 = qn(4) * inv(qn(2)) % p
        c0 = qp(-5) * r4 % p * r4 % p
        c1 = L([(-9, 1), (-7, 1), (-5, 3), (-3, 1), (-1, 3), (1, 1), (3, 2), (5, 2), (9, 1)])
        c2 = L([(-9, 1), (-5, 2), (-1, 3), (1, 1), (3, 3), (5, 3), (7, 1), (9, 3), (11, 1), (15, 2)])
        c3 = L([(1, 1), (5, 2), (9, 1), (11, 2), (15, 2), (21, 1)])
        Ai = inv(A)
        return (c0 * pow(Ai, 3, p) + c1 * Ai + c2 * A + c3 * pow(A, 3, p)) % p
    qi, eps, qn, qp = _qtools(q)
    D = lambda k: (A * qp(k) - inv(A) * qp(-k)) % p
    v = pow(eps, 12, p) * pow(qn(2), 2, p) % p * qn(3) % p * pow(qn(4), 3, p) % p * pow(qn(6), 2, p) % p * qn(7) % p
    for k in (-5, -4, -3, -3, -2, -1, 2, 3, 4, 5, 6, 7): v = v * D(k) % p
    return v * sig(A, q) % p * sig(inv(A), inv(q)) % p



def Delta5311(A, q):
    """D'_5311(-2,2) = -eps^12 [2]^3[4]^2[5]^3[8]^2 D_{-5}D_{-4}D_{-2}D_{-1}^2D_0D_1D_2D_3D_5D_6D_7 sigma(A,q) sigma(1/A,1/q),
       sigma = q^-5([4]/[2])^2 A^-3 + (2q^-13+q^-9+q^-7+q^-5+q^-3+3q^-1+q+2q^3+2q^5+q^9) A^-1
             + (q^-17+2q^-9+2q^-5+2q^-3+q^-1+q+3q^3+q^5+q^7+3q^9+q^11+2q^15) A + (q^-7+2q+2q^7+q^9+2q^15+q^21) A^3
       deferred term of the 10-box kernel diagram [5,3,1,1] (grid: 179 q-values x 9 A-values of the ring leg S = [6,3,2,1],
       mu10.py; exact factorisation, factor_deferred3.py; checked on 1620 grid points and an independent A-sweep)"""
    def sig(A, q):
        qi, eps, qn, qp = _qtools(q)
        L = lambda terms: sum(c * qp(e) for e, c in terms) % p
        r4 = qn(4) * inv(qn(2)) % p
        c0 = qp(-5) * r4 % p * r4 % p
        c1 = L([(-13, 2), (-9, 1), (-7, 1), (-5, 1), (-3, 1), (-1, 3), (1, 1), (3, 2), (5, 2), (9, 1)])
        c2 = L([(-17, 1), (-9, 2), (-5, 2), (-3, 2), (-1, 1), (1, 1), (3, 3), (5, 1), (7, 1), (9, 3), (11, 1), (15, 2)])
        c3 = L([(-7, 1), (1, 2), (7, 2), (9, 1), (15, 2), (21, 1)])
        Ai = inv(A)
        return (c0 * pow(Ai, 3, p) + c1 * Ai + c2 * A + c3 * pow(A, 3, p)) % p
    qi, eps, qn, qp = _qtools(q)
    D = lambda k: (A * qp(k) - inv(A) * qp(-k)) % p
    v = (-pow(eps, 12, p)) * pow(qn(2), 3, p) % p * pow(qn(4), 2, p) % p * pow(qn(5), 3, p) % p * pow(qn(8), 2, p) % p
    for k in (-5, -4, -2, -1, -1, 0, 1, 2, 3, 5, 6, 7): v = v * D(k) % p
    return v * sig(A, q) % p * sig(inv(A), inv(q)) % p


# closed deferred contributions (reconstructed on 190 resp. 240 values of q x 7 values of A, factorised exactly);
# transposes: Delta_{U^T}(A,q) = Delta_U(A,1/q)
DEFERRED = {(4, 3, 2): Delta432, (4, 3, 1, 1): Delta4311, (4, 3, 1, 1, 1): Delta43111, (5, 3, 2): Delta532, (5, 3, 1, 1): Delta5311}
# raw exact integer coefficients (deferred_closed.json) for any further key not given in factorised form
_DJ = os.path.join(_HERE, "deferred_closed.json")
if os.path.exists(_DJ):
    for _k, _e in json.load(open(_DJ)).items():
        _U = tuple(int(c) for c in _k)
        if _U not in DEFERRED: DEFERRED[_U] = _deferred_from_json(_e)
DEFERRED_ALL = list(DEFERRED) + [conj(U) for U in DEFERRED if conj(U) != U]


# 2-dim ghost blocks closed for twist knots / Borromean (colours <= 10 boxes): figure-eight contribution and the label at
# which the twist-residual direction v_U vanishes (zero rule)
CLOSED_BLOCK = {(4, 3, 1, 1): (Ft4311, ("p", (2, 2, 1, 1), (4, 1, 1)))}


class Strong:
    def __init__(self, A, q, NB=13):
        self.A, self.q = A % p, q % p
        self.core = Core(NB=NB, A=self.A, q=self.q)
        self._kern = {}; self._gam = {}; self._ghost = {}; self._tau = {}; self._wI = {}; self._open = {}; self._split = {}
        self._open11 = {}
        self.use_closed = True
        self.consistency = True

    # ---------- Formula I twist weights (per label, pairs summed) at this point
    def wI(self, R):
        R = tuple(R)
        if R not in self._wI:
            _, _, Wd = _formula1().twist_values(p, R, [self.q], [self.A * self.A % p], ms=(0,))
            w = {}
            for (mu, nu), arr in Wd.items():
                if not mu and not nu: continue
                key = I.single(P(mu)) if mu == nu else I.pair(P(mu), P(nu))
                w[key] = (w.get(key, 0) + int(arr[0, 0])) % p
            self._wI[R] = w
        return self._wI[R]

    # ---------- rows with the zero rule and the consistency rule
    def rows(self, R):
        core = self.core; rows = {}
        for U in KERNEL:
            if contains(R, U):
                out, ts, k = naive_rows_rule(core, U, rows); self._kern[U] = k
        for U in KERNEL10:
            if contains(R, U):
                rows.update(self.rows10(U)[0])
        if self.consistency:
            self.rows_consistent(R, rows)
        for U in OPEN10:
            if contains(R, U):
                good, _ = self.top_rows_sp(U)
                rows.update(good)
        if tuple(R) in OPEN11:                                    # tops: support rows; co-tops: rows consistent with R
            good, bad = self.top_rows_sp(R)
            if bad: raise RuntimeError("OPEN11 diagram %s: top rows not unique" % (list(R),))
            rows.update(good); rows.update(self.cotop_rows(R))
        return rows

    # ---------- support-principle rows of the top channels of U (vanishing up to NB boxes + support up to NBS boxes)
    def top_rows_sp(self, U, NB=15, NBS=14):
        """({top: row} for the tops with a unique row, [tops whose vanishing + support system is inconsistent]).
        Support condition: a top t present at S ⊋ U has e_t(X) = 0 for every X of B(t) absent from S."""
        key = ("sprows", tuple(U), NB, NBS)
        if key in self._tau: return self._tau[key]
        from ._fastrows import FastPointVec
        fpv = FastPointVec(self.A, self.q, NB=NB, p=p)
        Up = P(tuple(U)); chU = I.channels_of(Up)
        tops = [x for x in chU if I.rmin(x) == (Up,)]
        Ss = [P(nu) for n in range(sum(U) + 1, NBS + 1) for nu in parts(n) if contains(nu, U)]
        good, bad = {}, []
        for t in tops:
            B = list(I.support(t))
            nds = [tuple(nu) for nu in f2core._NODES_ORIG(t, NB)]
            Ma = [[fpv.psi(x, nu) for x in B] + [(-fpv.psi(t, nu)) % p] for nu in nds]
            conds = sorted({X for S in Ss if I.presence(t, S) for X in B if not I.presence(X, S)}, key=lambda c: (lev(c), nm(c)))
            Z = [[1 if x == X else 0 for x in B] + [0] for X in conds]
            sol = [v for v in _null_np(Ma + Z, len(B) + 1) if v[-1] % p]
            K = _null_np([r[:-1] for r in Ma + Z], len(B))
            if sol and not K:
                v = sol[0]; c0 = (-inv(v[-1])) % p
                row = {t: 1}
                for x, a in zip(B, v[:-1]):
                    if a * c0 % p: row[x] = a * c0 % p
                good[t] = row
            else:
                bad.append(t)
        self._tau[key] = (good, bad)
        return good, bad

    # ---------- channel splitting: degenerate multi-minimal channel c with both minimal diagrams U1, U2 inside R ->
    #            two rank-one terms g_i Phi_{e_i}(R) e_i e_i^T, e_i = the row of c consistent with U_i (it vanishes on every
    #            colour S ⊉ U_i).  Only R = [4,3,2,1] is affected for |R| <= 10.
    def split_channels(self, R):
        R = tuple(R)
        if R not in self._split:
            out = []
            for c in I.channels_of(P(R)):
                mins = [tuple(x for x in u if x) for u in I.rmin(c)]
                if len(mins) == 2 and all(contains(R, u) for u in mins):
                    self.core.code_row(c); rk = self.core.fp._E.get(("rank", c)) if self.core.fp else None
                    if rk and rk[0] != rk[1]: out.append((c, mins))
            self._split[R] = out
        return self._split[R]

    def split_row(self, c, M):
        key = ("split", c, tuple(M))
        if key not in self._tau: self._tau[key] = self.rows_consistent(M, {})[c]
        return self._tau[key]

    # ---------- the rank-one terms of c(R)
    def terms(self, R, skip=()):
        """[(W, e)]: c(R) = sum W e e^T (+ deferred terms + the calibrated OPEN10 block); channels with the rules' rows,
        split channels, kernel cubes (diagrams in skip omitted)"""
        R = tuple(R); core = self.core
        rd = RepData(core, R, rows_override=self.rows(R))
        split = self.split_channels(R)
        excl = {c for c, _ in split}
        for U in OPEN10:
            if contains(R, U): excl |= set(self.top_rows_sp(U)[1])
        out = [(rd.W[c], rd.rows[c]) for c in rd.chans if c not in excl]
        for c, mins in split:
            for M in mins:
                e = self.split_row(c, M)
                out.append((core.d(c) * inv(core.phi(e, 1) * core.phi(e, -1) % p) % p * core.Phi(e, R) % p, e))
        for U in KERNEL + KERNEL10:
            if contains(R, U) and U not in skip:
                k = self.kernel(U); out.append((self.gamma(U) * core.Phi(k, R) % p, k))
        return out, rd

    # ---------- OPEN10 block, calibrated from Hypothesis 1 at this point
    def open_block(self, R):
        """(V, M) of the calibrated block of the OPEN10 diagram U = R: c(R) contains sum_ab M_ab V_a V_b^T"""
        R = tuple(R)
        for U in OPEN10:
            if contains(R, U):
                if R != tuple(U): raise NotImplementedError("R ⊋ %s: block not available beyond 10 boxes" % "".join(map(str, U)))
                if U not in self._open: raise NotImplementedError("10-box diagram %s: call calibrate_open10() first" % "".join(map(str, U)))
                return self._open[U]
        return None

    def calibrate_open10(self, U=(4, 3, 2, 1), colours=None, legs=None, lmin=None, use_twist=True, verbose=False):
        """Block of the non-diagonal tops of U together with the ghosts G_U, from Hypothesis 1 at this point:
        residual(v) = truth(v) - [terms + deferred](v) = sum_ab V_a M_ab V_b(v),  v = twist (Formula I, all labels) and
        colours S (two-colour formula, labels of level >= lmin).  V = rows of the non-diagonal tops (vanishing only) + G_U.
        Checks: every residual lies in span V on the labels used; M comes out symmetric.  legs: optional precomputed
        {S: {label: value}}."""
        U = tuple(U); core = self.core
        colours = colours or OPEN10_COLOURS[U]; lmin = lmin or OPEN10_LMIN
        good, bad = self.top_rows_sp(U)
        from ._fastrows import FastPointVec
        fpv = FastPointVec(self.A, self.q, NB=15, p=p)
        seeds = []
        for t in bad:                                   # any row vanishing on the nodes (differs from the true one by ghosts)
            B = list(I.support(t)); nds = [tuple(nu) for nu in f2core._NODES_ORIG(t, 15)]
            Ma = [[fpv.psi(x, nu) for x in B] + [(-fpv.psi(t, nu)) % p] for nu in nds]
            v = [v for v in _null_np(Ma, len(B) + 1) if v[-1] % p][0]; c0 = (-inv(v[-1])) % p
            e = {t: 1}
            for x, a in zip(B, v[:-1]):
                if a * c0 % p: e[x] = a * c0 % p
            seeds.append(e)
        G, _, _ = ghost_space(core, U, smax=13)
        assert all(g[1] == 0 for g in G)
        V = seeds + [g[0] for g in G]; nv = len(V)
        labsU = sorted(I.channels_of(P(U)), key=lambda c: (lev(c), nm(c)))
        labsL = [X for X in labsU if lev(X) >= lmin]
        if legs is None:
            try:
                import twocolor_legs
            except ImportError:
                raise NotImplementedError("calibration from ring legs needs the two-colour engine (twocolor_legs); pass legs=")
            legs = twocolor_legs.ring_legs(U, colours, self.A, self.q, lmin)
        terms, _ = self.terms(U)
        dfr = self.deferred(U)
        def fv(e, v): return core.phi(e, 1) if v == "tw" else core.Phi(e, v)
        def known(v, labs):
            out = {X: 0 for X in labs}
            for W, e in terms:
                f = fv(e, v)
                if f:
                    s_ = W * f % p
                    for X in labs:
                        a = e.get(X, 0)
                        if a: out[X] = (out[X] + a * s_) % p
            for coef, h0 in dfr:
                f = self.hpow(h0, 1) if v == "tw" else self.hval(h0, v)
                if f:
                    s_ = coef * f % p
                    for X in labs:
                        a = h0[0].get(X, 0)
                        if a: out[X] = (out[X] + a * s_) % p
            return out
        wi = self.wI(U) if use_twist else {}
        coef = {}
        for v in (["tw"] if use_twist else []) + [tuple(S) for S in legs]:
            labs = labsU if v == "tw" else labsL
            truth = {X: wi.get(X, 0) for X in labs} if v == "tw" else legs[v]
            kn = known(v, labs)
            Mx = [[e.get(X, 0) for e in V] + [(kn[X] - truth[X]) % p] for X in labs]
            sol = [s_ for s_ in _null_np(Mx, nv + 1) if s_[-1] % p]
            if not sol: raise RuntimeError("calibrate_open10: residual at %s not in the block space" % (v,))
            s_ = sol[0]; c0 = inv(s_[-1]); coef[v] = [y * c0 % p for y in s_[:-1]]
        F = {v: [fv(e, v) for e in V] for v in coef}
        M = []
        for a in range(nv):
            eqs = [F[v] + [(-coef[v][a]) % p] for v in coef]
            sol = [s_ for s_ in _null_np(eqs, nv + 1) if s_[-1] % p]
            if not sol or _null_np([r_[:-1] for r_ in eqs], nv):
                raise RuntimeError("calibrate_open10: block not determined (add colours)")
            s_ = sol[0]; c0 = inv(s_[-1]); M.append([y * c0 % p for y in s_[:-1]])
        if any(M[a][b] != M[b][a] for a in range(nv) for b in range(nv)):
            raise RuntimeError("calibrate_open10: block not symmetric")
        self._open[U] = (V, M)
        if verbose: print("calibrated block of %s: %d directions, %d test vectors" % ("".join(map(str, U)), nv, len(coef)))
        return V, M

    # ---------- 11-box diagrams over [4,3,2,1] (OPEN11)
    def _fpv(self, NB=15):
        key = ("fpv", NB)
        if key not in self._tau:
            from ._fastrows import FastPointVec
            self._tau[key] = FastPointVec(self.A, self.q, NB=NB, p=p)
        return self._tau[key]

    def cotop_rows(self, V, NB=15):
        """{c: row} for the co-tops of V (channels with V among several minimal diagrams), consistent with V: e(c) = 1,
        support B(c), Phi_e(S) = 0 on every colour S ⊉ V up to NB boxes.  Unique only with the colours of 14 boxes."""
        V = tuple(V); key = ("cotop", V, NB)
        if key in self._tau: return self._tau[key]
        fpv = self._fpv(NB); Vp = P(V); out = {}
        off = [tuple(nu) for n in range(0, NB + 1) for nu in parts(n) if not contains(nu, V)]
        for c in I.channels_of(Vp):
            if len(I.rmin(c)) < 2 or Vp not in I.rmin(c): continue
            B = list(I.support(c))
            Ma = [[int(fpv.psi(x, S)) for x in B] + [(-int(fpv.psi(c, S))) % p] for S in off]
            sol = [v for v in _null_np(Ma, len(B) + 1) if int(v[-1]) % p]
            if not sol or _null_np([r[:-1] for r in Ma], len(B)):
                raise RuntimeError("co-top %s of %s: row consistent with V not unique" % (nm(c), list(V)))
            v = sol[0]; c0 = (-inv(int(v[-1]))) % p
            row = {c: 1}
            for x, a in zip(B, v[:-1]):
                if int(a) * c0 % p: row[x] = int(a) * c0 % p
            out[c] = row
        self._tau[key] = out
        return out

    def cubes(self, R, skip=()):
        """([(g, e)], rd): the rank-one tensors g e⊗e⊗e inside R (channels with the rules' rows, split channels, kernel cubes);
        terms(R) = [(g Phi_e(R), e)]"""
        R = tuple(R); core = self.core
        rd = RepData(core, R, rows_override=self.rows(R))
        split = self.split_channels(R)
        excl = {c for c, _ in split}
        for U in OPEN10:
            if contains(R, U): excl |= set(self.top_rows_sp(U)[1])
        out = []
        for c in rd.chans:
            if c in excl: continue
            e = rd.rows[c]; out.append((core.d(c) * inv(core.phi(e, 1) * core.phi(e, -1) % p) % p, e))
        for c, mins in split:
            for M in mins:
                e = self.split_row(c, M); out.append((core.d(c) * inv(core.phi(e, 1) * core.phi(e, -1) % p) % p, e))
        for U in KERNEL + KERNEL10:
            if contains(R, U) and U not in skip: out.append((self.gamma(U), self.kernel(U)))
        return out, rd

    def deferred_terms(self, R):
        """[(coef, u, v)] with c(R) ∋ coef u v^T (u, v = (row, vacuum)): the deferred tensors tau Sym(h0 h0 h1), tau = mu_U(U)/h1(U),
        at R:  tau [h1(R) h0 h0^T + h0(R) (h0 h1^T + h1 h0^T)];  h0(R) = 0 unless R ⊋ [4,3,2,1] (then the cubic law
        mu_U(R) = mu_U(U) h1(R)/h1(U) of deferred() is not enough)"""
        out = []
        for U in DEFERRED_ALL + [V for V in KERNEL10 if ("mu", V) in self._tau and V not in DEFERRED_ALL]:
            if contains(R, U):
                h0, h1 = self.ghost(U)
                tau = self.mu(U) * inv(self.hval(h1, U)) % p
                a0, a1 = self.hval(h0, R), self.hval(h1, R)
                if a1: out.append((tau * a1 % p, h0, h0))
                if a0: out.append((tau * a0 % p, h0, h1)); out.append((tau * a0 % p, h1, h0))
        return out

    def blocks(self, R, extra=True):
        """[(B, M)] with c(R) ∋ sum_ab M_ab B_a B_b^T, B_a = (row, vacuum): the [4,3,2,1] block at R = [4,3,2,1] or, for
        R in OPEN11, contracted with f(R); and the extra block of R in OPEN11"""
        R = tuple(R); out = []
        for U in OPEN10:
            if not contains(R, U): continue
            if U not in self._open: raise NotImplementedError("10-box diagram %s: call calibrate_open10() first" % "".join(map(str, U)))
            Vb, MU = self._open[U]
            if R == tuple(U):
                out.append(([(e, 0) for e in Vb], MU))
            elif R in OPEN11:
                if R not in self._open11: raise NotImplementedError("11-box diagram %s: call calibrate_open11() first" % "".join(map(str, R)))
                MV, E, N = self._open11[R]
                out.append(([(e, 0) for e in Vb], MV))
                if extra and E and N is not None: out.append((E, N))
            else:
                raise NotImplementedError("R ⊋ %s: available only for R = %s and the 11-box diagrams %s" % (
                    "".join(map(str, U)), "".join(map(str, U)), ", ".join("".join(map(str, V)) for V in OPEN11)))
        return out

    def extra_space(self, V, NB=15):
        """basis [(row, vacuum)] of G_V ⊕ D_V on the labels of V: functions vanishing on every colour S ⊉ V (up to NB boxes) with
        zero components on the tops and co-tops of V; and, for every co-top whose vanishing system (nodes only) is degenerate,
        with minimal diagrams {V, W'}, the functions with zero components on the tops of V that vanish on every S ⊉ V ∪ W' and
        at V ∪ W' itself (deferred directions; for [4,4,2,1] one, over [5,4,2,1])"""
        V = tuple(V); key = ("extra", V, NB)
        if key in self._tau: return self._tau[key]
        fpv = self._fpv(NB); Vp = P(V)
        labsV = sorted(I.channels_of(Vp), key=lambda c: (lev(c), nm(c)))
        tops = [x for x in labsV if I.rmin(x) == (Vp,)]
        cot = [x for x in labsV if len(I.rmin(x)) > 1 and Vp in I.rmin(x)]
        own = set(tops) | set(cot)
        def ispace(W, also=()):
            off = [tuple(nu) for n in range(0, NB + 1) for nu in parts(n) if not contains(nu, W)] + list(also)
            return [[int(x) % p for x in f] for f in _null_np([[int(fpv.psi(X, S)) for X in labsV] + [1] for S in off], len(labsV) + 1)]
        vecs = []
        IV = ispace(V)
        idx = [j for j, X in enumerate(labsV) if X in own]
        if IV:
            K = _null_np([[f[j] for f in IV] for j in idx], len(IV)) if idx else [[int(i == k) for i in range(len(IV))] for k in range(len(IV))]
            for k in K: vecs.append([sum(int(c) * f[j] for c, f in zip(k, IV)) % p for j in range(len(labsV) + 1)])
        for c in cot:
            mins = [tuple(x for x in u if x) for u in I.rmin(c)]
            nds = [tuple(nu) for nu in f2core._NODES_ORIG(c, NB)]
            B = list(I.support(c))
            if not _null_np([[int(fpv.psi(x, S)) for x in B] for S in nds], len(B)): continue      # non-degenerate co-top
            W = V
            for m_ in mins: W = tuple(max(a, b) for a, b in itertools.zip_longest(W, m_, fillvalue=0))
            IW = ispace(W, also=[W])                              # vanishing off up(W) and at W itself
            jt = [j for j, X in enumerate(labsV) if X in set(tops)]
            if IW:                                                # deferred directions have no components on the tops of V
                K = _null_np([[f[j] for f in IW] for j in jt], len(IW)) if jt else [[int(i == k) for i in range(len(IW))] for k in range(len(IW))]
                for k in K: vecs.append([sum(int(c) * f[j] for c, f in zip(k, IW)) % p for j in range(len(labsV) + 1)])
        # independent basis
        basis = []
        for v in vecs:
            if len(_rref_np(basis + [v], len(labsV) + 1)[1]) > len(basis): basis.append(v)
        out = [({X: v[j] for j, X in enumerate(labsV) if v[j]}, v[-1]) for v in basis]
        self._tau[key] = out
        return out

    def calibrate_open11(self, V, legs, extra_legs=None, lmin=None, verbose=False):
        """Block of [4,3,2,1] contracted with f(V), M_V = T_U(.,.,f(V)), for V in OPEN11, from two-colour ring legs at colours S
        with S ∩ V = [4,3,2,1] (only structures inside [4,3,2,1] contribute there): coordinates beta(S) of the residual
        leg - old in V_U, beta(S) = M_V f(S), together with the tensor symmetry M_V f(W) = M_W f(V) for W = [4,3,2,1] and the
        OPEN11 diagrams calibrated before.  Then the extra block of V on G_V ⊕ D_V from ring legs at colours S ⊇ V
        (extra_legs).  legs, extra_legs: {S: {label: value}} on the labels of V of level >= lmin (twocolor_legs.ring_legs)."""
        V = tuple(V); U = tuple(OPEN10[0]); core = self.core; lmin = lmin or OPEN10_LMIN
        if V not in OPEN11: raise ValueError("%s is not an OPEN11 diagram" % (list(V),))
        if U not in self._open: raise RuntimeError("calibrate_open11: calibrate_open10() first")
        Vb, MU = self._open[U]; nv = len(Vb)
        cub, _ = self.cubes(U)
        dfr = []
        for W in DEFERRED_ALL:
            if contains(U, W):
                h0, h1 = self.ghost(W); dfr.append((self.mu(W) * inv(self.hval(h1, W)) % p, h0, h1))
        chV = set(I.channels_of(P(V))); chU = set(I.channels_of(P(U)))
        labs = sorted({X for X in chU | chV if lev(X) >= lmin}, key=lambda c: (lev(c), nm(c)))
        def f(S): return [core.Phi(e, S) for e in Vb]
        def oldU(S):
            out = {X: 0 for X in labs}
            for g, e in cub:
                a_ = core.Phi(e, S)
                if not a_: continue
                b_ = core.Phi(e, V)
                if not b_: continue
                s_ = g * a_ % p * b_ % p
                for X in labs:
                    y = e.get(X, 0)
                    if y: out[X] = (out[X] + y * s_) % p
            for tau, h0, h1 in dfr:
                h0R, h1R, h0S, h1S = self.hval(h0, V), self.hval(h1, V), self.hval(h0, S), self.hval(h1, S)
                a0 = tau * (h1R * h0S + h0R * h1S) % p; a1 = tau * h0R % p * h0S % p
                for X in labs:
                    y = (a0 * h0[0].get(X, 0) + a1 * h1[0].get(X, 0)) % p
                    if y: out[X] = (out[X] + y) % p
            return out
        pairs = [(a, b) for a in range(nv) for b in range(a, nv)]; pidx = {pq: i for i, pq in enumerate(pairs)}
        eqs = []
        def add_eq(fS, beta):                                     # M_V fS = beta
            for a in range(nv):
                r = [0] * (len(pairs) + 1)
                for b in range(nv):
                    if fS[b]: k = pidx[(min(a, b), max(a, b))]; r[k] = (r[k] + fS[b]) % p
                r[-1] = (-beta[a]) % p; eqs.append(r)
        n_iso = 0
        for S, lv in legs.items():
            S = tuple(S)
            if meet(S, V) != U: continue
            ol = oldU(S)
            Mx = [[e.get(X, 0) for e in Vb] + [(-(lv.get(X, 0) - ol[X])) % p] for X in labs]
            sol = [s_ for s_ in _null_np(Mx, nv + 1) if int(s_[-1]) % p]
            if not sol: raise RuntimeError("calibrate_open11: residual at %s not in the block space" % (S,))
            s_ = sol[0]; c0 = inv(int(s_[-1])); add_eq(f(S), [int(y) * c0 % p for y in s_[:-1]]); n_iso += 1
        fV = f(V)
        add_eq(f(U), [sum(MU[a][b] * fV[b] for b in range(nv)) % p for a in range(nv)])
        for W, (MW, _, _) in self._open11.items():
            if W != V: add_eq(f(W), [sum(MW[a][b] * fV[b] for b in range(nv)) % p for a in range(nv)])
        Rm, piv = _rref_np(eqs, len(pairs) + 1); piv = list(piv)
        if len(pairs) in piv: raise RuntimeError("calibrate_open11: equations for M_V inconsistent")
        if len(piv) < len(pairs): raise RuntimeError("calibrate_open11: M_V not determined (%d of %d; add colours)" % (len(piv), len(pairs)))
        x = {pc: (-int(Rm[i][-1])) % p for i, pc in enumerate(piv)}
        MV = [[x[pidx[(min(a, b), max(a, b))]] for b in range(nv)] for a in range(nv)]
        E = self.extra_space(V)
        self._open11[V] = (MV, E, None)
        N = None
        if E:
            co, ev = [], []
            for S, lv in (extra_legs or {}).items():
                S = tuple(S); lab = [X for X in lv]
                mo = self.ring_leg(V, S, labels=lab, extra=False)
                Mx = [[u.get(X, 0) for u, u0 in E] + [(-(lv[X] - mo[X])) % p] for X in lab]
                sol = [s_ for s_ in _null_np(Mx, len(E) + 1) if int(s_[-1]) % p]
                if not sol: raise RuntimeError("calibrate_open11: residual at %s not in G_V ⊕ D_V" % (S,))
                s_ = sol[0]; c0 = inv(int(s_[-1])); co.append([int(y) * c0 % p for y in s_[:-1]])
                ev.append([self.hval(e, S) for e in E])
            ne = len(E); prs = [(a, b) for a in range(ne) for b in range(a, ne)]; pid = {pq: i for i, pq in enumerate(prs)}
            rows_ = []
            for c_, e_ in zip(co, ev):
                for a in range(ne):
                    r = [0] * (len(prs) + 1)
                    for b in range(ne):
                        if e_[b]: k = pid[(min(a, b), max(a, b))]; r[k] = (r[k] + e_[b]) % p
                    r[-1] = (-c_[a]) % p; rows_.append(r)
            if not rows_: raise RuntimeError("calibrate_open11: extra block of %s needs ring legs at colours ⊇ V" % (list(V),))
            Rm, piv = _rref_np(rows_, len(prs) + 1); piv = list(piv)
            if len(prs) in piv or len(piv) < len(prs): raise RuntimeError("calibrate_open11: extra block of %s not determined" % (list(V),))
            xx = {pc: (-int(Rm[i][-1])) % p for i, pc in enumerate(piv)}
            N = [[xx[pid[(min(a, b), max(a, b))]] for b in range(ne)] for a in range(ne)]
        self._open11[V] = (MV, E, N)
        if verbose:
            print("calibrated %s: M_V from %d colours (+ symmetry); extra block dim %d" % ("".join(map(str, V)), n_iso, len(E)))
        return MV, E, N

    # ---------- 10-box kernel diagrams with an inconsistent top system (532, 5311 and transposes)
    def enlarged_support(self, U):
        """B'(t) for the tops t of U: B(t) plus the multi-minimal channels present in U with some, but not all, minimal
        diagrams strictly inside U (the strict 'precedes' of interpolation.py excludes them)"""
        U = P(tuple(U)); chU = I.channels_of(U)
        tops = [x for x in chU if I.rmin(x) == (U,)]
        B = list(I.support(tops[0]))
        extra = [x for x in chU if x not in B and x not in tops and len(I.rmin(x)) > 1
                 and any(contains(U, s) and s != U for s in I.rmin(x))
                 and not all(contains(U, s) and s != U for s in I.rmin(x))]
        return tops, B, extra

    def rows10(self, U):
        """(rows of the tops with an inconsistent vanishing system, on the enlarged support; common kernel k_U of the
        enlarged systems of the other tops).  Checked: the residual of the formula at R = U is gamma k k^T (+ the deferred
        mu' g0 g0^T of U, seen only by double braids |m|,|n| >= 2 and by colours S ⊋ [5,3,2,1])"""
        key = ("rows10", tuple(U))
        if key in self._tau: return self._tau[key]
        core = self.core
        tops, B, extra = self.enlarged_support(U)
        cols = B + extra
        out = {}; hom = []
        for t in tops:
            nds = [tuple(nu) for nu in I.nodes(t)]
            r0 = core.code_row(t)
            if all(core.Phi(r0, nu) == 0 for nu in nds):
                hom += [[core.psi(x, nu) for x in cols] for nu in nds]
                continue
            Ma = [[core.psi(x, nu) for x in cols] + [(-core.psi(t, nu)) % p] for nu in nds]
            ns = [v for v in _null_np(Ma, len(cols) + 1) if v[-1] % p]
            if len(ns) != 1 or len(_null_np([r[:-1] for r in Ma], len(cols))) != 0:
                raise RuntimeError("enlarged system of %s not uniquely solvable" % nm(t))
            v = ns[0]; c0 = (-inv(v[-1])) % p                 # sum_x E_x psi_x(nu) + psi_t(nu) = 0
            row = {t: 1}
            for x, a in zip(cols, v[:-1]):
                if a * c0 % p: row[x] = a * c0 % p
            out[t] = row
        K = _null_np(hom, len(cols)) if hom else []
        if not out:
            raise RuntimeError("U=%s: no inconsistent top system" % (list(U),))
        k = None
        if len(K) == 1:                     # [5,3,2]: the common kernel is the cube direction (a kernel diagram)
            k = {x: a for x, a in zip(cols, K[0]) if a}
        else:                               # [5,3,1,1]: no common kernel -> ghost_rule_direction
            k = self.ghost_rule_direction(U)
        if k is not None:
            X0 = min(k, key=lambda c: (lev(c), nm(c))); s0 = inv(k[X0])
            k = {x: a * s0 % p for x, a in k.items()}
        self._tau[key] = (out, k)
        return out, k

    def ghost_rule_direction(self, U):
        """conjectural rule (fits [5,3,2] and [5,3,1,1]): the cube direction is the element of the 2-dim ghost space G_U that
        vanishes on the multi-minimal channels with U among their minimal diagrams whose own vanishing system is degenerate
        ([5,3,2]: {[3,2,2],[5,1,1]};  [5,3,1,1]: {[4,1,1,1],[4,3]})"""
        core = self.core; Up = P(tuple(U))
        G, _, _ = ghost_space(core, U)
        if len(G) != 2: return None
        deg = []
        for x in I.channels_of(Up):
            if len(I.rmin(x)) > 1 and Up in I.rmin(x):
                core.code_row(x); rk = core.fp._E.get(("rank", x)) if core.fp else None
                if rk and rk[0] < rk[1]: deg.append(x)
        if not deg: return None
        M = [[G[0][0].get(x, 0), G[1][0].get(x, 0)] for x in deg]
        ns = nullspace(M, 2)
        if len(ns) != 1: return None
        a, b = ns[0]
        if (a * G[0][1] + b * G[1][1]) % p: return None          # no vacuum component expected
        v = {x: (a * G[0][0].get(x, 0) + b * G[1][0].get(x, 0)) % p for x in set(G[0][0]) | set(G[1][0])}
        return {x: c for x, c in v.items() if c}

    def twist_direction(self, U):
        """direction of the twist-leg residual at R = U (Formula I minus the formula without the cube of U); it lies in the
        two-dimensional ghost space G_U (checked) -- used for [5,3,1,1], whose cube direction has no closed rule yet"""
        core = self.core
        w, rd = self.bridge(U, skip=(U,))
        wi = self.wI(U)
        r = {X: (wi.get(X, 0) - w.get(X, 0)) % p for X in rd.chans if X[1] != ()}
        r = {X: a for X, a in r.items() if a}
        G, _, _ = ghost_space(core, U)
        labs = sorted(set(r) | set(G[0][0]) | set(G[1][0]), key=lambda c: (lev(c), nm(c)))
        M = [[G[0][0].get(X, 0), G[1][0].get(X, 0), (-r.get(X, 0)) % p] for X in labs] + [[G[0][1], G[1][1], 0]]
        if not any(t[-1] % p for t in nullspace(M, 3)):
            raise RuntimeError("twist residual of %s not in G_U" % (list(U),))
        X0 = min(r, key=lambda c: (lev(c), nm(c))); s0 = inv(r[X0])
        return {X: a * s0 % p for X, a in r.items()}

    def rows_consistent(self, R, rows):
        """channels with several minimal diagrams and a degenerate vanishing system: shift the row along the kernel so that it
        has no entries on labels absent from R (consistency rule; makes [4,3,2], [4,3,1,1] exact without extra terms)"""
        core = self.core; labsR = set(I.channels_of(P(R)))
        for c in labsR:
            if len(I.rmin(c)) < 2: continue
            if all(contains(R, tuple(x for x in u if x)) for u in I.rmin(c)): continue    # split channel: rows split_row()
            r0 = core.code_row(c); rk = core.fp._E.get(("rank", c)) if core.fp else None
            if not rk or rk[0] == rk[1]: continue
            B = list(I.support(c))
            ks = [{x: a for x, a in zip(B, kv) if a} for kv in nullspace([[core.psi(x, tuple(nu)) for x in B] for nu in I.nodes(c)], len(B))]
            absent = [x for x in B if x not in labsR]
            r = dict(rows.get(c, r0))
            M = [[k.get(x, 0) for k in ks] + [r.get(x, 0)] for x in absent]
            sol = nullspace(M, len(ks) + 1); ok = [t for t in sol if t[-1] % p]
            if not ok: raise RuntimeError("no consistent row for %s in R=%s" % (nm(c), list(R)))
            t = ok[0]; c0 = inv(t[-1])
            for tt, k in zip([x * c0 % p for x in t[:-1]], ks):
                for x, a in k.items(): r[x] = (r.get(x, 0) + tt * a) % p
            rows[c] = {x: a for x, a in r.items() if a}
        return rows

    def kernel(self, U):
        if U not in self._kern:
            if tuple(U) in KERNEL10:
                k = self.rows10(U)[1]
                self._kern[U] = k if k is not None else self.twist_direction(U)
            else:
                self._kern[U] = naive_rows_rule(self.core, U, None)[2]
        return self._kern[U]

    # ---------- kernel cube constants
    def gamma(self, U):
        if U in self._gam: return self._gam[U]
        core = self.core; k = self.kernel(U)
        if U in CLOSED and self.use_closed:
            g = CLOSED[U](self.A, self.q) * inv(core.Phi(k, U) * core.phi(k, 1) % p * core.phi(k, -1)) % p
        elif conj(U) in CLOSED and self.use_closed:
            # transposition: Ft_{U^T}(A,q) = Ft_U(A,1/q)
            g = CLOSED[conj(U)](self.A, inv(self.q)) * inv(core.Phi(k, U) * core.phi(k, 1) % p * core.phi(k, -1)) % p
        else:
            g = self.calibrate_cube(U)
        self._gam[U] = g
        return g

    def bridge(self, R, extra=True, skip=()):
        """twist leg c(R) Lambda on the labels of R by the strengthened formula (without the blocks in skip)"""
        core = self.core
        if not extra:
            rd = RepData(core, R, rows_override=self.rows(R))
            return {X: sum(rd.rows[c].get(X, 0) * rd.W[c] % p * rd.ph1[c] for c in rd.chans) % p for X in rd.chans}, rd
        terms, rd = self.terms(R, skip=skip)
        w = {X: 0 for X in rd.chans}
        for W, e in terms:
            s_ = W * core.phi(e, 1) % p
            if not s_: continue
            for X, a in e.items():
                if X in w: w[X] = (w[X] + a * s_) % p
        if not any(tuple(U) in skip for U in OPEN10):
            for B, M in self.blocks(R):
                f1 = [self.hpow(b, 1) for b in B]
                for a, (u, u0) in enumerate(B):
                    s_ = sum(M[a][b] * f1[b] for b in range(len(B))) % p
                    if not s_: continue
                    for X, y in u.items():
                        if X in w: w[X] = (w[X] + y * s_) % p
        for coef, u, v in self.deferred_terms(R):             # h0(Lambda^{±1}) = 0: only R ⊋ [4,3,2,1] contributes
            s_ = coef * self.hpow(v, 1) % p
            if not s_: continue
            for X, y in u[0].items():
                if X in w: w[X] = (w[X] + y * s_) % p
        if extra:
            for U in (GHOST2 if not self.consistency else GHOST2_10):
                if contains(R, U) and U not in skip:
                    for X, v in self.block_twist(U, R).items():
                        if X in w: w[X] = (w[X] + v) % p
        return w, rd

    def calibrate_cube(self, U):
        core = self.core; k = self.kernel(U)
        w, rd = self.bridge(U, skip=(U,))
        wi = self.wI(U); f1 = core.phi(k, 1); kU = core.Phi(k, U)
        lams = set()
        for X, a in k.items():
            if X in w and a:
                lams.add((wi.get(X, 0) - w[X]) * inv(a * f1 % p) % p)
        assert len(lams) == 1, ("inconsistent cube calibration", U, len(lams))
        return lams.pop() * inv(kU) % p

    # ---------- 2-dim ghost blocks
    def ghost(self, U):
        if U not in self._ghost:
            G, tops, labs = ghost_space(self.core, U)
            a0, a1 = [fval(self.core, g, "S" + "".join(map(str, U))) if max(U) < 10 else None for g in G]
            h0 = ({x: (a1 * G[0][0].get(x, 0) - a0 * G[1][0].get(x, 0)) % p for x in set(G[0][0]) | set(G[1][0])},
                  (a1 * G[0][1] - a0 * G[1][1]) % p)
            h0 = ({x: v for x, v in h0[0].items() if v}, h0[1])
            h1 = G[1] if a1 else G[0]
            self._ghost[U] = (h0, h1)
        return self._ghost[U]

    def hval(self, f, S):
        u, u0 = f; return (self.core.Phi(u, S) + u0) % p

    def hpow(self, f, m):
        u, u0 = f; return (self.core.phi(u, m) + u0) % p

    def tau_closed(self, U):
        """(tau_011, tau_111) from the closed figure-eight contribution and the zero rule of the twist direction"""
        transp = U not in CLOSED_BLOCK
        Ft, X0 = CLOSED_BLOCK[conj(U) if transp else U]
        if transp: X0 = I.pair(P(conj(X0[1])), P(conj(X0[2])))
        X0 = I.pair(P(X0[1]), P(X0[2])) if X0[0] == "p" else I.single(P(X0[1]))
        h0, h1 = self.ghost(U)
        a, b = h0[0].get(X0, 0), h1[0].get(X0, 0)
        al, be = b, (-a) % p                                   # v = al*h0 + be*h1 vanishes at X0
        vm1 = (al * self.hpow(h0, -1) + be * self.hpow(h1, -1)) % p
        F = Ft(self.A, inv(self.q)) if transp else Ft(self.A, self.q)
        sc = F * inv(vm1) % p
        den = inv(self.hval(h1, U) * self.hpow(h1, 1) % p)
        return (sc * al % p * den % p, sc * be % p * den % p)

    def tau(self, U):
        """(tau_011, tau_111) from the twist leg at R = U:  residual = h0 * tau_011 h1(U) h1(Lam) + h1 * tau_111 h1(U) h1(Lam)"""
        if U in self._tau: return self._tau[U]
        if self.use_closed and (U in CLOSED_BLOCK or conj(U) in CLOSED_BLOCK):
            self._tau[U] = self.tau_closed(U); return self._tau[U]
        h0, h1 = self.ghost(U)
        w, rd = self.bridge(U, skip=(U,))
        wi = self.wI(U)
        L = [X for X in w]
        M = [[h0[0].get(X, 0), h1[0].get(X, 0), (-(wi.get(X, 0) - w[X])) % p] for X in L]
        sol = nullspace(M, 3); ok = [s for s in sol if s[-1] % p]
        assert ok and len(sol) == 1, ("ghost block calibration failed", U, len(sol))
        s = ok[0]; c0 = inv(s[-1]); a, b = s[0] * c0 % p, s[1] * c0 % p
        den = inv(self.hval(h1, U) * self.hpow(h1, 1) % p)
        self._tau[U] = (a * den % p, b * den % p)
        return self._tau[U]

    def block_twist(self, U, R):
        """twist-leg contribution of the ghost block of U at R (requires h0(R) = 0)"""
        h0, h1 = self.ghost(U)
        if self.hval(h0, R): raise NotImplementedError("h0(R) != 0: tau_000/tau_001 needed (R ⊋ deferred diagram)")
        t011, t111 = self.tau(U)
        s = self.hval(h1, R) * self.hpow(h1, 1) % p
        out = {}
        for X in set(h0[0]) | set(h1[0]):
            out[X] = (h0[0].get(X, 0) * t011 + h1[0].get(X, 0) * t111) % p * s % p
        return out

    # ---------- knots
    def twist_knots(self, R, ms=range(-4, 5)):
        """H_R(Tw_m) = 1 + sum_X w_X (Lambda_X^m - 1)"""
        w, rd = self.bridge(R)
        core = self.core
        return {m: (1 + sum(v * (pow(core.Lam(X), m % (p - 1), p) - 1) for X, v in w.items())) % p for m in ms}

    # ---------- deferred term mu_U h0 h0^T (U = [4,3,2], [4,3,1,1] and transposes)
    def mu(self, U):
        """mu_U(U) from the closed double-braid contribution Delta_U = mu h0(Lam^-2) h0(Lam^2) (transposes: q -> 1/q)"""
        U = tuple(U)
        key = ("mu", U)
        if key not in self._tau:
            h0, h1 = self.ghost(U)
            if U in DEFERRED: Dl = DEFERRED[U](self.A, self.q)
            elif conj(U) in DEFERRED: Dl = DEFERRED[conj(U)](self.A, inv(self.q))
            else: raise NotImplementedError("no closed deferred term for %s" % (U,))
            self._tau[key] = Dl * inv(self.hpow(h0, -2) * self.hpow(h0, 2) % p) % p
        return self._tau[key]

    def set_mu(self, U, value):
        """deferred constant mu_U(U) supplied from outside (KERNEL10 diagrams: no closed form; e.g. from one 12-box ring leg
        S ⊋ [5,3,2,1] of the two-colour formula, see verify532.py)"""
        self._tau[("mu", tuple(U))] = value % p

    def deferred(self, R):
        """[(mu_U(R), h0_U)] for the deferred diagrams U ⊆ R;  mu_U(R) = mu_U(U) h1(R)/h1(U) (cubic law)"""
        out = []
        for U in DEFERRED_ALL + [V for V in KERNEL10 if ("mu", V) in self._tau and V not in DEFERRED_ALL]:
            if contains(R, U):
                h0, h1 = self.ghost(U)
                if self.hval(h0, R): raise NotImplementedError("h0_%s(R) != 0 (R ⊋ [4,3,2,1])" % "".join(map(str, U)))
                out.append((self.mu(U) * self.hval(h1, R) % p * inv(self.hval(h1, U)) % p, h0))
        return out

    def _check_blocks(self, R):
        self.blocks(R)                                            # raises if a needed block is missing or not available
        for U in KERNEL10:
            if contains(R, U) and ("mu", U) not in self._tau and U not in DEFERRED_ALL:
                raise NotImplementedError("deferred constant of %s not closed: supply it with set_mu()" % "".join(map(str, U)))

    def double_braid(self, R, mn):
        """H_R(m,n) = 1 + sum_c W_c phi_c(m) phi_c(n) + cubes + deferred terms, for the pairs (m,n) in mn"""
        R = tuple(R); self._check_blocks(R); core = self.core
        ms = sorted({m for m, n in mn} | {n for m, n in mn})
        tl, rd = self.terms(R)
        terms = [(W, {m: core.phi(e, m) for m in ms}) for W, e in tl]
        H = {(m, n): (1 + sum(c * f[m] % p * f[n] for c, f in terms)) % p for m, n in mn}
        for coef, u, v in self.deferred_terms(R):
            fu = {m: self.hpow(u, m) for m in ms}; fw = {m: self.hpow(v, m) for m in ms}
            for (m, n) in mn: H[(m, n)] = (H[(m, n)] + coef * fu[m] % p * fw[n]) % p
        for B, M in self.blocks(R):
            ph = [{m: self.hpow(b, m) for m in ms} for b in B]
            for (m, n) in mn:
                H[(m, n)] = (H[(m, n)] + sum(M[a][b] * ph[a][m] % p * ph[b][n] for a in range(len(B)) for b in range(len(B)))) % p
        return H

    def ring_leg(self, R, S, labels=None, extra=True):
        """c(R) chi(S) on the labels of R (ring leg; compare with the two-colour formula); extra=False omits the extra block
        of an OPEN11 diagram (used while calibrating it)"""
        R = tuple(R); S = tuple(S); core = self.core
        blocks = self.blocks(R, extra=extra)
        for U in KERNEL10:
            if contains(R, U) and ("mu", U) not in self._tau and U not in DEFERRED_ALL:
                raise NotImplementedError("deferred constant of %s not closed: supply it with set_mu()" % "".join(map(str, U)))
        tl, rd = self.terms(R)
        labels = labels or [X for X in rd.chans if X[1] != ()]
        leg = {X: 0 for X in labels}
        for W, e in tl:
            s_ = W * core.Phi(e, S) % p
            if not s_: continue
            for X in labels:
                a = e.get(X, 0)
                if a: leg[X] = (leg[X] + a * s_) % p
        for B, M in blocks:
            fS = [self.hval(b, S) for b in B]
            for a, (u, u0) in enumerate(B):
                s_ = sum(M[a][b] * fS[b] for b in range(len(B))) % p
                if not s_: continue
                for X in labels: leg[X] = (leg[X] + u.get(X, 0) * s_) % p
        for coef, u, v in self.deferred_terms(R):
            hs = coef * self.hval(v, S) % p
            if not hs: continue
            for X in labels: leg[X] = (leg[X] + hs * u[0].get(X, 0)) % p
        return leg


