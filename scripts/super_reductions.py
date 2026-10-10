"""All universal coincidences of colored HOMFLY on the lines A = q^{-m} (D_m = 0) from U_q gl(k|k+m).

On A = q^{-m} the reduced H_R is the (1,1)-tangle invariant of U_q gl(k|k+m) on the covariant module V_R, for every
k >= 0 with m + k >= 0 and R in the (k|k+m)-hook (R_{k+1} <= k+m); its highest weight is
(R_1..R_k | <R^T_1 - k>, .., <R^T_{k+m} - k>).  Tensoring with the Berezinian (1..1 | -1..-1) only changes the
framing, so

  (b_k)  R_{k+1} = m + k  ==>  H_R = H_{R'},   R' = (R_1+1, .., R_k+1, R_{k+2}, R_{k+3}, ..)       (k >= 1)

k = 1 is the rule (b) of scripts/line_reductions.py; k = 2 gives e.g. [2,2,2] = [3,3] and [3,2,2] = [4,3] at A = 1.
Together with the same rules for R^T on the line -m and sl duality (rule (a)) these generate every coincidence
H_R = H_R' observed on the lines.  Checked: random knots of data/homfly (|R| <= 6, -8 <= m <= 8) and the double
braids of homfly.formula2 (|R| <= 10, m = -2..3): no predicted identity fails and no other identity holds.

usage: python3 scripts/super_reductions.py [NKNOTS] [SEED]      (data check on NKNOTS random knots, default 40)
"""
import glob
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
P = 2147483647


def conj(R):
    R = tuple(x for x in R if x > 0)
    return tuple(sum(1 for x in R if x > i) for i in range(R[0])) if R else ()


def ber_moves(R, m):
    """one Berezinian twist of gl(k|k+m), k >= 1, in either direction"""
    out = set()
    L = list(R) + [0] * 24
    for k in range(1, 16):
        if m + k < 0:
            continue
        if L[k] == m + k:
            out.add(tuple(x for x in [x + 1 for x in L[:k]] + L[k + 1:] if x > 0))
        if L[k - 1] - 1 >= m + k >= L[k] and all(x >= 1 for x in L[:k]):
            out.add(tuple(x for x in [x - 1 for x in L[:k]] + [m + k] + L[k:] if x > 0))
    return out


def moves(R, m):
    R = tuple(R)
    s = set(ber_moves(R, m)) | {conj(x) for x in ber_moves(conj(R), -m)}
    for N, S, back in ((-m, R, lambda t: t), (m, conj(R), conj)):        # sl_N duality (rule (a))
        if N >= 1 and len(S) <= N:
            L = list(S) + [0] * N
            s.add(back(tuple(x for x in (L[0] - L[N - 1 - i] for i in range(N)) if x > 0)))
    s.discard(R)
    return s


def orbit(R, m, maxsize):
    """the class of R on the line A = q^{-m}, restricted to |R'| <= maxsize"""
    seen, st = {tuple(R)}, [tuple(R)]
    while st:
        for y in moves(st.pop(), m):
            if sum(y) <= maxsize and y not in seen:
                seen.add(y)
                st.append(y)
    return frozenset(seen)


def _load(path):
    from homfly.algebra.laurent import parse_laurent
    H, R = {(): {(0, 0): 1}}, None
    for line in open(path):
        if line.startswith("R = "):
            R = tuple(int(x) for x in line[4:].strip(" []\n").split(","))
        elif line.startswith("H = ") and R is not None:
            H[R] = dict(parse_laurent(line[4:].strip()).terms)
    return H


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    files = sorted(glob.glob(os.path.join(HERE, "..", "data", "homfly", "*.txt")))
    files = [f for f in files if not os.path.basename(f).startswith("L")]
    rng.shuffle(files)
    Hs = [_load(f) for f in files[:n]]
    allR = sorted(set().union(*Hs), key=lambda t: (sum(t), t))
    q = rng.randrange(2, P - 1)
    pred = bad = 0
    for m in range(-8, 9):
        a = pow(q, (-m) % (P - 1), P)
        vals = {}
        for i, h in enumerate(Hs):
            for R, t in h.items():
                vals.setdefault(R, {})[i] = sum(c * pow(a, x % (P - 1), P) * pow(q, y % (P - 1), P)
                                                for (x, y), c in t.items()) % P
        for j, R in enumerate(allR):
            orb = orbit(R, m, 30)
            for Rp in allR[:j]:
                common = [i for i in vals[R] if i in vals[Rp]]
                if len(common) < 5:
                    continue
                obs = all(vals[R][i] == vals[Rp][i] for i in common)
                pr = Rp in orb
                pred += pr
                if obs != pr:
                    bad += 1
                    print("m=%d %s %s observed %s predicted %s" % (m, R, Rp, obs, pr))
    print("%d knots: %d predicted identities, %d mismatches" % (n, pred, bad))


if __name__ == "__main__":
    main()
