"""Universal reductions of colored HOMFLY on the lines A = q^{-n} (D_n = A q^n - A^-1 q^-n = 0).

On the line A = q^{-n}, n >= 0 (negative n: transpose R and use -n):

  (a) n >= R_1  (transposed sl_n):   H_R = H_{R*},  R* = (complement of R^T in the n x l(R) box)^T;
  (b) 0 <= n < R_1 and R_2 = n + 1:  H_R = H_{(R_1 + 1, R_3, R_4, ...)}
      (n = 0 is the hook property H_R(A=1) = Delta(q^{2|R|}); n >= 1 is the analogue
      for D_n: the second row slides into the first, minus n boxes);
  (c) otherwise no universal reduction: there the knot-dependent (defect) structure lives.

NOTE: (a), (b) are not complete, and normal_form below follows the reductions in one direction
only.  The complete set of coincidences comes from the Berezinian twists of gl(k|k+m) for all k
(rule (b) is k = 1; k = 2 gives e.g. [2,2,2] = [3,3] at A = 1): see scripts/super_reductions.py
and docs/notes/nonrect_lines.tex.

usage: python3 scripts/line_reductions.py KNOT [...]   (prints mismatches, numeric check mod p)
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
P = 2147483647


def conj(R):
    return tuple(sum(1 for x in R if x > i) for i in range(R[0])) if R else ()


def _red_pos(R, n):
    if not R:
        return None
    if n >= R[0] and n >= 1:
        T, c = conj(R), len(R)
        if len(T) > n:
            return None
        Tc = sorted((c - (T[n - 1 - i] if n - 1 - i < len(T) else 0) for i in range(n)), reverse=True)
        Tc = tuple(x for x in Tc if x > 0)
        Rc = conj(Tc) if Tc else ()
        return Rc if Rc != R else None
    if len(R) >= 2 and R[1] == n + 1:
        return tuple(sorted((R[0] + 1,) + R[2:], reverse=True))
    return None


def reduce_once(R, n):
    """one reduction step of H_R on the line A = q^{-n} (None: no rule applies)"""
    if n >= 0:
        return _red_pos(tuple(R), n)
    r = _red_pos(conj(tuple(R)), -n)
    return None if r is None else (conj(r) if r else ())


def normal_form(R, n):
    orbit = [tuple(R)]
    while True:
        r = reduce_once(orbit[-1], n)
        if r is None or r in orbit:
            break
        orbit.append(r)
    return min(orbit, key=lambda t: (sum(t), t))


def _check(knot, lines=range(-8, 9)):
    from homfly.algebra.laurent import parse_laurent
    H, R = {(): {(0, 0): 1}}, None
    for line in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "homfly", knot + ".txt")):
        if line.startswith("R = "):
            R = tuple(int(x) for x in line[4:].strip(" []\n").split(","))
        elif line.startswith("H = ") and R is not None:
            H[R] = dict(parse_laurent(line[4:].strip()).terms)
    rng = random.Random(1)
    qs = [rng.randrange(2, P - 1) for _ in range(2)]

    def ev(t, n, q):
        return sum(c * pow(q, (j - n * i) % (P - 1), P) for (i, j), c in t.items()) % P
    bad = 0
    for n in lines:
        vals = {R: tuple(ev(t, n, q) for q in qs) for R, t in H.items()}
        for R in H:
            for Rp in H:
                if sum(Rp) < sum(R) and (vals[R] == vals[Rp]) != (normal_form(R, n) == normal_form(Rp, n)):
                    bad += 1
                    print("%s n=%d %s vs %s: observed %s" % (knot, n, R, Rp, vals[R] == vals[Rp]))
    return bad


if __name__ == "__main__":
    tot = sum(_check(k) for k in sys.argv[1:])
    print("mismatches:", tot)
