"""Data check of the Berezinian reduction rule on the lines A = q^-m, for every k >= 1:

  (b_k)  if R_{k+1} = m + k  then  H_R(q^-m, q) = H_{R'}(q^-m, q),  R' = (R_1+1, ..., R_k+1, R_{k+2}, ...)

(V_R (x) Ber = V_{R'} for U_q gl(k|m+k); k = 1 is rule (b) of defect_tower.tex).  Every knot in data/homfly,
all pairs with both R and R' present.  Identity of Laurent polynomials in q is tested modulo the prime
p = 2^61 - 1 at two random points q (a false identity passes with probability < 1e-15 per pair).

usage: python3 scripts/nonrect_rules_check.py > data/nonrect_rule_bk.tsv
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nonrect_supergroup as ns  # noqa: E402
from nonrect_h1_scan import parts  # noqa: E402

P = 2 ** 61 - 1
QS = (1234567891011, 987654321987654)


TERM = re.compile(r"([+-]?)\s*(\d+)?\*?(A(?:\^\(?(-?\d+)\)?)?)?\*?(q(?:\^\(?(-?\d+)\)?)?)?")


def fast_terms(s):
    """[(i, j, c)] of a Laurent polynomial string like '-A^4 + 2*A^2*q^(-2) + 1' (fast regex parser)"""
    out = []
    for tok in s.replace(" - ", " + -").split(" + "):
        tok = tok.strip()
        mt = TERM.fullmatch(tok)
        if not mt:
            raise ValueError(tok)
        sg, c, a, ai, qq, qi = mt.groups()
        c = int(c) if c else 1
        c = -c if sg == "-" else c
        i = (int(ai) if ai else 1) if a else 0
        j = (int(qi) if qi else 1) if qq else 0
        out.append((i, j, c))
    return out


def load(knot):
    d, cur = {}, None
    for line in open(os.path.join(ns.DATA, knot + ".txt")):
        if line.startswith("R = "):
            cur = tuple(int(x) for x in line[4:].strip(" []\n").split(","))
        elif line.startswith("H = ") and cur is not None:
            d[cur] = fast_terms(line[4:].strip())
    return d


def val(t, m, q):
    return sum(c * pow(q, (-m * i + j) % (P - 1), P) for i, j, c in t) % P


def main():
    knots = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(ns.DATA, "*.txt")))
    rules = []
    for n in range(1, 7):
        for R in parts(n):
            for k in range(1, len(R)):
                m = R[k] - k
                if m < 0:
                    continue
                Rp = tuple(x + 1 for x in R[:k]) + R[k + 1:]
                rules.append((m, k, R, Rp))
    res = {r: [0, []] for r in rules}
    for kn in knots:
        d = load(kn)
        for rule in rules:
            m, k, R, Rp = rule
            if R not in d or Rp not in d:
                continue
            res[rule][0] += 1
            if not all(val(d[R], m, q) == val(d[Rp], m, q) for q in QS):
                res[rule][1].append(kn)
    print("m\tk\tR\tR'\tknots_tested\tfailures")
    for rule in rules:
        m, k, R, Rp = rule
        print("%d\t%d\t%s\t%s\t%d\t%s" % (m, k, list(R), list(Rp), res[rule][0], ",".join(res[rule][1]) or "-"))


if __name__ == "__main__":
    main()
