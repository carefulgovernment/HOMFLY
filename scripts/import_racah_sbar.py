"""Import exclusive S̄ / T̄ data of families H (hecke_Ygauge) and G (gtpath)
from the unpacked release into data/racah/portable/sbar_<R>.json.gz.

    python scripts/import_racah_sbar.py --archive <racah_matrices_upto6> --out data/racah/portable

Stored: S̄ (rational Y-gauge) entries as [num, den] term lists [[i, j, c]] = c A^i q^j,
framing-free T̄² (diagonal, sign-free; t0 = 1), t0² for reference, labels, vacuum index 0.
No sympy needed: the release gives integer coefficient encodings.
"""
import argparse
import gzip
import json
import os
import re

H_REPS = {"6": "R6", "51": "R51", "411": "R411", "33": "R33", "222": "R222",
          "3111": "R3111", "21111": "R21111", "111111": "R111111"}

_MON = re.compile(r"(q|A)\*\*\(?(-?\d+)\)?")


def monomial(s):
    """exponents (A, q) and sign of a product like '-1*t0*q**(-6)*A**(-3)'."""
    ea = eq = 0
    for v, e in _MON.findall(s):
        if v == "A":
            ea += int(e)
        else:
            eq += int(e)
    sign = -1 if s.strip().startswith("-") else 1
    return sign, ea, eq


def poly_from_monomials(s):
    """'(q**30*A**6)/(1)' -> term list (single monomial numerator, den 1)."""
    num, den = s.split("/")
    _, ea, eq = monomial(num)
    if den.strip("() ") != "1":
        _, da, dq = monomial(den)
        ea, eq = ea - da, eq - dq
    return [[ea, eq, 1]]


def h_entry(e):
    if e.get("zero"):
        return [[], [[0, 0, 1]]]
    # entry = q^eq A^eA num(q^2, A^2)/den(q^2, A^2); triples [a, b, c] = c q^(2a) A^(2b)
    # coefficients may be rationals [n, d]: clear denominators into num/den
    from fractions import Fraction
    from math import lcm
    fr = lambda c: Fraction(c[0], c[1]) if isinstance(c, list) else Fraction(c)  # noqa: E731
    L = lcm(*[fr(c).denominator for _, _, c in e["num"] + e["den"]])
    Ln = lcm(*[fr(c).denominator for _, _, c in e["num"]])
    Ld = lcm(*[fr(c).denominator for _, _, c in e["den"]])
    # num/den = (Ln*num)/(Ld*den) * Ld/Ln
    num = [[2 * b + e["eA"], 2 * a + e["eq"], int(fr(c) * Ln) * Ld] for a, b, c in e["num"]]
    den = [[2 * b, 2 * a, int(fr(c) * Ld) * Ln] for a, b, c in e["den"]]
    del L
    return [num, den]


def import_h(root, key, out):
    d = os.path.join(root, "level%d" % sum(int(c) for c in key), H_REPS[key], "hecke_Ygauge")
    S = json.load(open(os.path.join(d, "racah_%s_Sbar_Ygauge.json" % key)))
    T = json.load(open(os.path.join(d, "racah_%s_Tbar.json" % key)))
    assert [t["label"] for t in T["Tbar"]] == S["labels"]
    t0sq = poly_from_monomials(T["t0_squared"])
    (ta, tq, _), = t0sq
    tbar2 = []
    for t in T["Tbar"]:
        _, ea, eq = monomial(t["Tbar"].replace("t0", "1"))
        tbar2.append([2 * ea, 2 * eq])        # framing-free T̄² = A^{2ea} q^{2eq} (t0 = 1)
    res = {"R": [int(c) for c in key], "family": "H", "gauge": S["gauge"],
           "labels": S["labels"], "vacuum_index": 0, "t0_squared": [ta, tq],
           "Sbar": [[h_entry(e) for e in row] for row in S["Sbar_coeffs"]],
           "Tbar2": tbar2}
    path = os.path.join(out, "sbar_%s.json.gz" % key)
    with gzip.open(path, "wt") as f:
        json.dump(res, f)
    return path, len(res["labels"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", default="data/racah/portable")
    ap.add_argument("--reps", nargs="*", default=list(H_REPS))
    a = ap.parse_args()
    for key in a.reps:
        path, n = import_h(a.archive, key, a.out)
        print("H", key, n, path)


if __name__ == "__main__":
    main()
