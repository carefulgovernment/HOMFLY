"""Export the matrices needed by the Montesinos (tangle-sum) method.

Family P (racah_homfly v0.7), multiplicity-free R (rectangular):
    S̄ (antiparallel H <-> V), S (antiparallel <- u-channel parallel), V = S^-1,
    T (parallel eigenvalues), T̄ (antiparallel eigenvalues)  -- all diagonal
    eigenvalue matrices checked diagonal.

    python scripts/import_racah_montesinos.py --archive <racah_matrices_upto6> \
        --reps 1 2 11 3 111 4 22 1111 5 11111 6 111111 --out data/racah/montesinos

Output: <out>/P_<R>.json.gz  with entries [num, den] term lists [[i, j, c]].
"""
import argparse
import gzip
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from import_racah_portable import _entry_converter, rep_key  # noqa: E402


def export_P(archive, R, out):
    sys.path.insert(0, os.path.join(archive, "software", "RACAH_HOMFLY_PORTABLE"))
    import sympy as sp
    from racah_homfly import A, q
    from racah_homfly.exclusive import load_exclusive
    entry = _entry_converter(sp, A, q)
    t = time.time()
    m = load_exclusive(list(R), A, q, full=True)
    res = {"R": list(R), "family": "P", "vacuum_index": int(m["vacuum_index"])}
    for k in ("Sbar", "S", "V"):
        M = m[k]
        res[k] = [[entry(M[i, j]) for j in range(M.shape[1])] for i in range(M.shape[0])]
    for k in ("T", "Tbar"):
        M = m[k]
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if i != j and M[i, j] != 0:
                    raise ValueError("%s not diagonal for %s (multiplicities?)" % (k, R))
        res[k] = [entry(M[i, i]) for i in range(M.shape[0])]
    path = os.path.join(out, "P_%s.json.gz" % rep_key(R))
    with gzip.open(path, "wt") as f:
        json.dump(res, f)
    return R, len(res["Sbar"]), time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--reps", nargs="+", required=True)
    ap.add_argument("--out", default="data/racah/montesinos")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    reps = [tuple(int(c) for c in s) for s in a.reps]
    with ProcessPoolExecutor(a.jobs) as ex:
        for R, n, sec in ex.map(export_P, [a.archive] * len(reps), reps, [a.out] * len(reps)):
            print("P R=%s size %d %.1fs" % (list(R), n, sec), flush=True)


G_FILES = {"42": ("R42", "racah_42_Sbar_rational.json", "racah_42_S_mixed.json"),
           "2211": ("R2211", "racah_2211_Sbar_native_coeffs.json", "racah_2211_S_mixed.json"),
           "321": ("R321", "racah_321_Sbar_coeffs.json", "racah_321_S_mixed.json")}


def _load_json(path):
    import lzma
    if os.path.exists(path):
        return json.load(open(path))
    return json.load(lzma.open(path + ".xz", "rt"))


def _g_entry(e, cache):
    """num / prod f^e with [[a, b, c]] = c A^a q^b; also sympy strings."""
    if e is None:
        return [[], [[0, 0, 1]]]
    key = json.dumps(e["den"])
    if key not in cache:
        acc = [[0, 0, 1]]
        for f, k in e["den"]:
            for _ in range(k):
                acc = _pmul(acc, f)
        cache[key] = acc
    return [e["num"], cache[key]]


def _g_matrix(d, key_coeffs, key_strings):
    if key_coeffs in d:
        cache = {}
        return [[_g_entry(e, cache) for e in row] for row in d[key_coeffs]]
    import sympy as sp
    A, q = sp.symbols("A q")
    entry = _entry_converter(sp, A, q)
    return [[entry(sp.sympify(x, locals={"A": A, "q": q})) for x in row] for row in d[key_strings]]


def export_G(archive, key, out):
    lvl, fsb, fs = G_FILES[key]
    d = os.path.join(archive, "level6", lvl, "gtpath")
    Sb = _load_json(os.path.join(d, fsb))
    S = _load_json(os.path.join(d, fs))
    ev = lambda x: eval(x) if isinstance(x, str) else x  # noqa: E731
    sb_labels = ev(Sb["labels"])
    col_labels = ev(S["col_labels"])
    assert [json.dumps(x) for x in sb_labels] == [json.dumps(x) for x in col_labels], "label mismatch"
    anti = [{"Z": l[0][0], "Zp": l[0][1], "a": str(l[1]), "b": str(l[2]), "eps": 1} for l in sb_labels]
    par = [{"Q": l[0][0], "a": str(l[1]), "b": str(l[2])} for l in ev(S["row_labels"])]
    T = [[sg, 0, k] for sg, k in ev(S["T"])]
    res = {"R": [int(c) for c in key], "family": "G", "anti": anti, "par": par,
           # S̄ is not stored: the engine rebuilds it in the gauge of S's columns
           # from T̄^-1 S̄ T̄^-1 = S^-1 T S (the published vacuum-dual S̄ is not
           # the transform between the two antiparallel channel bases)
           "S": _g_matrix(S, "S_coeffs", "S_strings"), "T": T}
    path = os.path.join(out, "G_%s.json.gz" % key)
    with gzip.open(path, "wt") as f:
        json.dump(res, f)
    return path, len(anti)


def main_G(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--family-G", action="store_true")
    ap.add_argument("--archive", required=True)
    ap.add_argument("--reps", nargs="+", default=list(G_FILES))
    ap.add_argument("--out", default="data/racah/montesinos")
    a = ap.parse_args(argv)
    for key in a.reps:
        path, n = export_G(a.archive, key, a.out)
        print("G R=%s size %d -> %s" % (key, n, path), flush=True)


def main_H(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--family-H", action="store_true")
    ap.add_argument("--archive", required=True)
    ap.add_argument("--reps", nargs="+", default=list(H_DIRS))
    ap.add_argument("--out", default="data/racah/montesinos")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    for key in a.reps:
        path, n = export_H(a.archive, key, a.out)
        print("H R=%s size %d -> %s" % (key, n, path), flush=True)





# ---------------------------------------------------------------------------
# family H (hecke_Ygauge): S̄ and mixed S with explicit block labels
# ---------------------------------------------------------------------------
H_DIRS = {"21": "level3/R21", "31": "level4/R31", "51": "level6/R51", "411": "level6/R411",
          "3111": "level6/R3111", "21111": "level6/R21111", "33": "level6/R33", "222": "level6/R222",
          "6": "level6/R6", "111111": "level6/R111111"}


def _pmul(a, b):
    out = {}
    for i1, j1, c1 in a:
        for i2, j2, c2 in b:
            k = (i1 + i2, j1 + j2)
            out[k] = out.get(k, 0) + c1 * c2
    return [[i, j, c] for (i, j), c in out.items() if c]


def _padd(a, b):
    out = {}
    for i, j, c in a + b:
        out[(i, j)] = out.get((i, j), 0) + c
    return [[i, j, c] for (i, j), c in out.items() if c]


def _h_simple(e):
    """q^eq A^eA num(q^2,A^2)/den(q^2,A^2), triples [a,b,c] = c q^2a A^2b,
    coefficients possibly rational [n, d] -> [num, den] in (A, q) exponents."""
    from fractions import Fraction
    from math import lcm
    fr = lambda c: Fraction(c[0], c[1]) if isinstance(c, list) else Fraction(c)  # noqa: E731
    Ln = lcm(*[fr(c).denominator for _, _, c in e["num"]]) if e["num"] else 1
    Ld = lcm(*[fr(c).denominator for _, _, c in e["den"]]) if e["den"] else 1
    num = [[2 * b + e["eA"], 2 * a + e["eq"], int(fr(c) * Ln) * Ld] for a, b, c in e["num"]]
    den = [[2 * b, 2 * a, int(fr(c) * Ld) * Ln] for a, b, c in e["den"]]
    return num, den


def _h_entry(e):
    if e is None or e.get("zero"):
        return [[], [[0, 0, 1]]]
    if e.get("split"):
        ne, de = _h_simple(e["even"])
        no, do = _h_simple(e["odd"])
        no = [[i, j + 1, c] for i, j, c in no]            # q * odd
        return [_padd(_pmul(ne, do), _pmul(no, de)), _pmul(de, do)]
    return list(_h_simple(e))


def _mono(s):
    import re
    sign = -1 if s.strip().startswith("-") else 1
    ea = sum(int(x) for x in re.findall(r"A\*\*\(?(-?\d+)\)?", s))
    eq = sum(int(x) for x in re.findall(r"q\*\*\(?(-?\d+)\)?", s))
    if re.search(r"(?<![\w*])q(?!\*)", s.replace("q**", "")):
        eq += 1
    return sign, ea, eq


def export_H(archive, key, out):
    import csv
    d = os.path.join(archive, H_DIRS[key], "hecke_Ygauge")
    Sb = json.load(open(os.path.join(d, "racah_%s_Sbar_Ygauge.json" % key)))
    S = json.load(open(os.path.join(d, "racah_%s_S_inclusive_Ygauge.json" % key)))
    T = json.load(open(os.path.join(d, "racah_%s_T.json" % key)))
    lab = list(csv.DictReader(open(os.path.join(d, "labels.csv"))))
    anti = []
    for r in lab:
        X = eval(r['X=[Z,Z\']'])
        anti.append({"Z": X[0], "Zp": X[1], "a": r["a: Y of vertex RxRbar->X"],
                     "b": r["b: Y of vertex XxR->R"], "eps": int(r["eps (sign of Tbar)"])})
    labels_S = Sb["labels"]
    assert len(labels_S) == len(anti)
    rows = S["rows"] if isinstance(S["rows"], list) else eval(S["rows"])
    par = []
    for r in rows:
        Qs, a, b = r.split("|")
        par.append({"Q": eval(Qs), "a": a[2:], "b": b[2:]})
    tmap = {t["row"]: _mono(t["T"]) for t in T["T"]}
    Tpar = [list(tmap[r]) for r in rows]
    res = {"R": [int(c) for c in key], "family": "H", "anti": anti, "par": par,
           "Sbar": [[_h_entry(e) for e in row] for row in Sb["Sbar_coeffs"]],
           "S": [[_h_entry(e) for e in row] for row in S["S_coeffs"]],
           "T": Tpar}
    path = os.path.join(out, "H_%s.json.gz" % key)
    with gzip.open(path, "wt") as f:
        json.dump(res, f)
    return path, len(anti)


if __name__ == "__main__":
    if "--family-H" in sys.argv:
        main_H(sys.argv[1:])
    elif "--family-G" in sys.argv:
        main_G(sys.argv[1:])
    else:
        main()
