"""H1 scan: H_R(A = q^{-m}, q) versus the U_q gl(N|N+m) vertex model on V_R (scripts/nonrect_supergroup.py).

For every partition R with |R| <= 6 and each superalgebra in SPECS, builds V_R at one generic complex q,
records dim V_R, and compares the (1,1)-tangle scalar with the data (mirror convention, see
nonrect_supergroup.py) on all knots with data whose KnotInfo braid word is cheap enough
(dim^(2b+1) <= BUDGET, b = braid index), restricted to the knot list KNOTS (default: <= MAXC = 10 crossings
plus the anomalous 11/12-crossing knots).

usage: python3 scripts/nonrect_h1_scan.py > data/nonrect_h1_check.tsv
"""
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nonrect_supergroup as ns  # noqa: E402

Q = complex(os.environ.get("QV", "0.83+0.31j"))
BUDGET = float(os.environ.get("BUDGET", "2e9"))
MAXC = int(os.environ.get("MAXC", "10"))
SPECS = os.environ.get("SPECS", "01,011,0111,0011,00111").split(",")
ANOM = "11n_34 11n_42 11n_67 11n_73 11n_97 12n_23 12n_51 12n_63 12n_132 12n_268 12n_293 12n_313 12n_430 12n_457".split()


def parts(n, mx=None):
    if n == 0:
        yield ()
        return
    mx = mx or n
    for a in range(min(n, mx), 0, -1):
        for p in parts(n - a, a):
            yield (a,) + p


def knots():
    rows = ns.knot_rows()
    out = []
    for name, row in rows.items():
        c = int(row["crossing_number"])
        if (c <= MAXC or name in ANOM) and os.path.exists(os.path.join(ns.DATA, name + ".txt")):
            out.append((name, int(row["braid_index"]), ns.braid_word(row)))
    return sorted(out)


def job(args):
    spec, R = args
    par = [int(c) for c in spec]
    N, M = par.count(0), par.count(1)
    mod = ns.Module(par, R, Q, kmax=int(os.environ.get("KMAX", "60")))
    k = mod.k
    res = dict(ok=0, bad=0, worst=0.0, maxb=0, bad_list=[], scal=0.0)
    if k and mod.C is not None:
        for name, b, word in knots():
            if float(k) ** (2 * b + 1) > BUDGET:
                continue
            h = ns.data_value(name, R, Q ** (M - N), 1 / Q)
            if h is None:
                continue
            v, dev = mod.knot_value(word, check_scalar=(name in ("3_1", "4_1", "5_2")))
            rel = abs(v - h) / max(1.0, abs(h))
            res["scal"] = max(res["scal"], dev / max(1.0, abs(v)))
            res["worst"] = max(res["worst"], rel)
            res["maxb"] = max(res["maxb"], b)
            if rel < 1e-7:
                res["ok"] += 1
            else:
                res["bad"] += 1
                res["bad_list"].append(name)
    m = M - N
    hook = len(R) <= N or R[N] <= M          # R_{N+1} <= M
    if N == 1:
        typ = "zero" if not hook else ("typical" if R[0] >= m + 1 else "atypical")
    else:
        typ = "zero" if not hook else "-"
    return [spec, "gl(%d|%d)" % (N, M), m, "[" + ",".join(map(str, R)) + "]", k, typ, res["ok"], res["bad"],
            "%.1e" % res["worst"], res["maxb"], "%.1e" % res["scal"], "%.1e" % mod.worst_proj,
            ",".join(res["bad_list"]) or "-"]


def main():
    tasks = [(s, R) for s in SPECS for n in range(1, 7) for R in parts(n)]
    print("# vertex model vs data at q = %s (data at A = q'^-m, q' = 1/q); agree = relative difference < 1e-7" % Q)
    print("parities\talgebra\tm\tR\tdimV_R\tclass\tagree\tmismatch\tworst_rel\tmax_braid_index\tscalar_dev\t"
          "proj_err\tmismatching_knots")
    with Pool(int(os.environ.get("NPROC", os.cpu_count()))) as pool:
        for row in pool.imap(job, tasks):
            print("\t".join(map(str, row)), flush=True)


if __name__ == "__main__":
    main()
