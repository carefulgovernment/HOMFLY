"""Towers e^mu_m of the families R(r) = (r, mu) on A = q^-m with the weight-block engine (nonrect_weightblock.py).

Same quantity and fit as nonrect_family.py (H_{(r,mu)}(q'^-m, q') = sum_{|j|<=J} c_j lam^j, lam = q'^r, q' = e^{i theta},
least squares over r = r0..r0+NR-1), but each value is computed with the weight-block / strand-elimination
evaluator, so braid index 4-6 and dim V_R = 24 (gl(1|3) hooks) are feasible.  Values are computed in parallel over r.

Checks reported: data cross-check (max rel. diff vertex vs data/homfly at the r with data), scalar check (value with
a second start vector, at NCHK values of r), fit residual, window saturation, and (if WIN2 is set) the fit
with a second, larger window J2 on the same values (window independence).

usage: python3 scripts/nonrect_hook_tower.py M MU KNOT [r0 NR J theta]
       MU = '-' (symmetric), '1', '1,1', '2', ...;  env NPROC (default 4), MAXSUB (default 4), J2, VALS=file.npy,
       TOLREL (support threshold, default 1e-7; the output column gap = min on-support |c| / max
       off-support |c|), HMAX (drop r whose module health = max(weight/Cartan/pivot/R-leak errors) exceeds HMAX)
"""
import cmath
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nonrect_family as nf  # noqa: E402
import nonrect_supergroup as ns  # noqa: E402
import nonrect_weightblock as wb  # noqa: E402

nf.TOL = float(os.environ.get("TOLREL", "1e-7"))   # support threshold relative to the largest coefficient

_plans = {}


def one(args):
    par, R, word, qd, chk, maxsub = args
    wm = wb.WModule(par, R, 1 / qd)
    if wm.k == 0:
        return R, 0j, float("nan"), 0, 0.0
    key = (tuple(word), wm.rel.tobytes(), maxsub)
    if key not in _plans:
        _plans[key] = wb.best_plan(word, wb.Structure(wm.rel), max_sub=maxsub)
    P = _plans[key]
    v = P.value(wm)
    dv = float("nan")
    if chk and wm.alt is not None:
        dv = abs(P.value(wm, start=wm.alt) - v) / max(1, abs(v))
    health = max(wm.wt_err, wm.H_dev, wm.mu_dev, wm.C_leak)
    return R, v, dv, wm.k, health


def gap(c, sup):
    """smallest |c_j| on the support / largest |c_j| off it (separation of signal from fit noise)"""
    on = min(abs(c[j]) for j in sup)
    off = max([abs(v) for j, v in c.items() if j not in sup] or [0.0])
    return on / off if off else float("inf")


def run(m, mu, knot, r0, NR, J, theta, nproc=4, maxsub=4, nchk=3, J2=None):
    par = [0] + [1] * (m + 1)
    word = ns.braid_word(ns.knot_rows()[knot])
    qd = cmath.exp(1j * theta)
    rs = list(range(r0, r0 + NR))
    chk_r = set(rs[:: max(1, NR // nchk)][:nchk])
    t0 = time.time()
    with Pool(nproc) as pool:
        out = pool.map(one, [(par, (r,) + tuple(mu), word, qd, r in chk_r, maxsub) for r in rs], chunksize=1)
    hmax = float(os.environ.get("HMAX", "inf"))
    keep = [i for i, o in enumerate(out) if o[4] <= hmax]
    dropped = len(out) - len(keep)
    out = [out[i] for i in keep]
    rs = [rs[i] for i in keep]
    vals = np.array([o[1] for o in out])
    scal = max([o[2] for o in out if o[2] == o[2]] or [float("nan")])
    health = max(o[4] for o in out)
    dims = sorted(set(o[3] for o in out))
    dd = 0.0
    ndata = 0
    for r, v in zip(rs, vals):
        h = ns.data_value(knot, (r,) + tuple(mu), qd ** (-m), qd)
        if h is not None:
            ndata += 1
            dd = max(dd, abs(v - h) / max(1, abs(h)))
    res = {}
    for JJ in [J] + ([J2] if J2 else []):
        c, resid, cond = nf.fit(vals, rs, qd, JJ)
        lo, hi, sup, syms = nf.analyse(c, qd)
        res[JJ] = dict(lo=lo, hi=hi, span=hi - lo, e=(hi - lo) / 4, resid=resid, cond=cond, syms=syms,
                       sat=(hi >= JJ or lo <= -JJ),
                       parity="even" if all(j % 2 == 0 for j in sup) else ("odd" if all(j % 2 for j in sup) else "mixed"),
                       coef=c, gap=gap(c, sup))
    return dict(knot=knot, m=m, mu=list(mu), rs=(rs[0], rs[-1]), J=J, J2=J2, theta=theta, dims=dims, vals=vals,
                data_check=dd, ndata=ndata, dropped=dropped, scalar_dev=scal, health=health, fits=res, time=time.time() - t0)


def fmt(d):
    f = d["fits"][d["J"]]
    out = "%s\t%d\t[%s]\t%d..%d\t%d\t%s\t%s\t%d..%d\t%g\t%s\t%s\t%.1e\t%.1e\t%.1e(%d)\t%.1e\t%.1e\t%s" % (
        d["knot"], d["m"], ",".join(map(str, d["mu"])), d["rs"][0], d["rs"][1], d["J"], "%g" % d["theta"],
        "/".join(map(str, d["dims"])), f["lo"], f["hi"], f["e"], f["parity"], ",".join(map(str, f["syms"])) or "-",
        f["resid"], f["cond"], d["data_check"], d["ndata"], d["scalar_dev"], d["health"],
        "window" if f["sat"] else "-")
    if d["J2"]:
        g = d["fits"][d["J2"]]
        out += "\tJ2=%d:%d..%d,resid=%.1e%s" % (d["J2"], g["lo"], g["hi"], g["resid"], ",window" if g["sat"] else "")
    out += "\tgap=%.1e" % f["gap"]
    if d.get("dropped"):
        out += "\tdropped %d r with module_health > HMAX" % d["dropped"]
    return out + "\t%.0fs" % d["time"]


HEADER = ("knot\tm\tmu\tr_range\tJ\ttheta\tdimV\tlam_support\te\tparity\tsym_a\tfit_resid\tcond\tdata_check(n)\t"
          "scalar_dev\tmodule_health\tflag\tsecond_window\ttime")


def main():
    m = int(sys.argv[1])
    mu = () if sys.argv[2] == "-" else tuple(int(x) for x in sys.argv[2].split(","))
    knot = sys.argv[3]
    r0 = int(sys.argv[4]) if len(sys.argv) > 4 else max(m + 1, mu[0] if mu else 1)
    NR = int(sys.argv[5]) if len(sys.argv) > 5 else 60
    J = int(sys.argv[6]) if len(sys.argv) > 6 else 24
    theta = float(sys.argv[7]) if len(sys.argv) > 7 else 2.39996
    J2 = int(os.environ["J2"]) if os.environ.get("J2") else None
    d = run(m, mu, knot, r0, NR, J, theta, nproc=int(os.environ.get("NPROC", "4")),
            maxsub=int(os.environ.get("MAXSUB", "4")), J2=J2)
    if os.environ.get("VALS"):
        np.save(os.environ["VALS"], d["vals"])
    print(fmt(d), flush=True)
    if os.environ.get("COEF"):
        for j, c in sorted(d["fits"][J]["coef"].items()):
            if abs(c) > 1e-7 * max(1, max(abs(x) for x in d["fits"][J]["coef"].values())):
                print("#   lam^%d: %.6e%+.6ei" % (j, c.real, c.imag))


if __name__ == "__main__":
    main()
