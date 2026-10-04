"""Colored HOMFLY of a Montesinos knot in a representation beyond the stored
Racah data (|R| > 6), e.g.

    python scripts/pointwise_montesinos.py 9_46 4,3,1 \\
        --engine <racah_matrices_upto6>/code/mixed_S_gtpath --work /tmp/r431 --jobs 4

The mixed Racah matrix S of R is never reconstructed as rational functions:
the release engine (mix_pts.py: generic-A Gelfand-Tsetlin path model,
vectorised over points) evaluates it numerically mod p at exactly the points
the interpolation needs, and the Montesinos tangle calculus (family G
conventions, S̄ = T̄ S^-1 T S T̄) turns each point into H_R(A, q) mod p.

  1. exponent box: one q-line and one A-line (batch of points, Newton with
     a consistency tail);
  2. dense grid on that box (points fixed in advance, evaluated in chunks in
     parallel), interpolation;
  3. checks: fresh points mod an independent prime; q = 1 -> H_[1](A,1)^|R|;
     optionally the transposed representation at a few points
     (H_{R^T}(A, q) = H_R(A, -1/q)) from an independent engine run.
"""
import argparse
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from homfly.algebra.fields import GF, BadPoint  # noqa: E402
from homfly.algebra.laurent import AQ, Laurent  # noqa: E402
from homfly.compute import presentations  # noqa: E402
from homfly.io.export import to_json  # noqa: E402
from homfly.knots.families import MontesinosKnot  # noqa: E402
from homfly.methods import montesinos as MO  # noqa: E402
from homfly.reconstruction.interpolation import NewtonInterpolator, dense_bivariate  # noqa: E402

P1, P2 = 2147483647, 2147483629


def ks(Y):
    return sum(j - i for i, r in enumerate(Y) for j in range(r))


# ---------------------------------------------------------------------------
# engine: S at a batch of points
# ---------------------------------------------------------------------------

class Engine:
    def __init__(self, R, engine_dir, work, jobs, chunk, fast=False):
        self.R, self.work, self.jobs, self.chunk, self.fast = R, work, jobs, chunk, fast
        self.dir = os.path.join(work, "engine")
        if not os.path.isdir(self.dir):
            shutil.copytree(engine_dir, self.dir, ignore=shutil.ignore_patterns("__pycache__"))
            lr = os.path.join(engine_dir, "..", "..", "level6", "R42", "gtpath", "lr.py")
            shutil.copy(lr, self.dir)
        self.table = {}             # (p, A, q) -> (S uint32 N x N, ev)
        self.labels = None
        self.n_batch = 0

    def evaluate(self, p, pts):
        self.evaluate_many({p: pts})

    def evaluate_many(self, by_prime):
        """S at all points (A, q) mod p for every prime (cached); one engine
        round: the chunks of all primes run concurrently (each process pays
        the engine's set-up, the points themselves are cheap)."""
        rk = ",".join(map(str, self.R))
        jobs = []
        for i, (p, pts) in enumerate(by_prime.items()):
            todo = sorted({(a, q) for a, q in pts if (p, a, q) not in self.table})
            # the first prime takes the free slots, every further prime one
            n = max(1, min(self.jobs - len(by_prime) + 1 if i == 0 else 1, len(todo)))
            size = min(self.chunk, -(-len(todo) // n)) if todo else 1
            jobs += [(p, todo[i:i + size]) for i in range(0, len(todo), size)]
        procs = []
        for k, (p, chunk) in enumerate(jobs):
            tag = "b%03d_%d" % (self.n_batch, k)
            pf = os.path.join(self.work, tag + ".txt")
            out = os.path.join(self.work, tag + ".pkl")
            with open(pf, "w") as f:
                f.write("".join("%d %d\n" % x for x in chunk))
            if not os.path.exists(out):          # resumable
                while sum(pr is not None and pr.poll() is None for _, _, pr in procs) >= self.jobs:
                    time.sleep(10)
                env = dict(os.environ, RACAH_P=str(p), RACAH_P31=str(p), OMP_NUM_THREADS="1")
                cmd = ([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fast_mix_pts.py"),
                        self.dir, rk, pf, out] if self.fast else [sys.executable, "mix_pts.py", rk, pf, out])
                pr = subprocess.Popen(cmd, cwd=self.dir, env=env,
                                      stdout=open(out + ".log", "w"), stderr=subprocess.STDOUT)
            else:
                pr = None
            procs.append((p, out, pr))
        self.n_batch += 1
        for p, out, pr in procs:
            if pr is not None and pr.wait() != 0:
                raise RuntimeError("engine failed: see %s.log" % out)
            d = pickle.load(open(out, "rb"))
            assert d["P"] == p
            if self.labels is None and p == P1:
                self.labels = (d["qlab"], d["xlab"], d["ev"][:, 0].astype(np.int64), d["pts"][0])
            for k, pt in enumerate(d["pts"]):
                self.table[(p,) + tuple(pt)] = d["S"][:, :, k]
        print("engine: %s points done" % {p: len(v) for p, v in by_prime.items()}, flush=True)


class PointData(MO.HData):
    """Family-G data whose S comes from the engine table instead of formulas."""

    def __init__(self, R, engine):
        qlab, xlab, ev, (A0, q0) = engine.labels
        self.R, self.family, self.engine = tuple(R), "G", engine
        self.anti = [{"Z": list(X[0]), "Zp": list(X[1]), "a": str(i), "b": str(j), "eps": 1} for X, i, j in xlab]
        self.par = [{"Q": list(Q[0]), "a": str(a), "b": str(b)} for Q, a, b in qlab]
        p0 = P1
        self.Tpar = []
        for r, (Q, a, b) in enumerate(qlab):
            k = ks(Q[0]) - 2 * ks(R)
            x = pow(q0, k % (p0 - 1), p0)
            v = int(ev[r]) % p0
            s = 1 if v == x else (-1 if v == (p0 - x) % p0 else None)
            assert s is not None, "unexpected eigenvalue"
            self.Tpar.append([s, 0, k])
        self.n = len(xlab)
        self.blocks_anti = self._blocks([((tuple(X[0]), tuple(X[1])), i, j) for X, i, j in xlab])
        self.blocks_par = self._blocks([(tuple(Q[0]), a, b) for Q, a, b in qlab])
        self.vac = next(k for k, (X, i, j) in enumerate(xlab) if not X[0] and not X[1])
        self.mats = {}
        self._dims = None
        self._eig = None
        self._A = None
        outer = self

        class _S:
            def evaluate(self, p, pw, q):
                return outer.engine.table[(p, outer._A, q)].astype(np.int64)
        self.flat = {"S": _S()}

    def evaluate_np(self, p, A, q):
        self._A = A
        return super().evaluate_np(p, A, q)


def H_at(data, fracs, p, A, q):
    return MO.natural_value_np(fracs, data.evaluate_np(p, int(A), int(q)), p)


# ---------------------------------------------------------------------------
# reconstruction
# ---------------------------------------------------------------------------

def line_exponents(xs, ys, F, step, lower, tail=12):
    """Exponents of a Laurent polynomial from values on a line; ``lower`` must
    be <= the lowest exponent and the points must cover the span + tail."""
    ni = NewtonInterpolator(F)
    agree = 0
    for x, y in zip(xs, ys):
        X, Y = x ** step, y * x ** (-lower)
        if len(ni) and ni.value(X) == Y:
            agree += 1
            if agree >= tail:
                break
        else:
            agree = 0
        ni.add(X, Y)
    if agree < tail:
        return None
    c = ni.monomial_coefficients()
    nz = [i for i, a in enumerate(c) if a != 0]
    if nz and nz[0] == 0:
        raise ValueError("lower bound %d too high" % lower)
    return (lower + step * nz[0], lower + step * nz[-1]) if nz else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("knot")
    ap.add_argument("R")
    ap.add_argument("--engine", required=True, help="release code/mixed_S_gtpath")
    ap.add_argument("--work", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--qline", type=int, default=520, help="points on the q-line")
    ap.add_argument("--aline", type=int, default=90, help="points on the A-line")
    ap.add_argument("--qlower", type=int, default=-400)
    ap.add_argument("--alower", type=int, default=-160)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fast", action="store_true", help="fast_mix_pts.py (fast_mixed_ctx) instead of mix_pts.py")
    ap.add_argument("--check-transposed", type=int, default=0, metavar="N",
                    help="also evaluate R^T at N points: H_{R^T}(A,q) = H_R(A,-1/q)")
    a = ap.parse_args()
    R = tuple(int(x) for x in a.R.split(","))
    os.makedirs(a.work, exist_ok=True)
    mk = next(K for K in presentations(a.knot) if isinstance(K, MontesinosKnot))
    fracs = mk.fractions()
    print("%s = N(%s), R = %s" % (a.knot, ", ".join(map(str, fracs)), R), flush=True)
    eng = Engine(R, a.engine, a.work, a.jobs, a.chunk, fast=a.fast)
    F = GF(P1)
    rng = random.Random(1)
    steps = (2, 2)
    t0 = time.time()

    # 1. exponent box from one q-line and one A-line
    A0, q0 = F.random_element(rng), F.random_element(rng)
    qs = [F.random_element(rng) for _ in range(a.qline)]
    As = [F.random_element(rng) for _ in range(a.aline)]
    F2 = GF(P2)
    r2 = random.Random(3)
    pts2 = [(F2.random_element(r2), F2.random_element(r2)) for _ in range(6)]
    eng.evaluate_many({P1: [(int(A0), int(q)) for q in qs] + [(int(x), int(q0)) for x in As],
                       P2: [(int(x), int(y)) for x, y in pts2]})
    data = PointData(R, eng)
    yq = [F(H_at(data, fracs, P1, A0, q)) for q in qs]
    yA = [F(H_at(data, fracs, P1, x, q0)) for x in As]
    qbox = line_exponents(qs, yq, F, steps[1], a.qlower)
    Abox = line_exponents(As, yA, F, steps[0], a.alower)
    if qbox is None or Abox is None:
        raise SystemExit("exponent box not determined: more line points or lower bounds needed (q %s, A %s)"
                         % (qbox, Abox))
    print("box A %s q %s  (%.0fs)" % (Abox, qbox, time.time() - t0), flush=True)

    # 2. grid: record the points dense_bivariate will ask for, evaluate, interpolate
    seen = []

    def dry(x, y):
        seen.append((int(x), int(y)))
        return F.zero
    dense_bivariate(dry, F, Abox, qbox, steps=steps, rng=random.Random(2))
    eng.evaluate(P1, seen)
    img = dense_bivariate(lambda x, y: F(H_at(data, fracs, P1, x, y)), F, Abox, qbox, steps=steps,
                          rng=random.Random(2))
    terms = {}
    for k, v in img.items():
        c = int(v)
        c = c - P1 if c > P1 // 2 else c
        if c:
            terms[k] = c
    H = Laurent(terms, AQ)
    big = max(abs(c) for c in terms.values())
    print("interpolated: %d terms, max |c| = %d  (%.0fs)" % (len(terms), big, time.time() - t0), flush=True)

    # 3. checks
    ok = all(H.evaluate([x, y], one=F2.one) == F2(H_at(data, fracs, P2, x, y)) for x, y in pts2)
    print("fresh points mod %d: %s" % (P2, ok), flush=True)
    from homfly.checks.structural import specialise_q1
    from homfly.knots.table import load_table
    H1 = load_table()[a.knot].homfly_reference()
    q1 = specialise_q1(H) == specialise_q1(H1) ** sum(R)
    print("q=1 check H_R(A,1) = H_[1](A,1)^|R|: %s" % q1, flush=True)
    out = a.out or os.path.join(a.work, "%s_%s.json" % (a.knot, "".join(map(str, R))))
    json.dump({"knot": a.knot, "R": list(R), "method": "montesinos-pointwise (mixed_S_gtpath)",
               "seconds": round(time.time() - t0, 1), "primes": 1, "verified": ok, "q1_check": q1,
               "poly": to_json(H)}, open(out, "w"))
    print("wrote %s" % out, flush=True)
    tr = None
    if a.check_transposed:
        RT = tuple(sum(1 for x in R if x > j) for j in range(R[0]))
        engT = Engine(RT, a.engine, os.path.join(a.work, "transposed"), 1, a.chunk, fast=a.fast)
        os.makedirs(engT.work, exist_ok=True)
        rT = random.Random(5)
        ptsT = [(F.random_element(rT), F.random_element(rT)) for _ in range(a.check_transposed)]
        engT.evaluate(P1, [(int(x), int(y)) for x, y in ptsT])
        dataT = PointData(RT, engT)
        tr = all(H.evaluate([x, -1 / y], one=F.one) == F(H_at(dataT, fracs, P1, x, y)) for x, y in ptsT)
        print("transposition check with an independent %s computation at %d points: %s"
              % (list(RT), len(ptsT), tr), flush=True)
        rec = json.load(open(out))
        rec["transposition_check"] = tr
        json.dump(rec, open(out, "w"))
    if not (ok and q1) or tr is False:
        raise SystemExit("checks failed")


if __name__ == "__main__":
    main()
