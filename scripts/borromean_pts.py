"""usage: RACAH_P31=p python3 borromean_pts.py ENGINE_DIR R qfile PART NPARTS out.pkl

Traces T_Q(q) = Tr_Q (R1 R2^-1)^3 of the Borromean braid (sigma_1 sigma_2^-1)^3 on
the multiplicity spaces of R x R x R -> Q, at every q of qfile, for the channels Q
of part PART (of NPARTS, balanced by block size).  A-independent (pure Hecke
algebra), so the model's A is a dummy.  Then

    H_R(Borromean; A, q) = sum_Q dim_q(Q) T_Q(q) / dim_q(R)

(natural convention, evaluated at (A^-1, q^-1); no framing correction: every
crossing joins two different components and all linking numbers vanish).
"""
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fvn import FN, P  # noqa: E402
from ratmodel import Model  # noqa: E402
from lr import lr_coefficient, partitions, contains  # noqa: E402
from fast_mixed_ctx import FastCtx  # noqa: E402
from fast_inclusive import fast_build_inclusive, borromean_trace  # noqa: E402

R = tuple(int(x) for x in sys.argv[2].split(','))
qs = [int(L) for L in open(sys.argv[3]) if L.strip()]
part, nparts, out = int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
K = len(qs)


def lr_prod(a, b):
    res = {}
    for Q in partitions(sum(a) + sum(b)):
        Q = tuple(Q)
        if contains(a, Q):
            c = lr_coefficient(Q, a, b)
            if c:
                res[Q] = c
    return res


mult = {}
for X, a in lr_prod(R, R).items():
    for Q, b in lr_prod(X, R).items():
        mult[Q] = mult.get(Q, 0) + a * b
load = [0.0] * nparts
owner = {}
for Q in sorted(mult, key=lambda Q: -mult[Q]):
    i = min(range(nparts), key=lambda j: load[j])
    owner[Q] = i
    load[i] += mult[Q] ** 1.6 + 30.0
t0 = time.time()
M = Model(A=FN(np.full(K, 3, dtype=np.int64)), q=FN(np.array(qs, dtype=np.int64)))
ctx = FastCtx(M, R)
res = {}
# per-channel checkpoint: a killed run resumes after its last finished Q
ckpt = out + ".partial"
if os.path.exists(ckpt):
    with open(ckpt, "rb") as fh:
        while True:
            try:
                Q0, val = pickle.load(fh)
            except Exception:
                break
            res[Q0] = val
ck = open(ckpt, "ab")
log = (lambda s: print("%6.0f %s" % (time.time() - t0, s), flush=True)) if os.environ.get("FAST_LOG") else None


def sink(Q, labels, U, ev):
    res[Q[0]] = (len(labels), borromean_trace(U, ev))
    pickle.dump((Q[0], res[Q[0]]), ck)
    ck.flush()
    os.fsync(ck.fileno())


fast_build_inclusive(M, R, ctx, sink, log=log, Qsel=lambda Q: owner.get(Q[0]) == part and Q[0] not in res)
assert set(res) == {Q for Q, i in owner.items() if i == part}
pickle.dump({"P": P, "R": R, "qs": qs, "part": part, "nparts": nparts, "T": res}, open(out + ".tmp", "wb"))
os.replace(out + ".tmp", out)
print("R=%s K=%d part %d/%d: %d channels in %.0fs" % (R, K, part, nparts, len(res), time.time() - t0), flush=True)
