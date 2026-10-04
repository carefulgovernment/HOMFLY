"""usage: RACAH_P31=p python3 fast_mix_pts.py ENGINE_DIR R ptsfile out.pkl
Like the release's mix_pts.py (gauged mixed S at all points, vectorised), with
the fast context of fast_mixed_ctx.py."""
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fvn import FN, P  # noqa: E402
from ratmodel import Model  # noqa: E402
from mixedS import build_mixed, sbar_vacuum_column, gauge_S  # noqa: E402
from fast_mixed_ctx import FastCtx, fast_build_mixed, fast_gauge_S  # noqa: E402

R = tuple(int(x) for x in sys.argv[2].split(','))
pts = [tuple(int(x) for x in L.split()) for L in open(sys.argv[3]) if L.strip()]
K = len(pts)
t = time.time()
M = Model(A=FN(np.array([a for a, _ in pts], dtype=np.int64)), q=FN(np.array([q for _, q in pts], dtype=np.int64)))
ctx = FastCtx(M, R)
log = (lambda s: print(s, flush=True)) if os.environ.get("FAST_LOG") else None
ql, xl, S, E = fast_build_mixed(M, R, ctx, log=log)
t1 = time.time()
col0 = sbar_vacuum_column(ctx, R)
V = fast_gauge_S(ql, xl, S, col0, K)
pickle.dump({"P": P, "R": R, "pts": pts, "qlab": ql, "xlab": xl, "S": V.astype(np.uint32), "ev": E.astype(np.uint32)},
            open(sys.argv[4] + '.tmp', 'wb'), protocol=4)
os.replace(sys.argv[4] + '.tmp', sys.argv[4])
print(f"R={R} K={K} N={len(ql)} time {time.time()-t:.0f}s (build {t1-t:.0f}s) bases={ctx.stats['bases']} "
      f"fcross={ctx.stats['fcross']}", flush=True)
