"""Import A-independent inclusive 3-strand Racah matrices U_Q (families G and F
of the release, asset v1.1) into compact numpy archives:

    python scripts/import_racah_uform.py --archive <racah_matrices_upto6> --reps 42 2211 321

Input (per block Q ⊢ 3|R|): labels, R1 eigenvalues (sign, k) = sign q^k, and
U, U⁻¹ with entries null or [num, den] = integer coefficient lists in q (low
degree first).  Output data/racah/large/uform_<R>.npz with flat arrays:
coefficient, power, and entry pointers for num/den of U and U⁻¹, block sizes,
Q, eigenvalues.  (These files are large: not committed; see .gitignore.)
"""
import argparse
import json
import lzma
import os
import time

import numpy as np

SRC = {"42": ("R42", "gtpath", "racah_42_inclusive.json"),
       "2211": ("R2211", "gtpath", "racah_2211_inclusive.json"),
       "321": ("R321", "fused_inclusive", "racah_321_inclusive.json"),
       # v1.1: fused-path-model U_Q for the former two-bridge-only 6-box reps
       "51": ("R51", "fused_inclusive", "racah_51_inclusive.json"),
       "21111": ("R21111", "fused_inclusive", "racah_21111_inclusive.json"),
       "411": ("R411", "fused_inclusive", "racah_411_inclusive.json"),
       "3111": ("R3111", "fused_inclusive", "racah_3111_inclusive.json"),
       "33": ("R33", "fused_inclusive", "racah_33_inclusive.json"),
       "222": ("R222", "fused_inclusive", "racah_222_inclusive.json")}


def iter_blocks(path):
    """Yield blocks; handles plain JSON and 'one block per line' files, .xz or not."""
    opener = lzma.open if path.endswith(".xz") else open
    with opener(path, "rt") as f:
        head = f.read(4096)
        f.seek(0)
        if head.lstrip().startswith("{\"meta\"") and "\n" not in head.strip()[:4000]:
            for B in json.load(f)["blocks"]:
                yield B
            return
        for line in f:
            line = line.strip().rstrip(",")
            if line.startswith("{") and "\"Q\"" in line and "\"U\"" in line:
                yield json.loads(line)
            elif line.startswith("{\"meta\""):
                d = json.loads(line if line.endswith("}") else line + "]}")
                for B in d.get("blocks", []):
                    yield B


class Packer:
    """Flat (coef, pow, ptr) arrays; flushed to numpy per block to bound memory."""

    def __init__(self):
        self.chunks_c, self.chunks_p, self.chunks_ptr = [], [], []
        self.total = 0
        self.coef, self.pow, self.ptr = [], [], []

    def add(self, coeffs):
        start = len(self.coef)
        for k, c in enumerate(coeffs):
            if c:
                self.coef.append(c)
                self.pow.append(k)
        if len(self.coef) == start:                  # keep segments non-empty
            self.coef.append(0 if not coeffs or not any(coeffs) else coeffs[0])
            self.pow.append(0)
        self.ptr.append(self.total + start)

    def flush(self):
        if not self.coef:
            return
        self.chunks_c.append(np.array(self.coef, dtype=np.int64))
        self.chunks_p.append(np.array(self.pow, dtype=np.int16))
        self.chunks_ptr.append(np.array(self.ptr, dtype=np.int64))
        self.total += len(self.coef)
        self.coef, self.pow, self.ptr = [], [], []

    def arrays(self, name):
        self.flush()
        ptr = np.concatenate(self.chunks_ptr + [np.array([self.total], dtype=np.int64)])
        return {name + "_coef": np.concatenate(self.chunks_c),
                name + "_pow": np.concatenate(self.chunks_p),
                name + "_ptr": ptr}


def convert(archive, key, out):
    lvl, fam, fname = SRC[key]
    base = os.path.join(archive, "level%d" % sum(int(c) for c in key), lvl, fam, fname)
    path = base if os.path.exists(base) else base + ".xz"
    t = time.time()
    packs = {k: Packer() for k in ("Un", "Ud", "Vn", "Vd")}
    Qs, dims, eig_s, eig_k = [], [], [], []
    for B in iter_blocks(path):
        n = len(B["labels"])
        Qs.append(B["Q"] + [0] * (3 * sum(int(c) for c in key) - len(B["Q"])))
        dims.append(n)
        for s, k in B["R1_eigenvalues"]:
            eig_s.append(s)
            eig_k.append(k)
        for M, (pn, pd) in (("U", ("Un", "Ud")), ("Uinv", ("Vn", "Vd"))):
            for row in B[M]:
                for e in row:
                    if e is None:
                        packs[pn].add([0])
                        packs[pd].add([1])
                    else:
                        packs[pn].add(e[0])
                        packs[pd].add(e[1])
        for pk in packs.values():
            pk.flush()
    arrs = {}
    for k, p in packs.items():
        arrs.update(p.arrays(k))
    R = [int(c) for c in key]
    os.makedirs(out, exist_ok=True)
    dest = os.path.join(out, "uform_%s.npz" % key)
    np.savez_compressed(dest, R=np.array(R), Q=np.array(Qs, dtype=np.int32),
                        dims=np.array(dims, dtype=np.int32),
                        eig_s=np.array(eig_s, dtype=np.int8), eig_k=np.array(eig_k, dtype=np.int32),
                        family=np.array("G" if fam == "gtpath" else "F"), **arrs)
    return dest, len(dims), time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--reps", nargs="+", default=["42", "2211"])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "racah", "large"))
    a = ap.parse_args()
    for key in a.reps:
        dest, nb, sec = convert(a.archive, key, a.out)
        print("uform R=%s  %d blocks  %.1fs  -> %s (%.1f MB)"
              % (key, nb, sec, dest, os.path.getsize(dest) / 1e6), flush=True)


if __name__ == "__main__":
    main()
