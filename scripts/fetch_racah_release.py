"""Download the Racah-matrix release, verify it and convert family P.

    python scripts/fetch_racah_release.py                 # -> data/racah/portable
    python scripts/fetch_racah_release.py --keep-archive /some/dir

Steps: download racah_matrices_upto6_v1.0.zip from the GitHub release
(carefulgovernment/HOMFLY, tag v1.0.0), check SHA-256, unpack, and run
scripts/import_racah_portable.py for every family-P representation
(inclusive 3-strand blocks: |R| <= 5; exclusive C, D̄²: |R| <= 5, [6], [1^6])
and family H (S̄, T̄² for [5,1], [4,1,1], [3,3], [2,2,2], [3,1^3], [2,1^4], [6], [1^6]).
The converted files are small (a few MB) and are also committed to the repo,
so this is only needed to regenerate them or to add more representations.
Requires sympy and numpy.
"""
import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

URL = ("https://github.com/carefulgovernment/HOMFLY/releases/download/v1.0.0/"
       "racah_matrices_upto6_v1.0.zip")
SHA256 = "a0aa3030fb8700ac39a81481201d7f675d885f66846a3b56edf758ad1373c2a6"
HERE = os.path.dirname(os.path.abspath(__file__))
INCLUSIVE = "1 2 11 3 21 111 4 31 22 211 1111 5 41 32 311 221 2111 11111".split()
EXCLUSIVE = INCLUSIVE + ["6", "111111"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-archive", help="directory to download/unpack into")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data", "racah", "portable"))
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--skip-large", action="store_true", help="skip [4,2], [2,2,1,1], [3,2,1]")
    a = ap.parse_args()
    work = a.keep_archive or tempfile.mkdtemp(prefix="racah_")
    os.makedirs(work, exist_ok=True)
    zpath = os.path.join(work, "racah_matrices_upto6_v1.0.zip")
    if not os.path.exists(zpath) or sha256(zpath) != SHA256:
        print("downloading", URL)
        urllib.request.urlretrieve(URL, zpath)
    if sha256(zpath) != SHA256:
        sys.exit("checksum mismatch")
    root = os.path.join(work, "racah_matrices_upto6")
    if not os.path.isdir(root):
        zipfile.ZipFile(zpath).extractall(work)
    imp = os.path.join(HERE, "import_racah_portable.py")
    for kind, reps in (("exclusive", EXCLUSIVE), ("inclusive", INCLUSIVE)):
        subprocess.check_call([sys.executable, imp, "--kind", kind, "--archive", root,
                               "--out", a.out, "--jobs", str(a.jobs), "--reps", *reps])
    # family H: S̄, T̄ for the other 6-box representations (no sympy needed)
    subprocess.check_call([sys.executable, os.path.join(HERE, "import_racah_sbar.py"),
                           "--archive", root, "--out", a.out])
    # [6], [1^6] 3-strand blocks (generator export, family P)
    subprocess.check_call([sys.executable, imp, "--kind", "generated", "--archive", root,
                           "--out", a.out, "--reps", "6", "111111"])
    # families G/F: A-independent U_Q for [4,2], [2,2,1,1], [3,2,1] -> data/racah/large
    # (large; not committed).  [3,2,1] needs ~4 GB RAM and ~20 min.
    if not a.skip_large:
        subprocess.check_call([sys.executable, os.path.join(HERE, "import_racah_uform.py"),
                               "--archive", root, "--reps", "42", "2211", "321"])


if __name__ == "__main__":
    main()
