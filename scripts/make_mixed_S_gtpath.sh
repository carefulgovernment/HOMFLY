#!/bin/bash
# Mixed Racah matrices S (G-family conventions) for representations the release
# ships without them: [2,1,1] and the non-rectangular five-box reps.
#
# Uses the release's own engine code/mixed_S_gtpath (generic-A Gelfand-Tsetlin
# path model, vectorised over points): probe lines in A and q -> degrees and
# denominator factors -> product grid -> 2D reconstruction mod 2^31-1 ->
# fresh-point check against the direct engine mod the independent prime
# 2147483629 -> export racah_<R>_S_mixed.json -> G_<R>.json.gz.
#
#   scripts/make_mixed_S_gtpath.sh <racah_matrices_upto6> [workdir] [reps...]
#
# reps default: 2,1,1 4,1 3,2 3,1,1 2,2,1 2,1,1,1  (each a few minutes on 2 cores)
set -e
ARCH=$(realpath "$1"); W=${2:-/tmp/mixed_S_gtpath}; shift 2 || shift $#
REPS=${@:-"2,1,1 4,1 3,2 3,1,1 2,2,1 2,1,1,1"}
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$W/out"
cp "$ARCH"/code/mixed_S_gtpath/*.py "$ARCH"/code/mixed_S_gtpath/ratrec.c "$W"/
cp "$ARCH"/level6/R42/gtpath/lr.py "$W"/
cd "$W"
[ -f libratrec.so ] || gcc -O2 -shared -fPIC ratrec.c -o libratrec.so
# the export scripts take cyclotomic coefficients mod RACAH_P: must be the pipeline prime
export RACAH_P=2147483647 RACAH_P31=2147483647
C=200
grid() {  # ptsfile prefix R
  n=$(wc -l < "$1"); i=0
  while [ $((i*C)) -lt "$n" ]; do
    out=${2}_$(printf %03d $i).pkl
    if [ ! -f "$out" ]; then
      sed -n "$((i*C+1)),$(((i+1)*C))p" "$1" > "${2}_chunk$i.txt"
      python3 mix_pts.py "$3" "${2}_chunk$i.txt" "$out" >> "${2}.log" 2>&1
    fi
    i=$((i+1))
  done
}
for R in $REPS; do
  T=${R//,/}
  mkdir -p "g$T"
  python3 - <<PY
import random
P=2147483647; random.seed(1000+len("$T"))
q0=random.randrange(2,P-1); A0=random.randrange(2,P-1)
open('pA_$T.txt','w').write('\n'.join(f'{random.randrange(2,P-1)} {q0}' for _ in range(200)))
open('pQ_$T.txt','w').write('\n'.join(f'{A0} {random.randrange(2,P-1)}' for _ in range(400)))
PY
  grid "pA_$T.txt" "g$T/pa" "$R"
  grid "pQ_$T.txt" "g$T/pq" "$R"
  python3 - <<PY
import pickle,glob,numpy as np
for k,o in (('pa','probeA_$T.pkl'),('pq','probeQ_$T.pkl')):
    ds=[pickle.load(open(f,'rb')) for f in sorted(glob.glob('g$T/'+k+'_[0-9]*.pkl'))]
    d=ds[0]; d['S']=np.concatenate([x['S'] for x in ds],axis=2); d['pts']=sum([x['pts'] for x in ds],[])
    pickle.dump(d,open(o,'wb'),protocol=4)
PY
  python3 mix_an.py "probeA_$T.pkl" "probeQ_$T.pkl" "an$T.pkl"
  python3 mix_plan.py "an$T.pkl" 11 "grid${T}_p1.txt"
  grid "grid${T}_p1.txt" "g$T/p1" "$R"
  python3 mix_rec.py "an$T.pkl" "grid${T}_p1.txt" "g$T/p1" "rec$T.pkl"
  RACAH_P31=2147483629 python3 mix_fresh.py "$R" "rec$T.pkl" 30 77
  python3 mix_final.py "rec$T.pkl" "g$T/p1_000.pkl" "$R"
  python3 mix_export.py "rec$T.pkl" "$R" "out/racah_$T"
  (cd "$REPO" && python3 scripts/import_racah_montesinos.py --family-G --mixed-dir "$W/out" --reps "$T")
done
