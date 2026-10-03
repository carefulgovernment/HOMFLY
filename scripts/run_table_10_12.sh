#!/bin/bash
# Colored HOMFLY (all |R| <= 6) for the 10-12 crossing knots, easiest classes first.
# Resumable: every (knot, R) is cached in data/homfly/.cache, so rerunning after
# an interruption only redoes the tasks that were in flight.
#
#   scripts/run_table_10_12.sh [jobs]
set -e
# one BLAS thread per worker: the jobs already use all cores
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")/.."
J=${1:-4}
C=${CLASSES:-/tmp/homfly_classes}
[ -f "$C/easy.txt" ] || {
  python3 scripts/classify_knots.py --crossings 10 11 12 --out "$C"
  cat "$C/torus.txt" "$C/two_bridge.txt" "$C/montesinos.txt" "$C/algebraic.txt" > "$C/easy.txt"
}
SLOW="cabling-paths racah-3strand-U cabling"
run() { python3 scripts/compute_table.py --out data/homfly --jobs "$J" --partial --derive-transposed "$@"; }
# 1. torus, two-bridge, Montesinos, other algebraic (arborescent): |R| <= 5, then |R| <= 6
run --knots-file "$C/easy.txt" --max-size 5 --exclude-methods $SLOW
run --knots-file "$C/easy.txt" --max-size 6 --exclude-methods $SLOW
# 2. other 3-braids: 3-strand Racah data (the 6-box U-form ones are slow)
run --knots-file "$C/braid3.txt" --max-size 5 --exclude-methods cabling-paths cabling
run --knots-file "$C/braid3.txt" --max-size 6 --exclude-methods cabling-paths cabling
# 3. the rest (polyhedral): cabling in multiplicity spaces, small R and few
#    strands first (cost grows steeply with |R| x strands)
[ -f "$C/other_by_strands.txt" ] || python3 -c "
import sys; sys.path.insert(0, 'src')
from homfly.knots.table import load_table
t = load_table()
names = [x.strip() for x in open('$C/other.txt') if x.strip()]
print('\n'.join(sorted(names, key=lambda n: t[n].braid.strands)))" > "$C/other_by_strands.txt"
run --knots-file "$C/other_by_strands.txt" --max-size 3 --exclude-methods cabling racah-3strand-U
run --knots-file "$C/other_by_strands.txt" --max-size 4 --exclude-methods cabling racah-3strand-U
echo "ALL STAGES DONE"
