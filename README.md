# homfly: colored HOMFLY-PT polynomials of knots

This package computes **colored HOMFLY-PT polynomials** H_R(K; A, q) of knots in
arbitrary representations R (Young diagrams). The target is the full
Rolfsen / Hoste–Thistlethwaite table up to **12 crossings** (2977 knots), plus
parametric families: torus knots, 2-bridge knots, and double-braid knots in
rectangular representations.

Every result is an exact Laurent polynomial in `A, q`. Internally no symbolic
algebra is used. Each method is a **black box that evaluates H_R at a point of a
finite field GF(p)**. A reconstruction layer then recovers the polynomial:
degree detection → dense (later sparse) interpolation → CRT over primes →
integer lift or rational reconstruction → verification at fresh points.

```python
from homfly import homfly, TorusKnot
homfly("3_1")                       # -A^4 + A^2 q^2 + A^2 q^-2
homfly("4_1", R=(2,))               # 3-strand Racah blocks (family P)
homfly("7_4", R=(3, 2))             # two-bridge, exclusive Racah data
homfly(TorusKnot(3, 4), R=(2, 1))   # Rosso–Jones
```

```
$ PYTHONPATH=src python -m homfly.cli compute 4_1 --rep 2
$ PYTHONPATH=src python -m homfly.cli compute --torus 3,5 --rep 3
$ PYTHONPATH=src python -m homfly.cli info 10_124
$ python -m pytest                  # all tests, ~1 min
$ python scripts/validate_fundamental.py 12   # all 2977 knots vs KnotInfo
```

## Status

| Component | Status |
|---|---|
| Fields: ℚ, GF(p) (with Tonelli–Shanks √), ℂ | ✅ |
| Laurent polynomials, parser | ✅ |
| Newton interpolation with early termination, degree detection, dense bivariate | ✅ |
| CRT, symmetric lift, Wang and MQRR rational reconstruction, modular pipeline | ✅ |
| Sparse interpolation (Zippel / Ben-Or–Tiwari), Thiele rational interpolation | ⏳ M7 |
| Young diagrams, S_n characters, Adams operations, Littlewood–Richardson, dim_q | ✅ |
| Composite representations (R ⊗ R̄) | ⏳ M3 |
| Hecke algebra, seminormal form | ✅ |
| **Fundamental HOMFLY, any braid** (Ocneanu trace) | ✅ matches KnotInfo for all 2977 knots ≤ 12 crossings |
| **Colored HOMFLY by cabling + idempotents** | ✅ validation reference (m·\|R\| ≲ 10) |
| **Rosso–Jones** (torus knots, any R) | ✅ agrees with cabling |
| **RT in multiplicity spaces, m strands, from Racah matrices** | ✅ engine; validated with fundamental Racah matrices on 3 and 4 strands |
| **Racah tables, family P** (release v1.0.0): 3-strand inclusive blocks, \|R\| ≤ 5 | ✅ imported, fast GF(p) path; agrees with cabling and Rosso–Jones |
| **Two-bridge knots from exclusive data**: family P (C, T̄²) for \|R\| ≤ 5, [6], [1⁶]; family H (S̄, T̄²) for [5,1], [4,1,1], [3,3], [2,2,2], [3,1³], [2,1⁴] | ✅ all 362 two-bridge knots ≤ 12 crossings; P agrees with the 3-strand data; H agrees with P on [6] and passes the sl_N reductions |
| **3-strand, 6-box**: [6], [1⁶] (family P generator export); [4,2], [2,2,1,1], [3,2,1] (families G/F, A-independent U_Q, float64-BLAS mod p < 2²¹) | ✅ agree with Rosso–Jones T[3,4], T[3,5]; transposition; G two-bridge |
| **Two-bridge [4,2], [2,2,1,1]** (family G S̄, alternating chain) | ✅ agrees with the G 3-strand data on all 30 overlap knots |
| Two-bridge [3,2,1] (family C, 270 MB S̄) | ⏳ |
| Racah data model, JSON store, checks | ✅ |
| Eigenvalue hypothesis | 2×2 ✅, 3×3–5×5 ⏳ M4 |
| Highest-weight Racah matrices | ⏳ M5 |
| Arborescent / 2-bridge (exclusive S, S̄) | ⏳ M3 |
| Double-braid knots, rectangular R | ⏳ M6 |
| SQLite results database, batch tabulation, CLI | ✅ |

## Which knot × representation can be computed now

| Knots (≤ 12 crossings) | Representations | Method |
|---|---|---|
| all 2977 | [1] | Hecke (also two-bridge / 3-strand when applicable) |
| 185 with braid index ≤ 3 | every R with \|R\| ≤ 5; [6], [1⁶]; [4,2], [2,2,1,1], [3,2,1] | `racah-3strand`, `racah-3strand-U` |
| 362 two-bridge | every R with \|R\| ≤ 6 except [3,2,1] | `two-bridge` |
| torus knots T[m,n] | any R | `rosso-jones` |
| any braid (small) | m·\|R\| ≲ 10 | `cabling` (reference) |

The six 6-box representations [5,1], [4,1,1], [3,3], [2,2,2], [3,1³], [2,1⁴] have
no 3-strand blocks in the release (only S̄ and the mixed S), so they are two-bridge-only.
[4,2], [2,2,1,1], [3,2,1] 3-strand data live in `data/racah/large/` (not committed,
~190 MB); run `scripts/fetch_racah_release.py` or `scripts/import_racah_uform.py`.

`homfly("name", R)` chooses the presentation automatically: 3-strand braid first,
then two-bridge (chirality fixed against KnotInfo), then the general braid.

## Documentation

* [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): layers, data flow, extension points
* [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md): normalisation of A, q, framing, and how it is pinned
* [`docs/METHODS.md`](docs/METHODS.md): the formulas behind each method, their domains, costs, and references
* [`docs/RACAH_FORMAT.md`](docs/RACAH_FORMAT.md): how to upload your inclusive/exclusive Racah matrices
* [`docs/ROADMAP.md`](docs/ROADMAP.md): milestones and the plan to cover the 12-crossing table

## Layout

```
src/homfly/
  algebra/         fields.py  laurent.py  quantum.py  linalg.py
  reconstruction/  interpolation.py  crt.py  pipeline.py
  reps/            partitions.py  characters.py (MN, Adams, LR)  qdim.py  composite.py*
  hecke/           seminormal.py
  racah/           portable.py (family-P data)  model.py  provider.py  store.py  fundamental.py
                   eigenvalue_hypothesis.py  highest_weight.py*  adapters.py*  checks.py
  knots/           braid.py  families.py  table.py  data/knotinfo_upto12.csv
  methods/         base.py  hecke_fundamental.py  cabling.py  rosso_jones.py
                   rt_braid.py  racah3.py  two_bridge.py  arborescent.py*  double_braid.py*
  checks/          structural.py (special polynomial, transposition, DE)
  io/              export.py  database.py
  compute.py       cli.py  conventions.py
scripts/           tabulate.py  validate_fundamental.py  extract_knotinfo.py
                   fetch_racah_release.py  import_racah_portable.py
data/racah/portable/  converted family-P tables (inclusive_<R>, exclusive_<R>)
tests/
(* = interface and specification only)
```

Knot data comes from KnotInfo (C. Livingston, A. H. Moore, knotinfo.org) via the
`database_knotinfo` package.
