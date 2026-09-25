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
homfly("4_1", R=(2,))               # via cabling (or Racah tables once loaded)
homfly(TorusKnot(3, 4), R=(2, 1))   # Rosso–Jones
```

```
$ PYTHONPATH=src python -m homfly.cli compute 4_1 --rep 2
$ PYTHONPATH=src python -m homfly.cli compute --torus 3,5 --rep 3
$ PYTHONPATH=src python -m homfly.cli info 10_124
$ python -m pytest                  # 32 tests, ~8 s
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
| Racah data model, JSON store, checks | ✅; adapter for your tables ⏳ M2 |
| Eigenvalue hypothesis | 2×2 ✅, 3×3–5×5 ⏳ M4 |
| Highest-weight Racah matrices | ⏳ M5 |
| Arborescent / 2-bridge (exclusive S, S̄) | ⏳ M3 |
| Double-braid knots, rectangular R | ⏳ M6 |
| SQLite results database, batch tabulation, CLI | ✅ |

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
  racah/           model.py  provider.py  store.py  fundamental.py
                   eigenvalue_hypothesis.py  highest_weight.py*  adapters.py*  checks.py
  knots/           braid.py  families.py  table.py  data/knotinfo_upto12.csv
  methods/         base.py  hecke_fundamental.py  cabling.py  rosso_jones.py
                   rt_braid.py  arborescent.py*  double_braid.py*
  checks/          structural.py (special polynomial, transposition, DE)
  io/              export.py  database.py
  compute.py       cli.py  conventions.py
scripts/           tabulate.py  validate_fundamental.py  extract_knotinfo.py
data/racah/        drop your Racah tables here
tests/
(* = interface and specification only)
```

Knot data comes from KnotInfo (C. Livingston, A. H. Moore, knotinfo.org) via the
`database_knotinfo` package.
