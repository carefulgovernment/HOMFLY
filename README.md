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
homfly(DoubleBraid(1, -1), R=(4, 2, 1))  # 4_1 in [4,2,1]: interpolation formula
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
| **3-strand, all 11 six-box reps**: [6], [1⁶] (family P generator export); the other nine from A-independent U_Q (families G/F, release v1.1), float64-BLAS mod p < 2²¹ | ✅ Rosso–Jones T[3,4], T[3,5]; transposition pairs; = two-bridge on all 30 overlap knots |
| **Two-bridge [4,2], [2,2,1,1], [3,2,1]** (family G S̄, alternating chain) | ✅ = G/F 3-strand data on all 30 overlap knots |
| **Montesinos knots K(p₁/q₁;…;p_k/q_k)**, tangle calculus with S̄ and the mixed S (families P, H, G; mixed S of [2,1,1] and the non-rectangular 5-box reps generated with the release's G engine, `scripts/make_mixed_S_gtpath.sh`) | ✅ fundamental = KnotInfo on all 721 Montesinos knots ≤ 12 crossings; colored = 3-strand data and two-bridge data for every R; independent R/Rᵀ data agree by transposition |
| **Interpolation formula for antiparallel double braids H_R(m,n)**, any R, no Racah matrices (Hopf-link characters + interpolation matrix E) | ✅ = Racah two-bridge data for all 29 R with \|R\| ≤ 6 × 12 (m,n) (348/348); \|R\| = 7: trefoil = Rosso–Jones, 4_1 amphichiral, transposition |
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
| 185 with braid index ≤ 3 | every R with \|R\| ≤ 6 (all 29) | `racah-3strand`, `racah-3strand-U` |
| 362 two-bridge | every R with \|R\| ≤ 6 (all 29) | `two-bridge` |
| 709 Montesinos (≥ 3 tangles, chirality fixed by H_[1]; not 9_42, 10_48, 10_71, 10_125, 11n_82 and 7 more with mirror-symmetric H_[1]) | every R with \|R\| ≤ 6 (all 29) | `montesinos` |
| double braids H(m,n) (4_1, 3_1, twist knots, all two-bridge knots with a 2-term even cf) | any R | `double-braid-interpolation` |
| torus knots T[m,n] | any R | `rosso-jones` |
| any braid (small) | m·\|R\| ≲ 10 | `cabling` (reference) |

The 3-strand U_Q arrays for the nine 6-box reps other than [6], [1⁶] live in
`data/racah/large/` (not committed, ~230 MB). Run `scripts/fetch_racah_release.py`
(release asset v1.1) or `scripts/import_racah_uform.py` to create them.

`homfly("name", R)` chooses the presentation automatically: 3-strand braid first,
then two-bridge and Montesinos (chirality fixed against KnotInfo), then the
general braid.

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
