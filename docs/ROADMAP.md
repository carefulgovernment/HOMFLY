# Roadmap

Each milestone ends with tests that compare against an *independent* method.

## M0: foundations ✅ (this commit)
Fields, Laurent polynomials, interpolation, CRT, rational reconstruction, and
the modular pipeline. Partitions, characters, Adams operations, LR, and dim_q.
Seminormal Hecke form. KnotInfo table (≤ 12 crossings). Methods: Hecke
(fundamental), cabling, Rosso–Jones, and the generic RT m-strand engine.
CLI, SQLite DB, and batch tabulation.

## M1: fundamental table and pipeline hardening
* Tabulate H_[1] for all 2977 knots (`scripts/tabulate.py --rep 1 --max-crossings 12`)
  and diff against KnotInfo (evaluation-level agreement is already verified).
* Pipeline: adaptive prime count, a report of coefficient sizes, and better
  degree bounds from Morton–Franks–Williams (A-span ≤ 2(m−1)|R|) to skip
  detection.

## M2: user Racah tables → 3-strand and 4-strand colored HOMFLY
* Adapter for the uploaded format → canonical JSON; conventions pinned by
  3-strand results vs cabling for [2], [1,1], [2,1].
* Coverage: 185 knots with braid index 3 in every representation with
  |R| ≤ 6 boxes of Q (as available), then 1022 knots with braid index 4 when
  U[Y,R,R→Z] for Y ∈ R⊗R is available.
* Gauge rationalisation of stored √-matrices (ARCHITECTURE § square roots).

## M3: arborescent / 2-bridge via exclusive matrices
* Composite representations (R ⊗ R̄), antiparallel eigenvalues.
* Fingers and propagators; 2-bridge chains from `two_bridge_notation`
  (362 knots ≤ 12 crossings); then Montesinos (721 more, `montesinos_notation`);
  then general arborescent from Conway notation.
* Validation: fundamental vs Hecke, [2] and [2,1] vs cabling.

## M4: Racah matrices without tables, part 1
* Eigenvalue hypothesis 3×3, 4×4, 5×5.
* Size-agnostic alternative: solve Yang–Baxter + orthogonality for U at each
  GF(p) point (Newton / Gröbner on small systems), then feed RT.

## M5: highest-weight method
Modular highest-weight vectors in U_q(sl_N) give rational (non-unitary)
inclusive and exclusive Racah matrices for |R| > 6 and for the large
intermediate Y needed by 5- and 6-strand braids.

## M6: double-braid knots in rectangular R
Closed-form S̄ for [r] and [r^s], an evolution formula symbolic in (m, n),
factorised differential expansion checks, and comparison with arborescent
and cabling results.

## M7: reconstruction upgrades
Sparse interpolation (Zippel; Ben-Or–Tiwari with early termination) for
large sparse outputs. Thiele or multivariate rational interpolation for
unreduced quantities and for Racah entries. Reconstruction of symbolic
families (dependence on N via A = q^N, and on twist parameters).

## M8: performance
python-flint `nmod_mat` back-end, batched evaluation over many points, an
optional C++/Rust kernel for the path-space action, multiprocessing over
points and primes, and caching of path spaces and Racah blocks.

## M9: orchestration
Cost-based method selection. Automatic cross-checks (two methods on a random
sample). Structural checks on every stored result (special polynomial,
transposition, differential expansion, sl_2 colored Jones). Export of full
tables (Mathematica, JSON, knot atlas style).

## Coverage strategy for the 12-crossing table

| Knot class (≤ 12 crossings) | Count | Primary method | Cross-check |
|---|---|---|---|
| torus | handful | Rosso–Jones | cabling |
| braid index 3 | 185 | RT 3-strand (inclusive tables) | cabling, arborescent |
| 2-bridge | 362 | arborescent chain | RT, cabling |
| Montesinos (non-2-bridge) | 721 | arborescent | RT 4/5-strand |
| braid index 4 | 1022 | RT 4-strand | arborescent |
| remaining (braid index 5–7, non-arborescent) | rest | RT 5–7 strands with HW Racah (M5) | — |

The number of Racah blocks for m strands grows quickly with |R|. The arborescent
method covers most knots with exclusive matrices only, so M3 is the main
route to the full table in large R.
