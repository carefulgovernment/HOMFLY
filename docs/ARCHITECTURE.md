# Architecture

## Guiding principle: black boxes over a field, then reconstruction

Colored HOMFLY polynomials are large. At 12 crossings in R = [2,1] they have
thousands of terms, while the intermediate objects (Racah matrices, unreduced
sums, dim_q) are *rational functions* with square roots. Symbolic
manipulation of those intermediates is the classic bottleneck.

The package therefore separates two concerns:

1. **Methods** (`methods/`) evaluate the *reduced* H_R(K; A₀, q₀) at a
   numerical point in a field F. They are written once, generically, and run
   over F = GF(p) in production, over ℚ for exact debugging, and over ℂ for
   numerical checks and root-of-unity specialisations.
2. **Reconstruction** (`reconstruction/`) turns the black box into a Laurent
   polynomial:

```
             ┌─────────────── for each prime p ───────────────┐
 method  ──▶ │ detect exponent box (univariate early          │
 black box   │ termination on random lines)                   │
 (F,A,q)→F   │ dense tensor-grid interpolation  (M7: sparse)  │──▶ images mod p
             └────────────────────────────────────────────────┘
                                   │ CRT
                                   ▼
                symmetric lift (ℤ coefficients) or rational reconstruction
                                   │ stabilises over two consecutive primes?
                                   ▼
                  verification at fresh random points / fresh prime
                                   ▼
                          Laurent in (A, q)
```

Bad points (division by zero, a quantum number vanishing, a missing square root
in GF(p)) raise `BadPoint`, and the drivers resample.

## Layers (bottom-up)

| Layer | Contents | Depends on |
|---|---|---|
| `algebra` | `QQ`, `GF(p)`, `CC` fields; `Laurent`; quantum numbers; small dense linear algebra | — |
| `reconstruction` | Newton / early termination / bivariate interpolation, CRT, Wang and MQRR, pipeline | algebra |
| `reps` | partitions, SYT, S_n characters (Murnaghan–Nakayama via β-numbers), Adams operations, LR, dim_q, θ_R; composite reps (M3) | algebra |
| `hecke` | Young's rational seminormal form of H_n(q) | reps |
| `racah` | Racah data model, providers (store, fundamental, eigenvalue hypothesis, highest weight), consistency checks | reps, algebra |
| `knots` | `Braid` (writhe, components, cabling), families (`TorusKnot`, `TwoBridge`, `DoubleBraid`), KnotInfo table | algebra |
| `methods` | engines implementing `Method.evaluate(knot, R, F, A, q)` | all above |
| `checks` | structural identities that are independent of the method | algebra |
| `io`, `compute`, `cli` | export, SQLite results DB, method selection, command line | all |

## Method interface

```python
class Method:
    name: str
    steps = (2, 2)                       # H ∈ ℤ[A^±2, q^±2] for knots
    def supports(self, knot, R) -> bool
    def evaluate(self, knot, R, F, A, q) # reduced, standard convention, value in F
    def cost_estimate(self, knot, R) -> float
```

`compute.choose_method` currently takes the first method that supports the
input: Rosso–Jones → Hecke (fundamental) → RT/arborescent (if Racah tables are
loaded) → cabling. M9 replaces this with cost-based selection, and it can run
two methods and cross-check them.

## Square roots (key design issue for modular arithmetic)

Unitary Racah matrices contain √(ratios of quantum numbers). Over GF(p) a
square root exists at only about half the points *per distinct radicand*, so
naive evaluation fails for large blocks. Strategies, in order of preference:

1. **No roots at all.** Only U and U⁻¹ enter the trace, so any non-unitary,
   rational normalisation of eigenvectors works. The seminormal Hecke form is
   an example. The highest-weight method (M5) should produce rational U
   directly.
2. **Gauge rationalisation** (M4) for stored tables whose entries are single
   terms s·√r. Rescale the path basis by c_path with c_j/c_i ∼ √r_ij along a
   spanning tree of the path graph; this is consistent when every cycle
   product of radicands is a square. For 3 strands the row/column scalings
   are free, because σ₁ is diagonal.
3. **Point selection.** Choose (A, q) so that all radicands are squares.
   This is fine for a handful of radicands (the current fundamental provider
   does this implicitly by resampling).
4. **Multiquadratic extension arithmetic.** F_p(√a₁, …, √a_k) as a vector space
   of dimension 2^k. This is a last resort for blocks with multiplicities
   whose entries are sums of different roots.

## Performance plan

Pure Python is used now for clarity. The hot spots are the vector-generator
products in `hecke/seminormal.py` and `methods/rt_braid.py`, plus Newton
interpolation. Planned swaps (M8), with no interface change:

* python-flint `nmod` / `nmod_mat` for GF(p) arithmetic and dense blocks;
* batching: evaluate many (A, q) points at once (vectorised over points), which
  maps naturally to numpy `uint64` with Montgomery reduction or to a small
  C++/Rust kernel;
* parallelism: points and primes are embarrassingly parallel
  (`scripts/tabulate.py` already parallelises across knots);
* caching of R-independent data: path spaces, LR decompositions, and Racah
  blocks per (prime, point).
