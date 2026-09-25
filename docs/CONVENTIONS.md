# Conventions

All public results use the **standard convention**, which is defined in
`src/homfly/conventions.py` and enforced by tests.

| Item | Convention |
|---|---|
| Variables | `A`, `q`; for knots H ∈ ℤ[A^{±2}, q^{±2}] (the pipeline uses exponent steps (2, 2)) |
| Skein (fundamental) | A^{-1} H(L₊) − A H(L₋) = (q − q^{-1}) H(L₀) |
| KnotInfo | P(v, z) with v = A, z = q − q^{-1} |
| Normalisation | reduced: H_R(unknot) = 1; unreduced = reduced · dim_q R |
| dim_q R | ∏_{boxes} {A q^{c}} / {q^{h}}, where {x} = x − x^{-1}, c = content, h = hook |
| Framing | topological (writhe-corrected with θ_R = A^{\|R\|} q^{2κ_R}, κ_R = Σ contents) |
| Positive trefoil σ₁³ | H_[1] = A²q² + A²q^{-2} − A⁴ |
| sl_N | H_R(A = q^N, q) is the U_q(sl_N) invariant; e.g. H_[1,1](A = q²) = 1 |
| Transposition | H_{R^T}(A, q) = H_R(A, −1/q) |
| Mirror | H_R(K*; A, q) = H_R(K; 1/A, 1/q) |

## Natural (engine) convention

The Hecke, cabling, Rosso–Jones, and RT engines work internally with R-matrix
eigenvalues `q` (symmetric) and `−1/q` (antisymmetric) on [1]⊗[1], eigenvalue
`ε_X q^{κ_X − 2κ_R}` on X ∈ R⊗R, framing factor θ_R, and the dim_q above. The
two conventions are related by the mirror map:

    H_standard(K; A, q) = H_natural(K; 1/A, 1/q)      (conventions.natural_point)

## How the conventions are pinned (tests)

* The fundamental Hecke result equals KnotInfo for every knot up to 12 crossings.
* Cabling (framing θ_R, idempotents) equals Rosso–Jones (Adams operations) for
  R = [1], [2], [1,1], [2,1], [3].
* T[m,n] = T[n,m], and T[1,n] = unknot.
* sl_1 and sl_2 specialisations of [2] and [1,1].
* Transposition symmetry, special polynomials, and differential-expansion zeros.

External Racah tables (Mironov–Morozov–Morozov and others) come in their own
conventions: basis order, signs ε_X, A ↔ A^{-1}, q ↔ q^{-1}, and possibly
framing. Each table records its convention in the JSON file, the adapter
converts it, and a 3-strand result is compared with cabling before the
table is accepted.
