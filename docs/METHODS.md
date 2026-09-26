# Methods

All formulas are given in the *natural* convention (see CONVENTIONS.md);
the engines convert to the standard convention with `natural_point`.

## 1. Hecke / Ocneanu trace: fundamental representation (implemented)

For a braid β on m strands with writhe w:

    H^{unred}_[1] = A^{-w} Σ_{Q ⊢ m} dim_q(Q) · Tr_{S^Q} ρ(β)

ρ is Young's rational seminormal representation of the Hecke algebra
(`hecke/seminormal.py`). It needs no square roots, and generators act as
2-sparse matrices on SYT bases.
*Domain:* any braid (knots and links). *Cost:* Σ_Q (f^Q)² · |β|, trivial
up to 7 strands. *Validated:* all 2977 knots ≤ 12 crossings against KnotInfo.

## 2. Cabling with Hecke idempotents: any R, small (implemented)

    H^{unred}_R = θ_R^{-w} Σ_{Q ⊢ m|R|} dim_q(Q) · Tr_{S^Q}(E_R ρ(cable_{|R|}(β)))

E_R is the minimal idempotent of H_{|R|} on the first bundle. In the
seminormal basis it is the *diagonal* projector onto tableaux whose restriction
to {1..|R|} is a fixed SYT of shape R.
*Domain:* any braid, m·|R| ≲ 10 in pure Python. *Role:* independent reference
for every colored method (it uses no Racah matrices at all).

## 3. Rosso–Jones: torus knots (implemented)

    H^{unred}_R(T[m,n]) = θ_R^{-mn} Σ_{Q ⊢ m|R|} c^Q_{R,m} θ_Q^{n/m} dim_q(Q),
    ψ_m(s_R) = s_R[p_m] = Σ_Q c^Q_{R,m} s_Q  (Adams operation).

The Adams coefficients are computed exactly from S_n characters on classes mρ
(Murnaghan–Nakayama with β-numbers). They are cheap even for large m and R.
*TODO:* torus links (a product of Adams operations over components), iterated
torus and cable knots, and closed-form symbolic output.

## 4. RT in multiplicity spaces from Racah matrices: m-strand braids (engine implemented)

Path basis R = Y₁ → Y₂ → … → Y_m = Q with Y_{k+1} ∈ Y_k ⊗ R:

* ρ(σ₁) = diag(λ_{Y₂}) with λ_X = ε_X q^{κ_X − 2κ_R};
* ρ(σ_k) acts on Y_k by the block U[Y_{k−1}, R, R → Y_{k+1}] · diag(λ) · U⁻¹.

    H^{unred}_R = θ_R^{-w} Σ_Q dim_q(Q) · Tr_{M_Q} ρ(β)

Required data: 3 strands need U[R,R,R→Q] (your inclusive tables);
4 strands also need U[Y,R,R→Z] for Y ∈ R⊗R; m strands need all Y ∈ R^{⊗(m−2)}.
Labels are opaque, so multiplicities are supported.
*Validated:* with closed-form fundamental Racah matrices against Hecke
(3 and 4 strands, over ℂ and GF(p)). *Coverage:* the braid index of the
≤12-crossing knots is 3 for 185 knots, 4 for 1022, 5 for 1419, 6 for 329,
and 7 for 17.

## 4a. 3-strand knots from the family-P tables (implemented: `racah-3strand`)

The release v1.0.0 (`racah_matrices_upto6_v1.1.zip`, racah_homfly v0.7) gives,
for every Q ∈ R⊗R⊗R, rational matrices R1, R2 (and their inverses) of σ₁ and σ₂
on the multiplicity space of Q. The topological framing factor is already
included:

    H_R^{nat}(β) = Σ_Q dim_q(Q) Tr_Q(ρ(β)) / dim_q(R),     H_std(A, q) = H_nat(1/A, 1/q)

So family P uses exactly our natural convention. 2-strand braids are
Markov-stabilised. Checked against cabling ([2], [1,1]) and against Rosso–Jones
for T[3,4] and T[3,5] in every imported R.

## 4a'. 3-strand knots from A-independent U_Q (families G/F: `racah-3strand-U`)

For [4,2], [2,2,1,1] (family G) and [3,2,1] (family F), the release gives, for
each Q ⊢ 18, R₁ = diag(ε q^{κ_Y−2κ_R}) and U_Q, U_Q⁻¹ as rational functions
of q only (R₂ = U_Q R₁ U_Q⁻¹). Then

    H_nat = θ_R^{−w} Σ_Q dim_q(Q; A, q) t_Q(q) / dim_q(R),   t_Q = Tr_Q(word)

So only dim_q(Q) carries A. The t_Q are cached per q value, which makes the
A-direction of the interpolation grid almost free (16 ms per A value vs 6.6 s
per new q for [3,2,1], whose blocks go up to 220×220). Products are exact in
float64 BLAS for p < 2²¹. The framing θ_R^{−w} at the natural point was pinned
against Rosso–Jones.

## 4b. Two-bridge knots from the family-P exclusive data (implemented: `two-bridge`)

With the all-even negative continued fraction cf = (a₁,…,a_k) of the 4-plat
(`TwoBridge.even_cf`, the same convention as racah_homfly):

    H_R^{nat} = dim_q(R) · [ C D̄²^{a₁/2−1} C D̄²^{a₂/2−1} ⋯ C ]_{vac,vac},   C = S T⁻¹ V, D̄² = T̄²

Both C and D̄² are rational, so no square roots appear, even in the
multiplicity-4 sectors at 5 boxes. KnotInfo's [p,q] does not fix the
chirality consistently, so `two_bridge.chirality` fixes it per knot from the
fundamental HOMFLY. Checked on all 362 two-bridge knots ≤ 12 crossings
([1] vs KnotInfo), and against the independent 3-strand data on every knot
that has both presentations.

Family H (6-box reps without P data) uses S̄ (Y-gauge, S̄² = 1) and
framing-free T̄:

    H_R = [S̄ T̄^{a₁} S̄ T̄^{a₂} ⋯ T̄^{a_k} S̄]_{00} / S̄_{00}

This agrees with family P for R = [6] on all 362 two-bridge knots, and passes
H_[3,3](A=q²) = H_[2,2,2](A=q³) = 1 and H_[4,1,1](A=q³) = H_[3](A=q³) (and
similar). The latter is a check across the two families.

Family G ([4,2], [2,2,1,1]) uses a rational "vacuum-dual" gauge in which
S̄² ≠ 1, so the chain alternates:

    H_nat = ⟨0| S̄⁻¹ T̄^{a₁} S̄ T̄^{a₂} S̄⁻¹ ⋯ |0⟩ / ⟨0|S̄⁻¹|0⟩   (framing-free T̄, natural point)

It agrees with the G 3-strand data on all 30 knots that have both presentations.

## 4c. Antiparallel double braids in any R: interpolation formula (implemented: `double-braid-interpolation`)

This follows the September 2026 paper "An interpolation formula for colored
HOMFLY polynomials of the figure-eight knot and of double-braid knots in
arbitrary representations" (`methods/interpolation.py`):

    H_R(m,n) = 1 + Σ_c F_c(R) φ_c(m) φ_c(n) / (φ_c(1) φ_c(−1)),
    F_c(R) = d_c Σ_X E_cX (χ̂_X(R) − 1),   φ_c(m) = Σ_X E_cX (Λ_X^m − 1).

* Channels are singles [μ,μ] and pairs {Z,Z′}. Λ_[Z,Z′] = A^{2|Z|} q^{2(κ_Z+κ_Z′)}.
* χ̂ are Koike composite characters from generic-A power sums, normalised by d_X.
* Presence is m_c(R) = Σ_γ c^R_{Zγ} c^R_{Z′γ} (LR coefficients). R_min, the order ≺
  and the supports B(c) follow §4.4 of the paper.
* Each row of E is solved mod p from the vanishing conditions
  Σ E_cX ψ_X(ν) = 0 for all ν where c is absent. A rank-deficient system has its
  free unknowns set to 0 (the paper shows F_c does not depend on them).

Convention: the paper's H(m,n) at (A, q) is our standard H of
`DoubleBraid(m, n)`, the 4-plat with continued fraction (−2m, −2n); no point map
is needed. Table knots whose 4-plat has two twist regions get this presentation
automatically.

Validation (tests/test_interpolation.py and the log in the commit message):
* the Appendix-B rows of E, the level-1 box sum, the rectangular products, and the
  closed formula (21) for H_[2,1](4_1) all hold exactly;
* against the Racah two-bridge data: all 29 R with |R| ≤ 6 × 12 double braids
  (4_1, 3_1 and its mirror, twist knots, (2,2), (2,−3), (−2,−3), …): 348/348 agree;
* |R| = 7 ([7], [6,1], [5,2], [4,3], [4,2,1], [3,3,1], [3,2,2], [5,1,1]), which
  has no Racah data: the trefoil equals Rosso–Jones, 4_1 is amphichiral, and the
  transposition symmetry holds;
* the over-determined interpolation systems are consistent (no violated extra
  conditions);
* end to end, 4_1 in [3,1] reconstructed from the formula is identical to the
  Racah reconstruction, and its q = 1 value equals H_[1]⁴.

Cost per point: < 0.1 s up to 5 boxes, 1–2 s for most 6- and 7-box R, ~19 s for
[3,2,1] (cold caches).

## 5. Arborescent / 2-bridge from exclusive Racah matrices (general form, M3)

Twist regions become *fingers* S T^a S† (parallel) or S̄ T̄^a S̄† (antiparallel),
glued along the Conway tree. 2-bridge knots are chains, so H is one matrix
product. Needs S, S̄ (your exclusive tables), eigenvalues on R⊗R and R⊗R̄,
and composite representations. *Coverage:* 362 two-bridge knots, 721 further
Montesinos knots, and many of the remaining 1894 are arborescent (Conway
notation).
Ref: Mironov, Morozov, Morozov, Sleptsov, *Tabulating knot polynomials for
arborescent knots*; *Colored HOMFLY polynomials for the pretzel knots and
links*.

### 5a. Montesinos knots (implemented: `montesinos`, `methods/montesinos.py`)

N(T₁ + … + T_k) for rational tangles p_i/q_i (KnotInfo `montesinos_notation`).
Every tangle is built from the 0 or ∞ tangle by twists of the right (H) and
bottom (V) pairs; the orientation of every endpoint is tracked, so each twist
is parallel (T, channels Q ∈ R⊗R) or antiparallel (T̄, channels X ∈ R⊗R̄).
Basis changes: (anti, anti) S̄, (par, anti) mixed S, (anti, par) S⁻¹.
The sum is the block product C₁E⁻¹C₂E⁻¹… (E = 0-tangle), closed with
Σ_X d_X Tr: gauge covariant, no Gram matrices.

Data (`scripts/import_racah_montesinos.py` → `data/racah/montesinos/`):
* family P, rectangular R (|R| ≤ 6): S̄, S, V = S⁻¹, T, T̄;
* family H: [2,1], [3,1], [5,1], [4,1,1], [3,3], [2,2,2], [3,1³], [2,1⁴];
  satisfies T̄⁻¹S̄T̄⁻¹ = S⁻¹TS;
* family G: [4,2], [2,2,1,1], [3,2,1] from the v1.1 mixed S only; the
  published S̄ is in a vacuum-dual gauge (rows and columns normalised
  differently), so S̄ is rebuilt as T̄ S⁻¹TS T̄ in the gauge of S's columns;
* [2,1,1] from [3,1] by H_{Rᵀ}(A, q) = H_R(A, −1/q).

Chirality: the fractions' sign is fixed by H_[1] against KnotInfo; 12 knots
with mirror-symmetric H_[1] are left out. The non-rectangular 5-box R are not
covered: only family-P data exist for them, and their R⊗R̄ has multiplicities.

Checks (tests/test_montesinos.py): all 721 Montesinos knots against KnotInfo
in [1]; colored values equal the 3-strand Racah methods for P, H and G reps;
the tangle order is irrelevant; transposition.

## 6. Double-braid knots in rectangular R (specified, M6)

For R = [r^s], R⊗R̄ is multiplicity-free and labelled by sub-diagrams of R.
This gives an evolution formula
H_R(m,n) = Σ_{X,Y⊆R} C_{XY}(A,q) Λ_X^m Λ_Y^n, which is symbolic in (m, n)
and has a factorised differential expansion.
Ref: A. Morozov, *Factorization of differential expansion for antiparallel
double-braid knots*; Ya. Kononov, A. Morozov, *On rectangular HOMFLY for
twist knots*.

## 7. Racah matrix sources

| Source | Status |
|---|---|
| User tables (inclusive/exclusive, ≤ 6 boxes) → `RacahStore` | loader ✅, adapter ⏳ M2 |
| Closed-form fundamental (Young orthogonal form) | ✅ |
| Eigenvalue hypothesis (Itoyama–Mironov–Morozov–Morozov) | 2×2 ✅; 3×3–5×5 ⏳ M4 |
| Yang–Baxter solve at each GF(p) point (size-agnostic) | ⏳ M4 |
| Highest-weight method in U_q(sl_N), modular, rational gauge | ⏳ M5 |

## References

* M. Rosso, V. Jones, *On the invariants of torus knots derived from quantum groups* (1993).
* X.-S. Lin, H. Zheng, *On the Hecke algebras and the colored HOMFLY polynomial* (2010).
* A. Mironov, A. Morozov, An. Morozov, *Character expansion for HOMFLY polynomials* I–III.
* H. Itoyama, A. Mironov, A. Morozov, An. Morozov, *Eigenvalue hypothesis for Racah matrices…* (2012).
* A. Mironov, A. Morozov, An. Morozov, A. Sleptsov, *Tabulating knot polynomials for arborescent knots*;
  *Colored knot polynomials: HOMFLY in representation [2,1]*.
* Sh. Shakirov, A. Sleptsov, *Quantum Racah matrices and 3-strand braids in irreps R with |R| = 4*.
* P. S. Wang, rational reconstruction; M. Monagan, *Maximal quotient rational reconstruction* (2004).
* R. Zippel, sparse interpolation; M. Ben-Or, P. Tiwari (1988).
