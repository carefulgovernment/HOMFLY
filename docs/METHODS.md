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

## 5. Arborescent / 2-bridge from exclusive Racah matrices (specified, M3)

Twist regions become *fingers* S T^a S† (parallel) or S̄ T̄^a S̄† (antiparallel),
glued along the Conway tree. 2-bridge knots are chains, so H is one matrix
product. Needs S, S̄ (your exclusive tables), eigenvalues on R⊗R and R⊗R̄,
and composite representations. *Coverage:* 362 two-bridge knots, 721 further
Montesinos knots, and many of the remaining 1894 are arborescent (Conway
notation).
Ref: Mironov, Morozov, Morozov, Sleptsov, *Tabulating knot polynomials for
arborescent knots*; *Colored HOMFLY polynomials for the pretzel knots and
links*.

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
