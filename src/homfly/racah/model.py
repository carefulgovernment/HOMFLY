"""Data model for Racah (6j) matrices.

Terminology (Mironov--Morozov--Morozov et al.):

* INCLUSIVE Racah matrix  U[Y, R, R -> Z]: maps the basis of (Y ⊗ R) ⊗ R
  (labelled by the intermediate Y' ∈ Y ⊗ R) to the basis of Y ⊗ (R ⊗ R)
  (labelled by X ∈ R ⊗ R).  For 3-strand braids Y = R; m-strand braids need
  Y running over all irreps in R^{⊗(k-1)}.  The generator sigma_k acting on
  the path (…, Y_{k-1}, Y_k, Y_{k+1}, …) is the block  U diag(lambda_X) U^{-1}.
* EXCLUSIVE Racah matrices S = U[R, R, R̄ -> R] and S̄ = U[R, R̄, R -> R]:
  the ingredients of the arborescent (fingers/propagators) calculus used for
  2-bridge, pretzel, double-braid and general arborescent knots.  Their
  rows/columns are labelled by X ∈ R ⊗ R (Young diagrams) and by composite
  representations in R ⊗ R̄.

Entries are elements of Q(A, q)(sqrt(...)).  We store every entry as a sum of
terms  coef * sqrt(radicand)  with coef, radicand rational functions of (A, q)
given as (numerator, denominator) Laurent polynomials.  This covers all
multiplicity-free Racah matrices (single term per entry) as well as blocks
with multiplicities (several terms).  See docs/RACAH_FORMAT.md for the file
format and for strategies to avoid square roots in modular arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..algebra.laurent import Laurent


@dataclass
class RationalFunction:
    num: Laurent
    den: Laurent

    def evaluate(self, A, q, one):
        d = self.den.evaluate([A, q], one=one)
        return self.num.evaluate([A, q], one=one) / d


@dataclass
class Term:
    coef: RationalFunction
    radicand: RationalFunction | None = None  # None means sqrt(1)


@dataclass
class Entry:
    terms: list = field(default_factory=list)

    def evaluate(self, F, A, q):
        s = F.zero
        for t in self.terms:
            v = t.coef.evaluate(A, q, F.one)
            if t.radicand is not None:
                v = v * F.sqrt(t.radicand.evaluate(A, q, F.one))
            s = s + v
        return s

    def is_rational(self):
        return all(t.radicand is None for t in self.terms)


@dataclass
class RacahMatrix:
    kind: str                 # "inclusive" | "exclusive-S" | "exclusive-Sbar"
    reps: tuple               # (Y, R, R', Z) as partition tuples / composite labels
    rows: list                # labels of intermediate reps (with multiplicity index)
    cols: list
    entries: list             # list of rows of Entry
    convention: str = "MMM"   # declared normalisation, converted in racah.store

    def evaluate(self, F, A, q):
        return [[e.evaluate(F, A, q) for e in row] for row in self.entries]

    @property
    def shape(self):
        return (len(self.rows), len(self.cols))
