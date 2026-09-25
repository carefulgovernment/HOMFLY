"""Racah providers: the interface between the RT braid engine and the sources
of Racah matrices (user-supplied tables, closed formulas, eigenvalue
hypothesis, highest-weight computation, ...).

A provider evaluates everything directly in the target field F at the point
(A, q) so that engines never see symbolic data.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class RacahProvider(ABC):
    """All methods take natural-convention (A, q) already mapped by the engine."""

    @abstractmethod
    def decompose(self, Y, R):
        """Irreducible summands of Y ⊗ R as a list of labels (Z, mult_index)."""

    @abstractmethod
    def eigenvalues(self, R, F, A, q):
        """{(X, mult_index): lambda_X} for R ⊗ R in the natural convention,
        lambda_X = eps_X q^{kappa_X - 2 kappa_R} (framing handled by engine)."""

    @abstractmethod
    def inclusive(self, Y, R, Z, F, A, q):
        """(row_labels, col_labels, U, U_inv) for U[Y, R, R -> Z].

        rows: labels Y' with Y' ∈ Y⊗R and Z ∈ Y'⊗R (same labels as decompose);
        cols: labels X of R⊗R (same labels as eigenvalues).
        U_inv may be None, in which case the engine inverts U numerically.
        """

    def exclusive(self, R, F, A, q):
        """(S, Sbar) exclusive matrices -- needed by arborescent methods."""
        raise NotImplementedError
