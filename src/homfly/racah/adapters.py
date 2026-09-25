"""Converters from external Racah-matrix formats to the canonical JSON.

TODO (ROADMAP M2): implement once a sample of the user's tables is uploaded.
Expected sources:
  * Mathematica dumps, e.g. entries like  Sqrt[q^4+1+q^-4]/(q^2+q^-2)
    or in quantum-number notation  Sqrt[[3]]/[2], with A = q^N;
  * the knotebook.org tables (inclusive [R,R,R->Q] and exclusive S, S̄).

Plan: a tokenizer that recognises  Sqrt[...], [n] (quantum numbers),
{A q^k}-style brackets, A and q powers; normalise every entry to a sum of
coef * sqrt(radicand) terms; record the source convention and the basis
ordering; emit canonical JSON; run racah.checks on the result.
"""


def convert_mathematica(path, out_dir):  # pragma: no cover - placeholder
    raise NotImplementedError("send a sample file; see docs/RACAH_FORMAT.md")
