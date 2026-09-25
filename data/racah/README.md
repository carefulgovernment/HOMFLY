# Racah matrix tables

## `portable/`: family P from release v1.0.0 (imported)

Converted from `racah_matrices_upto6_v1.0.zip`
(https://github.com/carefulgovernment/HOMFLY/releases/tag/v1.0.0,
SHA-256 `a0aa3030…c2a6`) by `scripts/fetch_racah_release.py`:

| file | content | representations |
|---|---|---|
| `inclusive_<R>.json.gz` | per Q ∈ R⊗R⊗R: R1, R2, R1⁻¹, R2⁻¹ (framed, rational) | all 18 with \|R\| ≤ 5 |
| `exclusive_<R>.json.gz` | C = S T⁻¹ V, D̄² = T̄² (diagonal), vacuum index | \|R\| ≤ 5, [6], [1⁶] |
| `sbar_<R>.json.gz` | family H: S̄ (Y-gauge), framing-free T̄² | [6], [5,1], [4,1,1], [3,3], [2,2,2], [3,1³], [2,1⁴], [1⁶] |

Entries are `[num, den]` with `num`, `den` lists of `[i, j, c]` = c·A^i·q^j.
Format details: [`docs/RACAH_FORMAT.md`](../../docs/RACAH_FORMAT.md).

## Not yet imported (see ROADMAP M2b)

[4,2] and [2,2,1,1] (family G, `gtpath`) and [3,2,1] (families C/F) use
other gauges and vertical framing, and need their own adapters. The inclusive
3-strand blocks of family H are not present in the release.

The canonical JSON described in `docs/RACAH_FORMAT.md` (`inclusive/`,
`exclusive/`, `eigenvalues/` subdirectories) remains the input format of the
general m-strand engine (`methods/rt_braid.py`).
