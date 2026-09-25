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
| `sbar_<R>.json.gz` | family G: S̄ (vacuum-dual gauge), framing-free T̄² | [4,2], [2,2,1,1] |
| `inclusive_6`, `inclusive_111111` | family P generator export (3-strand) | [6], [1⁶] |

## `large/` (not committed): A-independent U_Q, families G/F

`uform_<R>.npz` for [4,2], [2,2,1,1] (16 MB each), and [3,2,1] (161 MB), built by
`scripts/import_racah_uform.py` (or `scripts/fetch_racah_release.py`).

Entries are `[num, den]` with `num`, `den` lists of `[i, j, c]` = c·A^i·q^j.
Format details: [`docs/RACAH_FORMAT.md`](../../docs/RACAH_FORMAT.md).

## Not yet imported (see ROADMAP M2b)

Not imported: the exclusive S̄[3,2,1] (family C). The release has no 3-strand
blocks for the family-H representations.

The canonical JSON described in `docs/RACAH_FORMAT.md` (`inclusive/`,
`exclusive/`, `eigenvalues/` subdirectories) remains the input format of the
general m-strand engine (`methods/rt_braid.py`).
