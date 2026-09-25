# Racah matrix tables

Put precomputed Racah matrices here in the canonical JSON format described in
[`docs/RACAH_FORMAT.md`](../../docs/RACAH_FORMAT.md):

```
data/racah/
  eigenvalues/<R>.json            e.g. 21.json
  inclusive/<Y>_<R>_<Z>.json      e.g. 2_2_42.json  = U[[2],[2],[2] -> [4,2]]
  exclusive/<R>.json              S and S̄ for R, e.g. 22.json
```

Raw files in other formats (Mathematica, Maple, text) go into `data/racah/raw/`
and are converted once by `homfly.racah.adapters`.
