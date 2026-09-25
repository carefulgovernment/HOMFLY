"""Export of results to computer-algebra formats and JSON."""
from __future__ import annotations

import json


def to_mathematica(H):
    return H.to_string(mul="*", pow_="^").replace("(", "(").replace("^(", "^(")


def to_maple(H):
    return H.to_string(mul="*", pow_="^")


def to_json(H):
    return json.dumps({"vars": list(H.vars),
                       "terms": [[list(e), str(c)] for e, c in sorted(H.terms.items())]})


def from_json(s):
    from fractions import Fraction
    from ..algebra.laurent import Laurent
    d = json.loads(s)
    return Laurent({tuple(e): Fraction(c) if "/" in c else int(c) for e, c in d["terms"]},
                   tuple(d["vars"]))
