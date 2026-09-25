"""End-to-end test of the canonical Racah JSON format: fundamental 3-strand
matrices written as files, read by RacahStore, fed into the RT engine."""
import json
import random

from homfly.algebra.fields import GF, NoSquareRoot
from homfly.knots.table import knot
from homfly.methods import HeckeFundamental, RTBraid
from homfly.racah.store import RacahStore

U21 = {
    "kind": "inclusive", "reps": [[1], [1], [1], [2, 1]], "convention": "natural",
    "rows": [[[2], 0], [[1, 1], 0]], "cols": [[[2], 0], [[1, 1], 0]],
    "entries": [
        [[{"coef": "q/(q^2+1)"}], [{"coef": "1", "sqrt": "(q^4+q^2+1)/(q^4+2*q^2+1)"}]],
        [[{"coef": "1", "sqrt": "(q^4+q^2+1)/(q^4+2*q^2+1)"}], [{"coef": "-q/(q^2+1)"}]],
    ],
}


def one_by_one(Z, X):
    return {"kind": "inclusive", "reps": [[1], [1], [1], Z], "rows": [[X, 0]],
            "cols": [[X, 0]], "entries": [[1]]}


def test_store_roundtrip(tmp_path):
    (tmp_path / "inclusive").mkdir()
    (tmp_path / "eigenvalues").mkdir()
    (tmp_path / "inclusive" / "1_1_21.json").write_text(json.dumps(U21))
    (tmp_path / "inclusive" / "1_1_3.json").write_text(json.dumps(one_by_one([3], [2])))
    (tmp_path / "inclusive" / "1_1_111.json").write_text(json.dumps(one_by_one([1, 1, 1], [1, 1])))
    (tmp_path / "eigenvalues" / "1.json").write_text(json.dumps(
        {"R": [1], "eigenvalues": [{"X": [2], "sign": 1}, {"X": [1, 1], "sign": -1}]}))
    store = RacahStore(str(tmp_path))
    assert store.available()["inclusive"] == ["1_1_111", "1_1_21", "1_1_3"]
    rt, h = RTBraid(store, max_strands=3), HeckeFundamental()
    F = GF(1000000007)
    rng = random.Random(2)
    checked = 0
    while checked < 3:
        A, q = F.random_element(rng), F.random_element(rng)
        for name in ["3_1", "4_1", "5_2", "8_20"]:
            b = knot(name).braid
            try:
                v = rt.evaluate(b, (1,), F, A, q)
            except NoSquareRoot:
                break
            assert v == h.evaluate(b, (1,), F, A, q), name
        else:
            checked += 1
