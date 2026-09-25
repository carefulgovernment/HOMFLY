"""Knot table (Rolfsen up to 10 crossings, Hoste--Thistlethwaite 11/12).

Data file ``data/knotinfo_upto12.csv`` is extracted from KnotInfo
(C. Livingston and A. H. Moore, knotinfo.org, via the ``database_knotinfo``
package, see ``scripts/extract_knotinfo.py``).  Columns:
name | crossing_number | alternating | braid_index | braid_notation |
bridge_index | two_bridge_notation | conway_notation | montesinos_notation |
pd_notation | symmetry_type | homfly_polynomial

The KnotInfo HOMFLY is P(v, z) with v^{-1} P(L+) - v P(L-) = z P(L0);
it is converted to our (A, q) by  A = v,  z = q - 1/q  (docs/CONVENTIONS.md).
"""
from __future__ import annotations

import ast
import csv
import os
from dataclasses import dataclass
from functools import lru_cache

from ..algebra.laurent import Laurent, parse_laurent, AQ
from .braid import Braid

DATA = os.path.join(os.path.dirname(__file__), "data", "knotinfo_upto12.csv")


@dataclass
class KnotRecord:
    name: str
    crossings: int
    alternating: bool
    braid_index: int
    braid: Braid | None
    bridge_index: int
    two_bridge: tuple | None
    conway: str
    montesinos: str
    pd: list | None
    symmetry: str
    homfly_knotinfo: str

    def homfly_reference(self):
        """Fundamental reduced HOMFLY in (A, q) from KnotInfo."""
        return knotinfo_homfly_to_Aq(self.homfly_knotinfo)

    @property
    def is_two_bridge(self):
        return self.two_bridge is not None

    @property
    def is_montesinos(self):
        return self.montesinos.startswith("K(")


def knotinfo_homfly_to_Aq(s):
    if not s:
        return Laurent.const(1, AQ)
    P = parse_laurent(s, vars=("v", "z"))
    A = Laurent.var("A", AQ)
    q = Laurent.var("q", AQ)
    return P.subs({"v": A, "z": q - q ** -1}, new_vars=AQ).normalize_integers()


def _parse_list(s):
    s = s.strip()
    if not s or s.startswith("does not"):
        return None
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None


@lru_cache(maxsize=1)
def load_table():
    out = {}
    with open(DATA, newline="") as f:
        for row in csv.DictReader(f, delimiter="|"):
            w = _parse_list(row["braid_notation"])
            if isinstance(w, list) and w and isinstance(w[0], list):
                w = w[0]  # some entries list several braids; take the first
            bi = int(row["braid_index"]) if row["braid_index"].isdigit() else 1
            braid = Braid(max(bi, max(abs(a) for a in w) + 1), tuple(w)) if w else None
            tb = _parse_list(row["two_bridge_notation"])
            out[row["name"]] = KnotRecord(
                name=row["name"],
                crossings=int(row["crossing_number"]),
                alternating=row["alternating"] == "Y",
                braid_index=bi,
                braid=braid,
                bridge_index=int(row["bridge_index"]) if row["bridge_index"].isdigit() else 1,
                two_bridge=tuple(tb) if tb else None,
                conway=row["conway_notation"],
                montesinos=row["montesinos_notation"],
                pd=_parse_list(row["pd_notation"]),
                symmetry=row["symmetry_type"],
                homfly_knotinfo=row["homfly_polynomial"],
            )
    return out


def knot(name):
    return load_table()[name]


def knots(max_crossings=12, braid_index=None, two_bridge=None, alternating=None):
    for rec in load_table().values():
        if rec.crossings > max_crossings:
            continue
        if braid_index is not None and rec.braid_index != braid_index:
            continue
        if two_bridge is not None and rec.is_two_bridge != two_bridge:
            continue
        if alternating is not None and rec.alternating != alternating:
            continue
        yield rec
