"""SQLite results database: one row per (knot, representation, method).

Tabulation runs (scripts/tabulate.py) write here; the table is the long-term
deliverable (Rolfsen/HT knots up to 12 crossings x representations).
"""
from __future__ import annotations

import sqlite3
import time

from .export import to_json, from_json

SCHEMA = """
CREATE TABLE IF NOT EXISTS homfly (
    knot TEXT NOT NULL,
    rep TEXT NOT NULL,
    method TEXT NOT NULL,
    poly TEXT NOT NULL,
    seconds REAL,
    created REAL,
    verified INTEGER,
    PRIMARY KEY (knot, rep)
)
"""


class ResultDB:
    def __init__(self, path):
        self.con = sqlite3.connect(path)
        self.con.execute(SCHEMA)

    def put(self, knot, rep, method, H, seconds=None, verified=True):
        self.con.execute("REPLACE INTO homfly VALUES (?,?,?,?,?,?,?)",
                         (knot, ",".join(map(str, rep)), method, to_json(H), seconds,
                          time.time(), int(bool(verified))))
        self.con.commit()

    def get(self, knot, rep):
        row = self.con.execute("SELECT poly FROM homfly WHERE knot=? AND rep=?",
                               (knot, ",".join(map(str, rep)))).fetchone()
        return from_json(row[0]) if row else None

    def has(self, knot, rep):
        return self.get(knot, rep) is not None
