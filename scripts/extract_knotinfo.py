"""Regenerate src/homfly/knots/data/knotinfo_upto12.csv from the
``database_knotinfo`` package (pip install database_knotinfo)."""
import csv
import os

from database_knotinfo import link_list

COLS = ["name", "crossing_number", "alternating", "braid_index", "braid_notation",
        "bridge_index", "two_bridge_notation", "conway_notation", "montesinos_notation",
        "pd_notation", "symmetry_type", "homfly_polynomial"]
OUT = os.path.join(os.path.dirname(__file__), "..", "src", "homfly", "knots", "data",
                   "knotinfo_upto12.csv")

if __name__ == "__main__":
    rows = [r for r in link_list()[1:]
            if r["crossing_number"].isdigit() and int(r["crossing_number"]) <= 12]
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, delimiter="|")
        w.writerow(COLS)
        for r in rows:
            w.writerow([r[c] for c in COLS])
    print("wrote %d knots to %s" % (len(rows), OUT))
