"""Reader for the data/homfly/<knot>.txt tables: H_R = N / den, evaluated mod p."""
import os
import re

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

def _terms(expr):
    """[(coeff, a, e)] of a Laurent polynomial written as  c*A^a*q^e + ..."""
    s = expr.replace(" ", "")
    toks, cur, depth = [], "", 0
    for ch in s:
        if ch in "+-" and depth == 0 and cur:
            toks.append(cur); cur = ""
        depth += (ch == "(") - (ch == ")")
        cur += ch
    toks.append(cur)
    out = []
    for tok in toks:
        sign = -1 if tok.startswith("-") else 1
        c, a, e = 1, 0, 0
        for f in tok.lstrip("+-").split("*"):
            if f.startswith("A"):
                a = int(f[2:].strip("()")) if "^" in f else 1
            elif f.startswith("q"):
                e = int(f[2:].strip("()")) if "^" in f else 1
            else:
                c = int(f)
        out.append((sign * c, a, e))
    return out


def _ev(terms, A, q, p):
    return sum(c * pow(A, a % (p - 1), p) * pow(q, e % (p - 1), p) for c, a, e in terms) % p


def _den(expr, A, q, p):
    v = 1
    for h, k in re.findall(r"\(q\^(\d+)\s*-\s*q\^\(-\d+\)\)(?:\^(\d+))?", expr):
        h = int(h); k = int(k or 1)
        v = v * pow((pow(q, h, p) - pow(q, (-h) % (p - 1), p)) % p, k, p) % p
    return v


def load(knot):
    """{R: (N terms, den string)} of data/homfly/<knot>.txt"""
    out = {}; R = None; den = None
    for line in open(os.path.join(DATA, "homfly", knot + ".txt")):
        line = line.strip()
        if line.startswith("R = "):
            R = tuple(int(x) for x in line[4:].strip("[]").split(",") if x.strip())
        elif line.startswith("den = "):
            den = line[6:]
        elif line.startswith("N = "):
            out[R] = (_terms(line[4:]), den)
        elif line.startswith("H = "):
            out[R] = (_terms(line[4:]), None)
    return out


def value(entry, A, q, p):
    N, den = entry
    v = _ev(N, A, q, p)
    return v if den is None else v * pow(_den(den, A, q, p), p - 2, p) % p
