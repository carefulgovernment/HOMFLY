"""Algebraic (arborescent) tangles from Conway notation.

A tangle is a tree of

    ("leaf", f)        rational tangle with fraction f (Fraction, or None for ∞)
    ("H", [t1, ...])   horizontal sum t1 + t2 + ...       (left to right)
    ("V", [t1, ...])   vertical stack t1 * t2 * ...       (top to bottom)

and an algebraic knot is the numerator closure N(t).  Conway's operations:

    x y   = x0 + y               (product; x0 = reflection of x in the NW-SE
                                  diagonal: leaf f -> 1/f, H <-> V)
    a,b,c = a0 + b0 + c0         (ramification; KnotInfo writes ';')
    x+ n  = in a ramification list, the element x followed by the integer
            tangle n (default 1; "x++" = "x+ 2"), "x- n" likewise with -n
    -x    = mirror image (all signs; within a run of integers a '-' carries
            over to the following integers: "-2 1 1" = (-2)(-1)(-1))

These conventions reproduce the KnotInfo Montesinos notation of 688 of the 689
Montesinos knots <= 12 crossings whose Conway notation is a flat list (the
exception, 12n_221, is listed there with a different number of tangles).
Polyhedral notations (containing '.', ':' or '*') are not algebraic.
"""
from __future__ import annotations

import re
from fractions import Fraction

INF = None


def leaf(f):
    return ("leaf", f)


def _inv(f):
    if f is INF:
        return Fraction(0)
    if f == 0:
        return INF
    return 1 / f


def flip(t):
    """Reflection in the NW-SE diagonal (Conway's t0)."""
    if t[0] == "leaf":
        return leaf(_inv(t[1]))
    return ("V" if t[0] == "H" else "H", [flip(c) for c in t[1]])


def mirror(t):
    if t[0] == "leaf":
        return leaf(INF if t[1] is INF else -t[1])
    return (t[0], [mirror(c) for c in t[1]])


def _is_int(t):
    return t[0] == "leaf" and t[1] is not INF and t[1].denominator == 1


def hsum(parts):
    """Horizontal sum, flattened.  An integer tangle is merged into an
    *adjacent* rational one (f + n is rational; moving it past a non-rational
    tangle would need a flype, which rotates that tangle); 0 tangles (the
    identity of the sum) are dropped."""
    flat = []
    for t in parts:
        flat.extend(t[1] if t[0] == "H" else [t])
    out = []
    for t in flat:
        if t[0] == "leaf" and t[1] == 0:
            continue
        prev = out[-1] if out else None
        rational = lambda u: u is not None and u[0] == "leaf" and u[1] is not INF
        if _is_int(t) and rational(prev):
            out[-1] = leaf(prev[1] + t[1])
        elif rational(t) and prev is not None and _is_int(prev):
            out[-1] = leaf(t[1] + prev[1])
        else:
            out.append(t)
        if out and out[-1][0] == "leaf" and out[-1][1] == 0:
            out.pop()
    if not out:
        return leaf(Fraction(0))
    return out[0] if len(out) == 1 else ("H", out)


def vsum(parts):
    """Vertical stack (the reflection of a horizontal sum)."""
    return flip(hsum([flip(t) for t in parts]))


def product(x, y):
    return hsum([flip(x), y])


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"\s*(\(|\)|;|,|[+-]|\d+)")


def _tokens(s):
    s = s.replace("[", "").replace("]", "").strip()     # (KnotInfo has stray brackets)
    spaced = re.search(r"\d\s+-?\d", s) is not None     # "2 1 1" vs compressed "211"
    out, pos = [], 0
    while pos < len(s):
        if s[pos].isspace():
            out.append(" ")
            while pos < len(s) and s[pos].isspace():
                pos += 1
            continue
        m = _TOKEN.match(s, pos)
        if not m:
            raise ValueError("cannot parse Conway notation %r at %d" % (s, pos))
        tok = m.group(1)
        if tok.isdigit() and not spaced:
            out.extend(tok)                 # compressed "211" = 2 1 1
        else:
            out.append(tok)
        pos = m.end()
    return out


class _Parser:
    def __init__(self, s):
        if re.search(r"[.:*]", s):
            raise ValueError("polyhedral Conway notation %r" % s)
        self.t = _tokens(s)
        self.i = 0

    def peek(self, skip_space=True):
        j = self.i
        while skip_space and j < len(self.t) and self.t[j] == " ":
            j += 1
        return self.t[j] if j < len(self.t) else None

    def take(self):
        while self.t[self.i] == " ":
            self.i += 1
        tok = self.t[self.i]
        self.i += 1
        return tok

    def parse(self):
        t = self.list_()
        if self.peek() is not None:
            raise ValueError("trailing input in Conway notation")
        return t

    def list_(self):
        elems = []
        while True:
            x, extra = self.elem()
            elems.append((x, extra))
            if self.peek() in (";", ","):
                self.take()
                continue
            break
        if len(elems) == 1 and elems[0][1] == 0:
            return elems[0][0]
        parts = [flip(x) for x, _ in elems]
        parts.append(leaf(Fraction(sum(e for _, e in elems))))
        return hsum(parts)

    def elem(self):
        x = self.prod()
        # suffix: signs, then an optional count
        signs = []
        while self.peek() in ("+", "-") and not self._sign_starts_factor():
            signs.append(self.take())
        extra = 0
        if signs:
            k = len(signs)
            if self.peek() is not None and self.peek().isdigit():
                k += int(self.take()) - 1
            extra = k if signs[0] == "+" else -k
        return x, extra

    def _sign_starts_factor(self):
        # "-(" is a mirrored factor, "-<digit>" a negative integer
        j = self.i
        while self.t[j] == " ":
            j += 1
        nxt = self.t[j + 1] if j + 1 < len(self.t) else None
        return self.t[j] == "-" and (nxt == "(" or (nxt is not None and nxt.isdigit()))

    def prod(self):
        x = None
        neg = False
        while True:
            tok = self.peek()
            if tok == "(" or (tok == "-" and self._sign_starts_factor()):
                mir = False
                if tok == "-":
                    self.take()
                    if self.peek(skip_space=False) == "(":
                        mir = True
                    else:                     # negative integer
                        neg = True
                        f = leaf(Fraction(-int(self.take())))
                        x = f if x is None else product(x, f)
                        continue
                if self.peek() == "(":
                    self.take()
                    y = self.list_()
                    if self.take() != ")":
                        raise ValueError("unbalanced parentheses")
                    if mir:
                        y = mirror(y)
                    neg = False
                    x = y if x is None else product(x, y)
                    continue
            if tok is not None and tok.isdigit():
                n = int(self.take())
                f = leaf(Fraction(-n if neg else n))
                x = f if x is None else product(x, f)
                continue
            break
        if x is None:
            raise ValueError("empty tangle")
        return x


def parse(s):
    """Tangle tree t of a Conway notation; the knot is N(t)."""
    return _Parser(s).parse()


def is_algebraic(s):
    return bool(s) and not re.search(r"[.:*]", s)


def depth(t):
    return 0 if t[0] == "leaf" else 1 + max(depth(c) for c in t[1])
