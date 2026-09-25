"""Sparse multivariate Laurent polynomials.

``Laurent`` stores ``{exponent_tuple: coefficient}``.  Coefficients are Python
ints / Fractions (or field elements for modular images).  The class is the
exchange format of the package: every final answer is a ``Laurent`` in the
variables ``("A", "q")``.

This is intentionally a thin, dependency-free implementation.  For heavy
symbolic work (e.g. manipulating Racah matrices with thousands of terms) an
adapter to python-flint (``fmpz_mpoly``) or Symengine should be added behind
the same interface -- see docs/ARCHITECTURE.md.
"""
from __future__ import annotations

import re
from fractions import Fraction

AQ = ("A", "q")


class Laurent:
    __slots__ = ("vars", "terms")

    def __init__(self, terms=None, vars=AQ):
        self.vars = tuple(vars)
        self.terms = {}
        if terms:
            for e, c in terms.items():
                if c != 0:
                    self.terms[tuple(e)] = c

    # -- constructors ------------------------------------------------------
    @classmethod
    def const(cls, c, vars=AQ):
        return cls({(0,) * len(vars): c}, vars)

    @classmethod
    def monomial(cls, exps, c=1, vars=AQ):
        return cls({tuple(exps): c}, vars)

    @classmethod
    def var(cls, name, vars=AQ):
        e = [0] * len(vars)
        e[vars.index(name)] = 1
        return cls({tuple(e): 1}, vars)

    # -- arithmetic --------------------------------------------------------
    def _lift(self, o):
        if isinstance(o, Laurent):
            if o.vars != self.vars:
                raise ValueError("variable mismatch %s vs %s" % (self.vars, o.vars))
            return o
        return Laurent.const(o, self.vars)

    def __add__(self, o):
        o = self._lift(o)
        t = dict(self.terms)
        for e, c in o.terms.items():
            t[e] = t.get(e, 0) + c
        return Laurent(t, self.vars)

    __radd__ = __add__

    def __neg__(self):
        return Laurent({e: -c for e, c in self.terms.items()}, self.vars)

    def __sub__(self, o):
        return self + (-self._lift(o))

    def __rsub__(self, o):
        return self._lift(o) - self

    def __mul__(self, o):
        o = self._lift(o)
        t = {}
        for e1, c1 in self.terms.items():
            for e2, c2 in o.terms.items():
                e = tuple(a + b for a, b in zip(e1, e2))
                t[e] = t.get(e, 0) + c1 * c2
        return Laurent(t, self.vars)

    __rmul__ = __mul__

    def __pow__(self, n):
        if n < 0:
            if len(self.terms) != 1:
                raise ValueError("only monomials can be inverted in a Laurent ring")
            (e, c), = self.terms.items()
            return Laurent({tuple(-x * (-n) for x in e): Fraction(1, 1) / c ** (-n)
                            if isinstance(c, int) else 1 / c ** (-n)}, self.vars)
        r, b = Laurent.const(1, self.vars), self
        while n:
            if n & 1:
                r = r * b
            b = b * b
            n >>= 1
        return r

    def __truediv__(self, o):
        """Exact division by a monomial or a scalar only."""
        if isinstance(o, Laurent):
            return self * o ** -1
        return Laurent({e: Fraction(c) / o if isinstance(c, int) else c / o
                        for e, c in self.terms.items()}, self.vars)

    def __eq__(self, o):
        try:
            o = self._lift(o)
        except ValueError:
            return False
        return self.terms == o.terms

    def __hash__(self):
        return hash((self.vars, frozenset(self.terms.items())))

    def is_zero(self):
        return not self.terms

    # -- inspection --------------------------------------------------------
    def degree_range(self, var):
        i = self.vars.index(var)
        es = [e[i] for e in self.terms]
        return (min(es), max(es)) if es else (0, -1)

    def coefficients(self):
        return list(self.terms.values())

    def normalize_integers(self):
        """Convert integral Fractions to int (cosmetic)."""
        t = {}
        for e, c in self.terms.items():
            if isinstance(c, Fraction) and c.denominator == 1:
                c = c.numerator
            t[e] = c
        return Laurent(t, self.vars)

    # -- evaluation / substitution ----------------------------------------
    def evaluate(self, values, one=1):
        """Evaluate at ``values`` (sequence aligned with ``self.vars``); works
        for any ring supporting ``* + **`` with negative powers."""
        pw = [dict() for _ in self.vars]
        acc = one * 0
        for e, c in self.terms.items():
            m = one * c
            for i, k in enumerate(e):
                if k:
                    cache = pw[i]
                    if k not in cache:
                        cache[k] = values[i] ** k
                    m = m * cache[k]
            acc = acc + m
        return acc

    def subs(self, mapping, new_vars=None):
        """Substitute Laurent polynomials for variables, e.g.
        ``P.subs({'v': A, 'z': q - q**-1}, new_vars=AQ)``."""
        new_vars = tuple(new_vars or self.vars)
        vals = []
        for v in self.vars:
            if v in mapping:
                vals.append(mapping[v])
            else:
                vals.append(Laurent.var(v, new_vars))
        return self.evaluate(vals, one=Laurent.const(1, new_vars))

    def map_coefficients(self, f):
        return Laurent({e: f(c) for e, c in self.terms.items()}, self.vars)

    # -- printing ----------------------------------------------------------
    def __repr__(self):
        return "Laurent(%s)" % self.to_string()

    def to_string(self, mul="*", pow_="^"):
        if not self.terms:
            return "0"
        out = []
        for e in sorted(self.terms, reverse=True):
            c = self.terms[e]
            mono = []
            for v, k in zip(self.vars, e):
                if k == 1:
                    mono.append(v)
                elif k:
                    mono.append("%s%s%s" % (v, pow_, k if k > 0 else "(%d)" % k))
            m = mul.join(mono)
            s = str(c)
            if m:
                if c == 1:
                    s = m
                elif c == -1:
                    s = "-" + m
                else:
                    s = "%s%s%s" % (s if not isinstance(c, Fraction) or c.denominator == 1
                                    else "(%s)" % s, mul, m)
            out.append(s)
        return " + ".join(out).replace("+ -", "- ")

    __str__ = to_string


# ----------------------------------------------------------------------------
# Parser for polynomial strings (KnotInfo, Mathematica-like input files)
# ----------------------------------------------------------------------------
_TOKEN = re.compile(r"\s*(?:(\d+)|([A-Za-z_][A-Za-z_0-9]*)|(\*\*|[-+*/^()]))")


def parse_laurent(s, vars=AQ):
    """Parse strings like ``"(2*v^2-v^4)+(v^2)*z^2"`` or ``"q^(-2)*A**4"``.

    Supported: integers, variables from ``vars``, ``+ - * /``, ``^`` / ``**``
    with integer (possibly negative, possibly parenthesised) exponents, and
    implicit multiplication between adjacent factors.  Division is allowed by
    integers and monomials only.
    """
    toks = []
    pos = 0
    s = s.strip()
    while pos < len(s):
        m = _TOKEN.match(s, pos)
        if not m:
            raise ValueError("cannot parse %r at %d" % (s, pos))
        pos = m.end()
        num, name, op = m.groups()
        toks.append(("n", int(num)) if num else ("v", name) if name else ("o", op))
    toks.append(("e", None))
    i = [0]

    def peek():
        return toks[i[0]]

    def take():
        t = toks[i[0]]
        i[0] += 1
        return t

    def expr():
        sign = 1
        if peek() == ("o", "-"):
            take()
            sign = -1
        elif peek() == ("o", "+"):
            take()
        r = term() * sign
        while peek() in (("o", "+"), ("o", "-")):
            op = take()[1]
            t = term()
            r = r + t if op == "+" else r - t
        return r

    def term():
        r = power()
        while True:
            t = peek()
            if t in (("o", "*"),):
                take()
                r = r * power()
            elif t == ("o", "/"):
                take()
                r = r / power()
            elif t[0] in ("n", "v") or t == ("o", "("):
                r = r * power()  # implicit multiplication
            else:
                return r

    def power():
        b = atom()
        if peek() in (("o", "^"), ("o", "**")):
            take()
            e = exponent()
            b = b ** e
        return b

    def exponent():
        t = take()
        if t == ("o", "("):
            e = exponent()
            if take() != ("o", ")"):
                raise ValueError("missing ) in exponent")
            return e
        if t == ("o", "-"):
            return -exponent()
        if t[0] == "n":
            return t[1]
        raise ValueError("bad exponent in %r" % s)

    def atom():
        t = take()
        if t[0] == "n":
            return Laurent.const(t[1], vars)
        if t[0] == "v":
            if t[1] not in vars:
                raise ValueError("unknown variable %r (vars=%s)" % (t[1], vars))
            return Laurent.var(t[1], vars)
        if t == ("o", "("):
            r = expr()
            if take() != ("o", ")"):
                raise ValueError("missing )")
            return r
        if t == ("o", "-"):
            return -power()
        raise ValueError("unexpected token %r in %r" % (t, s))

    r = expr()
    if peek()[0] != "e":
        raise ValueError("trailing input in %r" % s)
    return r.normalize_integers()
