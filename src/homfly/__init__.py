"""homfly -- colored HOMFLY-PT polynomials of knots.

Layers (bottom-up; see docs/ARCHITECTURE.md):
  algebra/         fields (QQ, GF(p), CC), Laurent polynomials, linear algebra
  reconstruction/  interpolation, CRT, rational reconstruction, pipeline
  reps/            Young diagrams, characters, Adams operations, LR, dim_q
  hecke/           Hecke algebra seminormal representations
  racah/           Racah matrices: data model, store, providers, checks
  knots/           braids, knot table (<= 12 crossings), families
  methods/         evaluation engines (black boxes over a field)
  io/              export and result database
"""
from .algebra.laurent import Laurent, parse_laurent  # noqa: F401
from .compute import homfly, resolve_knot  # noqa: F401
from .knots.braid import Braid  # noqa: F401
from .knots.families import TorusKnot, TwoBridge, DoubleBraid  # noqa: F401

__version__ = "0.1.0"
