"""Braid words.  Letter ``k`` = sigma_k, ``-k`` = sigma_k^{-1}, 1-based."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Braid:
    strands: int
    word: tuple

    def __post_init__(self):
        object.__setattr__(self, "word", tuple(int(x) for x in self.word))
        for a in self.word:
            if a == 0 or abs(a) >= self.strands:
                raise ValueError("letter %d invalid on %d strands" % (a, self.strands))

    @classmethod
    def from_word(cls, word, strands=None):
        word = tuple(word)
        if strands is None:
            strands = max((abs(a) for a in word), default=0) + 1
        return cls(strands, word)

    @property
    def writhe(self):
        return sum(1 if a > 0 else -1 for a in self.word)

    def permutation(self):
        perm = list(range(self.strands))
        for a in self.word:
            k = abs(a) - 1
            perm[k], perm[k + 1] = perm[k + 1], perm[k]
        return perm

    def components(self):
        perm, seen, comps = self.permutation(), set(), 0
        for s in range(self.strands):
            if s not in seen:
                comps += 1
                while s not in seen:
                    seen.add(s)
                    s = perm[s]
        return comps

    def is_knot(self):
        return self.components() == 1

    def mirror(self):
        return Braid(self.strands, tuple(-a for a in self.word))

    def cable(self, r):
        """r-parallel cable (blackboard framing): each strand -> r strands."""
        out = []
        for a in self.word:
            i = abs(a)
            block = []
            base = (i - 1) * r  # 0-based index of first strand of bundle i
            for j in range(r):
                for t in range(r):
                    # sequence sigma_{ir+j}, sigma_{ir+j-1}, ..., sigma_{(i-1)r+1+j}
                    block.append(base + r + j - t)
            if a < 0:
                block = [-x for x in reversed(block)]
            out.extend(block)
        return Braid(self.strands * r, tuple(out))

    def __len__(self):
        return len(self.word)


def torus_braid(m, n):
    """T[m, n] as the closure of (sigma_1 ... sigma_{m-1})^n."""
    return Braid(m, tuple(list(range(1, m)) * n))
