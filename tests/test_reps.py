from homfly.algebra.fields import GF
from homfly.algebra.linalg import matmul
from homfly.hecke.seminormal import SeminormalModule
from homfly.reps.characters import adams, character, lr_product
from homfly.reps.partitions import conjugate, kappa, num_standard_tableaux, partitions, standard_tableaux


def test_partitions():
    assert len(partitions(6)) == 11
    assert conjugate((3, 1)) == (2, 1, 1)
    assert kappa((2,)) == 1 and kappa((1, 1)) == -1
    for la in partitions(6):
        assert len(standard_tableaux(la)) == num_standard_tableaux(la)


def test_characters():
    # chi^{[2,1]} on classes 1^3, 21, 3
    assert [character((2, 1), r) for r in [(1, 1, 1), (2, 1), (3,)]] == [2, 0, -1]
    # column orthogonality-ish: sum chi(1)^2 = n!
    assert sum(character(la, (1,) * 5) ** 2 for la in partitions(5)) == 120


def test_lr():
    assert lr_product((1,), (1,)) == {(2,): 1, (1, 1): 1}
    assert lr_product((2, 1), (2, 1)) == {(4, 2): 1, (4, 1, 1): 1, (3, 3): 1, (3, 2, 1): 2,
                                          (3, 1, 1, 1): 1, (2, 2, 2): 1, (2, 2, 1, 1): 1}


def test_adams():
    assert adams((1,), 3) == {(3,): 1, (2, 1): -1, (1, 1, 1): 1}
    assert adams((2,), 2) == {(4,): 1, (3, 1): -1, (2, 2): 1}


def test_seminormal_braid_relations():
    F = GF(1000003)
    q = F(98765)
    for Q in [(2, 1), (3, 2, 1), (4, 2)]:
        M = SeminormalModule(Q, q)
        for k in range(1, M.n - 1):
            a, b = M.matrix(k), M.matrix(k + 1)
            assert matmul(matmul(a, b), a) == matmul(matmul(b, a), b)
