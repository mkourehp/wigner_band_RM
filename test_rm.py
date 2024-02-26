from random_matrix import RM
import pytest
import numpy as np


@pytest.mark.parametrize("mat_size", [2,50,100,1000])
def test_diagonalization(mat_size: int):
    rm_obj = RM(matrix_size=mat_size, v=1, band=mat_size)
    rm_ev = rm_obj.solve().eigenvalues
    np_ev = np.linalg.eigvalsh(rm_obj.h, "U")
    rm_ev.sort(); np_ev.sort()
    assert rm_ev.size == np_ev.size
    assert np.mean([ev1-ev2 for ev1, ev2 in zip(np_ev, rm_ev)]) < 1e-14



@pytest.mark.parametrize(("iteration","mat_size"), [(2,100),(50,10),(2000,5)])
def test_iteration(iteration: int, mat_size: int):
    rm_obj = RM(matrix_size=mat_size, v=1, iterate=iteration, band=mat_size)
    assert rm_obj.energies.size == rm_obj.size * iteration


def test_off_diagonal_variation():
    off_diag_av, mat_size = [], 500
    for v in np.linspace(0.1,10,10):
        rm_obj = RM(matrix_size=mat_size, v=v, iterate=0, band=mat_size)
        off_diag_av += [np.mean((rm_obj.h - np.diag(rm_obj.h.diagonal()))**2) / (v**2)]
        stopme=1
        # off_diag_av is the average of off diagonal elements, devided by v**2, which is independent of v
    assert np.var(off_diag_av) < 1e-5
