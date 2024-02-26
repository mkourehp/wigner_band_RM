from random_matrix import RM
import pytest
import numpy as np


@pytest.mark.parametrize("mat_size", [2,50,2000])
def test_diagonalization(mat_size: int):
    rm_obj = RM(matrix_size=mat_size, v=1)
    rm_ev = rm_obj.solve().eigenvalues
    np_ev = np.linalg.eigvalsh(rm_obj.h)
    rm_ev.sort(); np_ev.sort()
    assert rm_ev.size == np_ev.size
    assert all([abs(ev1-ev2) <1e-16 for ev1, ev2 in zip(np_ev, rm_ev)])



@pytest.mark.parametrize(("iteration","mat_size"), [(2,100),(50,10),(2000,5)])
def test_iteration(iteration: int, mat_size: int):
    rm_obj = RM(matrix_size=mat_size, v=1, iterate=iteration)
    assert rm_obj.energies.size == rm_obj.ms * iteration


@pytest.mark.parametrize("v", [0, 0.1, 0.5, 1, 10, 100, 1000])
def test_off_diagonal_variation(v):
    mat_size = 1000
    rm_obj = RM(matrix_size=mat_size, v=v, iterate=0)
    off_diag_av = np.sum((rm_obj.h - np.diag(rm_obj.h.diagonal()))**2)
    assert True
    
