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
    off_diag_av_per_v2, mat_size = [], 100
    for v in np.linspace(0.1,10,10):
        for _ in range(10): # average over 10 * (100*100) matrix 
            rm_obj = RM(matrix_size=mat_size, v=v, iterate=0, band=mat_size)
            off_diag_av = np.mean((rm_obj.h - np.diag(rm_obj.h.diagonal()))**2)
            off_diag_av_per_v2 += [off_diag_av / v**2] 
    assert np.var(off_diag_av_per_v2) < 1e-2


def test_localization_length():
    mat_size = 10
    for band in range(1, mat_size-1):
        rm_obj = RM(matrix_size=mat_size, v=1, iterate=0, band=band)
        assert np.sum([np.sum(rm_obj.h.diagonal(i)) for i in range(band+1, mat_size)]) == 0
        assert np.sum([np.sum(rm_obj.h.diagonal(i)) for i in range(band)]) != 0

