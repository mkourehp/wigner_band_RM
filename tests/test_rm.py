from random_matrix import RM
from models.models import Params
import pytest
import numpy as np
from conftest import get_params

PRECISION = 1e-15

def test_get_vector_from_eigenvalue(mat_size: int):
    params = get_params(size=mat_size, iterate=1)
    rm_obj = RM(params=params)
    result = rm_obj.results[0]
    e1 = rm_obj.pf.get_vector_from_eigenvalue(evecs=result.eigenvectors, value=0.0)
    stopme=True

@pytest.mark.parametrize("size", [2,5,10,100])
def test_ldos(size):
    for v in [0.01, 1]:
        params = get_params(size=size,v=v, iterate=5,band=size-1)
        rm = RM(params=params)
        ldos = rm.pf.ldos(energy=0.0)
        assert abs(np.sum(ldos["ldos"]) -1) < PRECISION
        if v==0: assert any([l==1.0 for l in ldos["ldos"]])
        if v!=0: assert not any([l==1.0 for l in ldos["ldos"]])


@pytest.mark.parametrize("v", np.linspace(0.01, 5, 10))
def test_consec_level_spacing(v):
    size=10
    params = get_params(size=size,v=v, iterate=500,band=size-1)
    rm = RM(params=params)
    assert np.average(rm.pf.consec_level_spacing)

def test_pr_and_ipr():
    pass



def test_entropy():
    pass


