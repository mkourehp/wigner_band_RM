from random_matrix import RM
from models.models import Params
import pytest
import numpy as np
from conftest import get_params
import matplotlib.pyplot as plt

PRECISION = 1e-14


@pytest.mark.parametrize("size", range(2,100,10))
def test_h_structure(size):
    for band in range(size):
        rm_obj = RM(Params(size=size, band=band, v=1.0))
        h = rm_obj.init.h
        for b in range(1,size):
            if b<=band :assert all([True for p in h.diagonal(b) if p!=0.0])
            if b>band  :assert all([True for p in h.diagonal(b) if p==0.0])



@pytest.mark.parametrize("v", np.linspace(0,1,10))
def test_get_vector_from_energy(v: int):
    size = 200
    params = get_params(size=size, iterate=1, v=v, band=size-1)
    rm_obj = RM(params=params)
    result = rm_obj.results[0]
    for desired_energy in np.linspace(-20, 20):
        psi = rm_obj.pf.get_vector_from_energy(energy=desired_energy)[0]
        if v==0: assert any([l==1.0 for l in psi])
        if v!=0: assert not any([l==1.0 for l in psi])
        psi_energy = np.sum(abs(psi)**2 * rm_obj.results[0].eigenvalues)
        assert abs(psi_energy-desired_energy) < 1.0 # average level spacing



@pytest.mark.parametrize("size", [2,5,10,100])
def test_ldos(size):
    for v in [0.01, 1]:
        params = get_params(size=size,v=v, iterate=5,band=size-1)
        rm = RM(params=params)
        ldos = rm.pf.ldos(energy=0.0)
        assert abs(np.sum(ldos["ldos"]) -1) < PRECISION
        if v==0: assert any([l==1.0 for l in ldos["ldos"]])
        if v!=0: assert not any([l==1.0 for l in ldos["ldos"]])


def test_consec_level_spacing():
    size=50
    r = []
    for v in np.linspace(0, 1, 5):
        params = get_params(size=size,v=v, iterate=500,
                            band=size-1, fixed_diagonal=False,
                            h0=np.random.randn(size))
        rm = RM(params=params)
        r += [np.average(rm.pf.consec_level_spacing)]
    assert 0.38 - 0.01 < np.min(r) < 0.38 + 0.01
    assert 0.53 - 0.01 < np.max(r) < 0.53 + 0.01


@pytest.mark.parametrize("v", np.linspace(0.0,1,3))
def test_pr_and_ipr(v):
    size = 500
    t_array = np.linspace(0,1,500)
    params = get_params(size=size, band=size-1, iterate=5, v=v, fixed_diagonal=True)
    rm_obj = RM(params=params)
    psi0 = rm_obj.pf.get_vector_from_energy(energy=0.0)[0]
    iprs = [[] for _ in range(params.iterate)]
    prs = [[] for _ in range(params.iterate)]
    for i in range(params.iterate):
        ipr, pr = [], []
        for t in t_array:
            psi=rm_obj.pf.get_psi_t(t,psi0=psi0[i],
                eigenvalues=rm_obj.results[i].eigenvalues)
            ipr.append(rm_obj.pf.ipr(psi=psi, psi0=psi0))
            pr.append(rm_obj.pf.pr(psi=psi, psi0=psi0))
        iprs[i], prs[i] = ipr, pr            
        # [plt.plot(ipr) for ipr in iprs]
        [plt.plot(ipr) for ipr in iprs]
    plt.show()
    assert True # make some correlation here



@pytest.mark.parametrize("v", np.linspace(0.01,1,4))
def test_entropy(v):
    size = 20
    t_array = np.linspace(0,10,100)
    params = get_params(size=size, band=2, iterate=100, v=v, fixed_diagonal=True)
    rm_obj = RM(params=params)
    psi0 = rm_obj.pf.get_vector_from_energy(energy=0.0)[0]
    s = []
    for i in range(params.iterate):
        s_t = []
        for j, t in enumerate(t_array):
            psi=rm_obj.pf.get_psi_t(t,psi0=psi0,
                eigenvalues=rm_obj.results[i].eigenvalues)
            s_t.append(rm_obj.pf.entropy(psi=psi,psi0=psi0))
        plt.plot(t_array, s_t, ".-", c="C01",alpha=0.4)
        s.append(s_t)
    plt.plot(t_array, np.average(s, axis=0), c="C00")
    plt.title(f"v={v:.3f}")
    plt.show()



