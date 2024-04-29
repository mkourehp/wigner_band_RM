import numpy as np
import utils
from typing import List
from models.models import Params, Result
import matplotlib.pyplot as plt
from src.time_evolution import TimeEvolution

class PhysicalFunctions(TimeEvolution):
    def __init__(self, params: Params, results: List[Result]) -> None:
        self.p: Params = params
        self.res: List[Result] = results
   
    e_min: float = property((lambda self : np.min([r.eigenvalues for r in self.res])))
    e_max: float = property((lambda self : np.max([r.eigenvalues for r in self.res])))
    energies: np.array = property(lambda self : np.array([r.eigenvalues for r in self.res]))
    eigfuncs: np.array = property(lambda self : np.array([r.eigenvectors for r in self.res]))
    coef = lambda i, j, evecs: evecs[i, j]

    def get_vector_from_eigenvalue(self, evecs: np.array, value: float=0.0):
        """returns a vector, with energy 'value' in non-interacting basis"""
        indx = np.argmin(np.abs(self.p.diagonal-value))
        return evecs[indx, :]

    def entropy(
            self, 
            res: Result,
            psi0: np.array,
            t0:float,
            tf:float,
            nstep:int = None) -> np.array:
        nstep = nstep or 100
        t_array = np.linspace(t0, tf, nstep)
        s_t = np.zeros(nstep, dtype=float)
        for i, t in enumerate(t_array):
            psi_t = self.get_psi_t(t=t, psi_0=psi0, eigenvalues=res.eigenvalues)
            s_t[i] = self._entropy(psi=psi_t, psi0=psi0, evec=res.eigenvectors)
        return s_t

    def _entropy(self, psi, psi0, evec):
        w0 =np.abs(np.sum(psi * psi0))**2
        wf = np.abs([np.dot(psi, e) for e in evec])**2
        return -w0 * np.log(w0) - np.sum([w * np.log(w) for w in wf if w])
    
    def ldos(self, ratio: float = None, 
             energy: float = None) -> dict:
        if ratio:
            assert 0<=ratio<=1, f"ratio must be in [0,1], it is {ratio}"
        if energy:
           assert np.min(self.energies)<=energy<=np.max(self.energies),"Emin={:.3f}, Emax={:.3f}, you={:.3f}!".format(
                np.min(self.energies), np.max(self.energies), energy)
        else:
            energy = 0.0
        ldos = np.zeros((self.p.iterate, self.p.size))
        for i, row in enumerate(self.res):
            E_min, E_max = min(row.eigenvalues), max(row.eigenvalues)
            if ratio:
                e = ratio * (E_max - E_min) + E_min # Get the energy in the pth portion of DoS
            else:
                e = energy
            index = np.argmin(abs(row.eigenvalues - e))
            ldos[i, :] = self.res[i].eigenvectors[index, :] ** 2
        ldos = np.sum(ldos, axis=0) / self.p.iterate # average 
        return {"energies": np.sum(self.energies, axis=0) / self.p.iterate,
                "ldos": ldos}


    def ipr(self, psi: np.array, evecs: np.array):
        ipr = np.array([])
        for k in range(self.p.size):
            ipr = np.append(ipr, abs(sum(psi[k]*evecs[k, :]))**4)
        return sum(ipr)

    
    def pr(self, psi: np.array, evecs: np.array):
        return 1./self.ipr(psi=psi, evecs=evecs)


    @property
    def consec_level_spacing(self):
        level_spacing = [np.diff(e.eigenvalues) for e in self.res]
        r = np.array([])
        for s in level_spacing:
            r = np.append(r, [min(s[i], s[i+1])/ max(s[i], s[i+1]) 
                      for i in range(s.size-1)])
        return r