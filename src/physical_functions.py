import numpy as np
import tools.utils as utils
from typing import List
from tools.models import Params, Result, EigFuncNotCalc
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

    def _gaussian_coef(self,x, E0, dE):
        if isinstance(x, float):
            return np.exp(-float(x-E0)**2/(2*dE**2)) / (dE*np.sqrt(2*np.pi))
        return np.array([np.exp(-float(xx-E0)**2/(2*dE**2)) / (dE*np.sqrt(2*np.pi)) for xx in x])


    def get_vector_from_energy(self, E0: float=0.0, dE: float=1.0):
        """returns a vector, with expectation energy 'E0' and variance 'dE'"""
        assert all(self.res[0].eigenvectors[0] != None), EigFuncNotCalc()
        psi0 =  np.array([self._gaussian_coef(r.eigenvalues, E0, dE) for r in self.res])
        psi0 = [utils.normalize(p) for p in psi0]
        return psi0
        



    def entropy(self, psi, psi0):
        w0 = np.abs(np.dot(psi0, psi))**2
        return - w0 * np.log(w0)

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
        

    def ipr(self, psi: np.array, psi0: np.array):
        return np.sum(np.abs(np.dot(psi, psi0))**4)
    
    
    def pr(self, psi: np.array, psi0: np.array):
        return 1./np.sum(np.abs(np.dot(psi, psi0))**4)
    

    @property
    def consec_level_spacing(self):
        level_spacing = [np.diff(e.eigenvalues) for e in self.res]
        r = np.array([])
        for s in level_spacing:
            r = np.append(r, [min(s[i], s[i+1])/ max(s[i], s[i+1]) 
                      for i in range(s.size-1)])
        return r