import numpy as np
import utils
from models.models import Params
import matplotlib.pyplot as plt
from src.time_evolution import TimeEvolution

class PhysicalFunctions(TimeEvolution):
    def __init__(self, params: Params) -> None:
        super().__init__(params)
        self.p = params
   
    e_min = property((lambda self : np.min([r.eigenvalues for r in self.results])))
    e_max = property((lambda self : np.max([r.eigenvalues for r in self.results])))
    energies = property(lambda self : np.array([r.eigenvalues for r in self.results]))
    eigfuncs = property(lambda self : np.array([r.eigenvectors for r in self.results]))

    def pr_t(self, t_final: float, energy: float = 0.0, steps: int = 100) -> np.array:
        ipr_t = np.zeros((self.p.iterate, steps))
        for ir, res in enumerate(self.results):
            c0_index = utils.get_index_for_value(res.eigenvalues, val=energy)
            c0 = res.eigenvectors[:, c0_index]
            ipr_t[ir, :] = np.abs([np.sum(self.get_psi_t(t=t, psi0=c0, energies=res.eigenvalues))
                  for t in np.linspace(0,t_final, steps)])**4
            # ipr_tmp = np.append(ipr_tmp, 1. / np.sum(ct)**4))
        ipr_t = np.sum(ipr_t, axis=0) / self.p.iterate
        return 1. / ipr_t




    def entropy_t(self, t_final: float, energy: float = 0.0, steps: int = 100) -> np.array:
        s_t = np.zeros((self.p.iterate, steps))
        for ir, res in enumerate(self.results):
            c0_index = utils.get_index_for_value(res.eigenvalues, val=energy)
            c0 = res.eigenvectors[:, c0_index]
            s_t[ir, :] = np.abs([np.sum(self.get_entropy_t(
                t=t, psi=self.get_psi_t(energies=res.eigenvalues, t=t, psi0=c0)))
                  for t in np.linspace(0,t_final, steps)])**4
        s_t = np.sum(s_t, axis=0) / self.p.iterate
        return s_t

    
    def ldos(self, ratio: float = None, 
             energy: float = None, 
             **kwargs):
        # Test this method
        if ratio:
            assert 0<=ratio<=1, f"ratio must be in [0,1], it is {ratio}"
        if energy:
           assert np.min(self.energies)<=energy<=np.max(self.energies),"Emin={:.3f}, Emax={:.3f}, you={:.3f}!".format(
                np.min(self.energies), np.max(self.energies), energy)
        else:
            energy = 0.0
        ldos = np.zeros((self.p.iterate, self.p.size))
        for i, row in enumerate(self.results):
            E_min, E_max = min(row.eigenvalues), max(row.eigenvalues)
            if ratio:
                e = ratio * (E_max - E_min) + E_min # Get the energy in the pth portion of DoS
            else:
                e = energy
            index = np.argmin(abs(row.eigenvalues - e))
            ldos[i, :] = self.results[i].eigenvectors[index, :] ** 2
        ldos = np.sum(ldos, axis=0) / self.p.iterate # average 
        x_axis = np.sum(self.energies, axis=0) / self.p.iterate
        plt.scatter(x_axis, ldos, **kwargs)


    def get_ipr(self, n):
        return np.sum([abs(c)**4 for c in self.h[:,n]]) / 3.


    @property
    def plot_ipr(self):
        plt.plot(range(self.p.size),[1/self.get_ipr(n) for n in range(self.p.size)], "o")
        
        
    @staticmethod
    def get_entropy_t(t: float, psi: np.array) -> np.array:
        return (psi**2) * np.log(psi**2)

    @staticmethod
    def _r(energies):
        en = np.sort(energies)
        s = np.diff(en)
        r = np.array([s[i] / s[i-1] for i in range(s.size) if s[i-1] != 0])
        return r
    
    @property
    def r_til(self) -> int:
        r = []
        for res in self.results:
            r += [np.average([min(rr,1./rr) for rr in self._r(res.eigenvalues)])]
        return np.average(r)


