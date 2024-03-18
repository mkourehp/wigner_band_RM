
from matplotlib import pyplot as plt
import numpy as np
import util
from typing import List, Optional
from dataclasses import dataclass
from tqdm import tqdm
from unfolding import Unfolding 
from models import Params, Result, Results


class RM:
    h = None
    unfolded_energies = np.array([])
    def __init__(self, params : Params
                 ) -> None:
        self.params = params
        self.results = Results([])
        self._initialize()

    def _initialize(self):
        assert self.params.size >= 2 , ValueError(f"Matrix size must be greater that 1. it is {self.params.size}")
        assert  self.params.band <= self.params.size , ValueError(
            f"Band must be greater that matrix size!  matrix_size: {self.params.size}, band: {self.params.band}")
        self.diagonal = np.sort(np.random.randn(self.params.size))
        self.results = np.zeros(self.params.iterate, dtype=Result)
        if self.params.iterate: self._iterate()

    e_min = property((lambda self : np.min([r.eigenvalues for r in self.results])))
    e_max = property((lambda self : np.max([r.eigenvalues for r in self.results])))
    energies = property(lambda self : np.array([r.eigenvalues for r in self.results]))
    eigfuncs = property(lambda self : np.array([r.eigenvectors for r in self.results]))
    
    
    @staticmethod
    def get_index_for_value(array: np.array, val: float) -> int:
        return np.argmin(abs(array- val))

    

    def _setH(self):
        self.h = sum(
            [(self.params.v if i!=0 else 1) * np.diag(np.random.randn(self.params.size-i), i) for i in range(1, self.params.band)]
              )
        self.h += np.diag(self.diagonal)
        if self.params.check:
            assert all([d1 == d2 for d1, d2 in zip(self.h.diagonal(), self.diagonal)])


    def _solve(self) -> Result:
        eigenvalues, self.h = np.linalg.eigh(a=self.h, UPLO="U")
        if self.params.unfold:
            self.unfolded_energies = np.append(
                self.unfolded_energies,Unfolding(
                    eigenvalues ,fit_poly_order=12,discard_percentage=10
                ).unfolded_energies
            )
        else:
            self.unfolded_energies = eigenvalues
        return Result(v=self.params.v, eigenvalues=eigenvalues, 
                      eigenvectors=self.h if self.params.eigfunctions else None)

    def _iterate(self):
        if self.params.iterate != 0:
            for i in tqdm(range(self.params.iterate)):
                self._setH()
                self.results[i] = self._solve()

    def dos(self, **kwargs):
        plt.hist(self.energies.flatten(), **kwargs)


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
        ldos = np.zeros((self.params.iterate, self.params.size))
        for i, row in enumerate(self.results):
            E_min, E_max = min(row.eigenvalues), max(row.eigenvalues)
            if ratio:
                e = ratio * (E_max - E_min) + E_min # Get the energy in the pth portion of DoS
            else:
                e = energy
            index = np.argmin(abs(row.eigenvalues - e))
            ldos[i, :] = self.results[i].eigenvectors[index, :] ** 2
        ldos = np.sum(ldos, axis=0) / self.params.iterate # average 
        x_axis = np.sum(self.energies, axis=0) / self.params.iterate
        plt.scatter(x_axis, ldos, **kwargs)



    @property
    def print_h(self):
        for row in self.h:
            [print("{:10.3}".format(r), end="\t") for r in row] ; print()

    @property
    def plot_h(self):
        plt.imshow(self.h)
        plt.colorbar()

    def get_ipr(self, n):
        return np.sum([abs(c)**4 for c in self.h[:,n]]) / 3.


    @property
    def plot_ipr(self):
        plt.plot(range(self.params.size),[self.get_ipr(n) for n in range(self.params.size)], "o")


    def pr_t(self, t_final: float, energy: float = 0.0):
        pr_t = np.zeros((self.params.iterate, self.params.size))
        for i, r in enumerate(self.results):
            index = self.get_index_for_value(r.eigenvalues, val=energy)
            pr_t[i, :] = np.sum([np.exp(complex(0,1)*r.eigenvalues[j]*t_final) * r.eigenvectors[index, j] for j in range(self.params.size)])
        return pr_t
       

    @property 
    def r(self):
        r = []   
        for row in self.energies:
            row.sort()
            s = np.diff(row)
            r += [s[i] / s[i-1] for i in range(s.size) if s[i-1] != 0]
        return r
    
    @property
    def r_til(self):
        return [min(rr,1./rr) for rr in self.r if rr != 0]  



if __name__ == "__main__":
    mat_size = 100
    # for v in np.linspace(0.,.2,5):
    for band in np.linspace(1,mat_size, 5):
        params = Params(size=mat_size, v=0.02, band=int(band), iterate=50, unfold=False, eigfunctions=True, ldos=True)
        obj = RM(params=params)
        # obj.dos(density=True, alpha=0.6, bins=20,histtype="step", label=f"{band}")
        # obj.ldos(energy=0.0, alpha=0.7, marker="o", label=f"{int(band):.2f}",)
        for t in np.linspace(0,1, 100):
            prt_t = obj.pr_t(energy=0.0, t_final=1.0)
            stopme=1
        
    plt.legend()
    plt.show()



# Calculate Dynamics of PR(t)