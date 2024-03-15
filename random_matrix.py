
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

    e_min = lambda self : np.min(self.results.energies)
    e_max = lambda self : np.max(self.results.energies)

    

    def _setH(self):
        self.h = sum(
            [(self.params.v if i!=0 else 1) * np.diag(np.random.randn(self.params.size-i), i) for i in range(1, self.params.band)]
              )
        self.h += np.diag(self.diagonal)
        if self.params.check:
            assert all([d1 == d2 for d1, d2 in zip(self.h.diagonal(), self.diagonal)])


    def _solve(self) -> Result:
        eigenvalues, self.h = np.linalg.eigh(a=self.h)
        if self.params.unfold:
            self.unfolded_energies = np.append(
                self.unfolded_energies,Unfolding(
                    self.results.energies,fit_poly_order=12,discard_percentage=10
                ).unfolded_energies
            )
        else:
            self.unfolded_energies = self.results.energies
        return Result(eigenvalues=eigenvalues, 
                      eigenvectors=self.h if self.params.eigfunctions else None)

    def _iterate(self):
        if self.params.iterate != 0:
            for i in tqdm(range(self.params.iterate)):
                self._setH()
                self.results[i] = self._solve()
        if self.params.iterate: self.results.energies = self.results.energies.reshape(self.params.iterate, self.params.size)

    def dos(self):
        plt.hist(self.results.energies.flatten(), bins="auto", density=True, histtype="bar", alpha=0.7)
        plt.show()


    def ldos(self, ratio: float = None, 
             energy: float = None, 
             s: int = 1,
             **kwargs):
        # Test this method
        if ratio:
            assert 0<=ratio<=1, f"ratio must be in [0,1], it is {ratio}"
        if energy:
           assert np.min(self.results.energies)<=energy<=np.max(self.results.energies),"Emin={:.3f}, Emax={:.3f}, you={:.3f}!".format(
                np.min(self.results.energies), np.max(self.results.energies), energy)
        else:
            energy = 0.0
        ldos = np.zeros((self.params.size, self.params.iterate))
        for i, row in enumerate(self.results.energies):
            E_min, E_max = min(row), max(row)
            if ratio:
                e = ratio * (E_max - E_min) + E_min # Get the energy in the pth portion of DoS
            else:
                e = energy
            index = np.argmin(abs(row - e))
            ldos[:,i] = [np.dot(self.h[:, i], self.h[:, index])**2
                         for i in range(self.params.size)]
        ldos = np.sum(ldos, axis=1) / self.params.iterate # average 
        x_axis = np.sum(self.results.energies, axis=0) / self.params.iterate
        plt.scatter(x_axis, ldos, **kwargs)



    @property
    def print_h(self):
        for row in self.h:
            [print("{:10.3}".format(r), end="\t") for r in row] ; print()

    @property
    def plot_h(self):
        plt.imshow(self.h)
        plt.colorbar()
        plt.show()

    def get_ipr(self, n):
        return np.sum([abs(c)**4 for c in self.h[:,n]]) / 3.


    @property
    def plot_ipr(self):
        plt.plot(range(self.params.size),[self.get_ipr(n) for n in range(self.params.size)], "o")
        plt.show()

       

    @property 
    def r(self):
        r = []   
        for row in self.results.energies:
            row.sort()
            s = np.diff(row)
            r += [s[i] / s[i-1] for i in range(s.size) if s[i-1] != 0]
        return r
    
    @property
    def r_til(self):
        return [min(rr,1./rr) for rr in self.r if rr != 0]



if __name__ == "__main__":
    # for v in np.linspace(1,5,3):
    # for v in np.linspace(0,1,3):
    params = Params(size=100, v=1, band=100, iterate=100, unfold=False)
    obj = RM(params=params)
    obj.ldos(energy=0.0, alpha=0.7, label=f"{1:.2f}", marker="o")
    plt.legend()
    plt.show()



# Calculate LDoS !!!