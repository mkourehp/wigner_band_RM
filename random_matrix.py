
from matplotlib import pyplot as plt
import numpy as np
import utils
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
        self.diagonal = np.sort(np.random.randn(self.params.size)) if params.diagonal.size==0 else params.diagonal
        self.results = np.zeros(self.params.iterate, dtype=Result)
        if self.params.iterate: self._iterate()

    e_min = property((lambda self : np.min([r.eigenvalues for r in self.results])))
    e_max = property((lambda self : np.max([r.eigenvalues for r in self.results])))
    energies = property(lambda self : np.array([r.eigenvalues for r in self.results]))
    eigfuncs = property(lambda self : np.array([r.eigenvectors for r in self.results]))

    

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
        plt.plot(range(self.params.size),[1/self.get_ipr(n) for n in range(self.params.size)], "o")
        
        
    @staticmethod
    def get_entropy_t(t: float, psi: np.array) -> np.array:
        return (psi**2) * np.log(psi**2)

    @staticmethod
    def get_psi_t(t: float, psi0: np.array, energies: np.array) -> np.array:
        return np.exp(-1j*energies*t) * (psi0**2)

    def pr_t(self, t_final: float, energy: float = 0.0, steps: int = 100) -> np.array:
        ipr_t = np.zeros((self.params.iterate, steps))
        for ir, res in enumerate(self.results):
            c0_index = utils.get_index_for_value(res.eigenvalues, val=energy)
            c0 = res.eigenvectors[:, c0_index]
            ipr_t[ir, :] = np.abs([np.sum(self.get_psi_t(t=t, psi0=c0, energies=res.eigenvalues))
                  for t in np.linspace(0,t_final, steps)])**4
            # ipr_tmp = np.append(ipr_tmp, 1. / np.sum(ct)**4))
        ipr_t = np.sum(ipr_t, axis=0) / self.params.iterate
        return 1. / ipr_t


    def entropy_t(self, t_final: float, energy: float = 0.0, steps: int = 100) -> np.array:
        s_t = np.zeros((self.params.iterate, steps))
        for ir, res in enumerate(self.results):
            c0_index = utils.get_index_for_value(res.eigenvalues, val=energy)
            c0 = res.eigenvectors[:, c0_index]
            s_t[ir, :] = np.abs([np.sum(self.get_entropy_t(
                t=t, psi=self.get_psi_t(energies=res.eigenvalues, t=t, psi0=c0)))
                  for t in np.linspace(0,t_final, steps)])**4
        s_t = np.sum(s_t, axis=0) / self.params.iterate
        return s_t



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



if __name__ == "__main__":
    mat_size = 100
    diag = np.linspace(-mat_size/2, mat_size/2, mat_size)
    for v in np.linspace(0.01,1,3):
    # for band in np.linspace(1,mat_size, 4):
        params = Params(size=mat_size, 
                        v=v, 
                        band=int(mat_size), 
                        iterate=10, 
                        unfold=False, 
                        eigfunctions=True, 
                        ldos=True,
                        diagonal=diag
                    )
        obj = RM(params=params)
        ########################################################################
        # DO NOT TOUCH AREA AS EXAMPLES
        # obj.dos(density=True, alpha=0.6, bins=20,histtype="step", label=f"{band}")
        # obj.ldos(energy=0.0, alpha=0.7, marker="o", label=f"{int(band)}")
        # obj.plot_ipr
        # pr_t = obj.pr_t(e6 for i in range(params.iterate)]
        ########################################################################
        plt.plot(obj.pr_t(t_final=100, steps=300), label=f"{v:.2f}")
        # plt.plot(obj.entropy_t(t_final=100, steps=100), label=f"{v:.2f}")
        # plt.plot(v, obj.r_til, "bo")
    plt.legend()
    plt.show()
    
    

# Compare S(t) and PR(t) with analitical solutions