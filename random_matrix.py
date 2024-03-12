
from matplotlib import pyplot as plt
import numpy as np
import util
from dataclasses import dataclass
from tqdm import tqdm
from unfolding import Unfolding 

@dataclass
class Result:
    eigenvalues: np.array
    eigenvectors: np.array


class RM:
    h = None
    energies = np.array([])
    unfolded_energies = np.array([])
    def __init__(self, 
                 matrix_size : int = 2,
                 v: float = 1.0,
                 iterate: int = 1,
                 band: int = False,
                 unfold: bool = False,
                 check: bool = False
                 ) -> None:
        assert matrix_size >= 2 , ValueError(f"Matrix size must be greater that 1. it is {matrix_size}")
        assert  band <= matrix_size , ValueError(
            f"Band must be greater that matrix size!  matrix_size: {matrix_size}, band: {band}")
        
        self.size, self.v, self.iterate, self.band, self.unfold, self.check = (
            matrix_size, v, iterate, band or matrix_size, unfold, check)
        self.diagonal = np.random.randn(self.size)
        if iterate: self._iterate()

    e_min = lambda self : np.min(self.energies)
    e_max = lambda self : np.max(self.energies)

    

    def _setH(self):
        self.h = sum(
            [(self.v if i!=0 else 1) * np.diag(np.random.randn(self.size-i), i) for i in range(1, self.band)]
              )
        self.h += np.diag(self.diagonal)
        if self.check:
            assert all([d1 == d2 for d1, d2 in zip(self.h.diagonal(), self.diagonal)])


    def _solve(self) -> Result:
        res = np.linalg.eigh(a=self.h, UPLO="U")
        self.energies = np.append(self.energies, res.eigenvalues)
        if self.unfold:
            self.unfolded_energies = np.append(
                self.unfolded_energies,Unfolding(
                    self.energies,fit_poly_order=12,discard_percentage=10
                ).unfolded_energies
            )
        else:
            self.unfolded_energies = self.energies
        return Result(eigenvalues=res.eigenvalues, eigenvectors=res.eigenvectors)

    def _iterate(self):
        if self.iterate != 0:
            for _ in range(self.iterate):
                self._setH()
                self._solve()
        if self.iterate: self.energies = self.energies.reshape(self.iterate, self.size)

    def dos(self):
        plt.hist(self.energies.flatten(), bins="auto", density=True, histtype="bar", alpha=0.7)
        plt.show()


    def ldos(self, ratio: float = None, 
             energy: float = None, 
             s: int = 2,
             **kwargs):
        if ratio:
            assert 0<=ratio<=1, f"ratio must be in [0,1], it is {ratio}"
        if energy:
           assert np.min(self.energies)<=energy<=np.max(self.energies),"Emin={:.3f}, Emax={:.3f}, you={:.3f}!".format(
                np.min(self.energies), np.max(self.energies), energy)
        else:
            energy = 0.0
        ldos = np.zeros((self.size, self.iterate))
        for i, row in enumerate(self.energies):
            E_min, E_max = min(row), max(row)
            if ratio:
                e = ratio * (E_max - E_min) + E_min # Get the energy in the pth portion of DoS
            else:
                e = energy
            index = np.argmin(abs(row - e))
            ldos[:,i] = self.h[:,index]**2
        ldos = np.sum(ldos, axis=1) / np.max(np.sum(ldos, axis=1)) # average 
        ldos = util.merge(ldos, s=s)
        plt.plot(np.linspace(self.e_min(), self.e_max(), ldos.size), ldos, **kwargs)



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
        plt.plot(range(self.size),[self.get_ipr(n) for n in range(self.size)], "o")
        plt.show()

       

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
    # for v in np.linspace(1,5,3):
    for s in range(5,10,2):
        obj = RM(matrix_size=100, v=1, band=100, iterate=100, unfold=False)
        obj.ldos(ratio=0.5, s=s, alpha=0.7, label=f"{s}", ls="-.")
    plt.legend()
    plt.show()



# Calculate LDoS !!!