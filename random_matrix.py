
from matplotlib import pyplot as plt
import numpy as np 
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
                 unfold: bool = False
                 ) -> None:
        assert matrix_size >= 2 , ValueError(f"Matrix size must be greater that 1. it is {matrix_size}")
        assert  band <= matrix_size , ValueError(
            f"Band must be greater that matrix size!  matrix_size: {matrix_size}, band: {band}")
        
        self.size, self.v, self.iterate, self.band, self.unfold = matrix_size, v, iterate, band or matrix_size, unfold
        self.diagonal = np.random.randn(self.size)
        if iterate: self._iterate()



    def _setH(self, check_diagonal: bool = False):
        self.h = sum(
            [(self.v if i!=0 else 1) * np.diag(np.random.randn(self.size-i), i) for i in range(1, self.band)]
              )
        self.h += np.diag(self.diagonal)
        if check_diagonal:
            assert all([d1 == d2 for d1, d2 in zip(self.h.diagonal(), self.diagonal)])


    def _solve(self) -> Result:
        res = np.linalg.eigh(a=self.h, UPLO="U")
        self.energies = np.append(self.energies, res.eigenvalues)
        if self.unfold:
            self.unfolded_energies = np.append(
                    self.unfolded_energies,
                    Unfolding(self.energies,fit_poly_order=12,discard_percentage=10).unfolded_energies
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


    def ldos(self, p: float):
        assert 0<=p<=1, "Correct Error"
        ldos = []
        for row in self.energies:
            E_min, E_max = min(row), max(row)
            e = p * (E_max - E_min) + E_min # Get the energy in the pth portion of DoS
            index = np.argmin(abs(row - e))
            ldos += self.h[:,index]**2
        plt.plot(row, self.h[:,index]**2, label= f"{self.v}")


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
    for v in np.linspace(1,5,3):
        obj = RM(matrix_size=500, v=v, band=200, iterate=1, unfold=False)
        # plt.plot(v, np.average(obj.r_til), "bo")
        obj.ldos(0.5)
    plt.legend()
    plt.show()



# Calculate LDoS !!!