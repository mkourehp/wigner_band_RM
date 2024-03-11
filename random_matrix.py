
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
                 band: int = False
                 ) -> None:
        assert matrix_size >= 2 , ValueError(f"Matrix size must be greater that 1. it is {matrix_size}")
        assert  band <= matrix_size , ValueError(
            f"Band must be greater that matrix size!  matrix_size: {matrix_size}, band: {band}")
        
        self.size, self.v, self.iterate, self.band = matrix_size, v, iterate, band or matrix_size
        self.diagonal = np.random.randn(self.size)
        self._setH()
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
        self.energies = sorted(np.append(self.energies, res.eigenvalues))
        if True:
            self.unfolded_energies = np.append(self.unfolded_energies, 
                                           Unfolding(self.energies,fit_poly_order=10,discard_percentage=10).unfolded_energies)
        return Result(eigenvalues=res.eigenvalues, eigenvectors=res.eigenvectors)

    def _iterate(self):
        if self.iterate == 1:
            self._solve()
        for _ in range(self.iterate):
            self._setH()
            self._solve()

    @property
    def dos(self):
        self._iterate() if self.energies.size == 0 else None
        plt.hist(self.energies, bins="auto", density=True, histtype="bar", alpha=0.7)
        plt.show()

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
        self._iterate() if self.energies.size == 0 else None
        plt.plot(range(self.size),[self.get_ipr(n) for n in range(self.size)], "o")
        plt.show()

       
    
    @property 
    def r(self):
        self._iterate() if self.energies.size == 0 else None
        self.energies.sort()
        s = np.diff(self.energies)
        return [s[i] / s[i-1] for i in range(s.size) if s[i-1] != 0]
    
    @property
    def r_til(self):
        return [min(rr,1./rr) for rr in self.r if rr != 0]


if __name__ == "__main__":
    import threading 
    def run(v):
        obj = RM(matrix_size=200, v=v, iterate=10)
        plt.hist(np.diff(obj.unfolded_energies), density=True, bins="auto",label=f"{v:.3f}", alpha=0.7)
    targets = [threading.Thread(target=run, args=(vv,))  for vv in np.linspace(0.01,10,5)]
    [t.start() for t in targets]
    [t.join() for t in targets]
    stopme = True
    # plt.legend()
    # plt.show()

