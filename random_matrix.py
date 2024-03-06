
from matplotlib import pyplot as plt
import numpy as np 
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class Result:
    eigenvalues: np.array
    eigenvectors: np.array


class RM:
    h = None
    energies = np.array([])
    def __init__(self, 
                 matrix_size : int = 2,
                 v: float = 1.0,
                 iterate: int = 1,
                 band: int = False
                 ) -> None:
        assert matrix_size >= 2 , ValueError(f"Matrix size must be greater that 1. it is {matrix_size}")
        assert  band <= matrix_size , ValueError(
            f"Band must be greater that matrix size!  matrix_size: {matrix_size}, band: {band}")
        
        self.size, self.v, self.iterate, self.band = matrix_size, v, iterate, band
        self._setH()
        if iterate: self._iterate()


    def _setH(self):
        # TODO: Diagonal must be the same always.
        self.h = sum(
            [(self.v if i!=0 else 1) * np.diag(np.random.randn(self.size-i), i) for i in range(self.band)]
              )
        self.h += self.h.T


    def solve(self) -> Result:
        res = np.linalg.eigh(a=self.h, UPLO="U")
        self.energies = np.append(self.energies, res.eigenvalues)
        return Result(eigenvalues=res.eigenvalues, eigenvectors=res.eigenvectors)

    def _iterate(self):
        if self.iterate == 1:
            self.solve()
        for _ in tqdm(range(self.iterate)):
            self._setH()
            self.solve()

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


if __name__ == "__main__":
    obj = RM(matrix_size=1000, v=1, iterate=1, band=300)
    obj.plot_ipr
    
