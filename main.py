
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
                 iterate: int = 1
                 ) -> None:
        assert matrix_size >= 2 , ValueError(f"Matrix size must be greater that 1. it is {matrix_size}")
        self.ms, self.v, self.iterate = matrix_size, v, iterate
        self._setH()
        self._itterate()


    def _setH(self):
        self.h = np.random.randn(self.ms,self.ms)
        self.h = np.diag(np.diagonal(self.h))+ (self.h - (np.diagonal(self.h)))*self.v
    
    def _solve(self) -> Result:
        res = np.linalg.eigh(a=self.h)
        self.energies = np.append(self.energies, res.eigenvalues)
        return Result(eigenvalues=res.eigenvalues, eigenvectors=res.eigenvectors)

    def _itterate(self):
        for _ in tqdm(range(self.iterate)):
            self._setH; self._solve()

    @property
    def dos(self):
        self._itterate() if self.energies.size == 0 else None
        plt.hist(self.energies, bins=500, density=False, histtype="stepfilled", alpha=0.7)
        plt.show()

    @property
    def print_h(self):
        for row in self.h:
            [print("{:10.3}".format(r), end="\t") for r in row] ; print()

if __name__ == "__main__":
    obj = RM(matrix_size=100, v=0.5, iterate=100)
    obj.dos
    print(obj.energies.size)



# Next: Calculate the density of state