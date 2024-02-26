
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
                 iterate: int = 0
                 ) -> None:
        assert matrix_size >= 2 , ValueError(f"Matrix size must be greater that 1. it is {matrix_size}")
        self.ms, self.v, self.iterate = matrix_size, v, iterate
        self._setH()
        if iterate: self._iterate()


    def _setH(self):
        self.h = np.random.randn(self.ms,self.ms).astype(np.float32)
        self.h = np.diag(np.diagonal(self.h))+ (self.h - (np.diagonal(self.h)))*self.v
        self.apply_localization()




    def apply_localization(self):
        def _localize(x, y):
            return [x[j]/(abs(y-j)**1 or 1) for j in range(x.size)]

        for i in range(self.ms):
            self.h[i] = np.apply_along_axis(_localize, 0, self.h[i], i)


    def solve(self) -> Result:
        res = np.linalg.eigh(a=self.h)
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




if __name__ == "__main__":
    obj = RM(matrix_size=100, v=1)
    plt.matshow(obj.h)
    plt.colorbar()
    plt.show()
