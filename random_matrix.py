
from matplotlib import pyplot as plt
import numpy as np
import utils
from physical_functions import PhysicalFunctions as PF
from tqdm import tqdm
from unfolding import Unfolding 
from models import Params, Result, Results

class RM(PF):
    h = None
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
        if self.params.H is not False:
            self.h = self.params.H
        else:
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


if __name__ == "__main__":
    params = Params(iterate=10,eigfunctions=True, H=np.array([1,1,1,1]).reshape(2,2))
    obj = RM(params=params)
    s = obj.entropy_t(energy=0,t_final=1, steps=100)
    plt.plot(s, ":")
    plt.show()