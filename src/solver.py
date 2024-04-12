from unfolding import Unfolding
from tqdm import tqdm
import numpy as np
from scipy import linalg
from models.models import Result, Results, Params
from src.time_evolution import TimeEvolution
from src.initializer import Initialize

class Solver(TimeEvolution):
    H = np.array
    def __init__(self, params: Params, init_obj) -> None:
        super().__init__(params=params)
        self.p = params
        self.init_obj: Initialize = init_obj
        self.results = np.zeros(self.p.iterate, dtype=Result)
        

    def get_results(self) -> Results:
        if self.p.iterate != 0:
            for i in tqdm(range(self.p.iterate)):
                self.H = self.init_obj._get_H()
                self.results[i] = self._solve()
            return self.results


    def _solve(self) -> Result:
        eigenvalues, self.H = np.linalg.eigh(a=self.H, UPLO="U")
        if self.p.unfold: # TODO: test it properly
            self.unfolded_energies = np.append(
                self.unfolded_energies,Unfolding(
                    eigenvalues ,fit_poly_order=12,discard_percentage=10
                ).unfolded_energies)
        else:
            self.unfolded_energies = eigenvalues
        return Result(eigenvalues=eigenvalues, 
                      eigenvectors=self.H if self.p.eigfunctions else None)


