from unfolding import Unfolding
from tqdm import tqdm
import numpy as np
from scipy import linalg
from models.models import Result, Results
from src.time_evolution import TimeEvolution


class Solver(TimeEvolution):

    def get_results(self) -> Results:
        results = np.zeros(self.params.iterate, dtype=Result)
        if self.params.iterate != 0:
            for i in tqdm(range(self.params.iterate)):
                self.h = self._get_H()
                results[i] = self._solve()
            return results


    def _solve(self) -> Result:
        eigenvalues, self.h = np.linalg.eigh(a=self.h, UPLO="U")
        if self.params.unfold: # TODO: test it properly
            self.unfolded_energies = np.append(
                self.unfolded_energies,Unfolding(
                    eigenvalues ,fit_poly_order=12,discard_percentage=10
                ).unfolded_energies)
        else:
            self.unfolded_energies = eigenvalues
        return Result(eigenvalues=eigenvalues, 
                      eigenvectors=self.h if self.params.eigfunctions else None)


