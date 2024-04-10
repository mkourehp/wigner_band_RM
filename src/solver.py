from unfolding import Unfolding
from tqdm import tqdm
import numpy as np
from scipy import linalg
from models.models import Result, Results

class Solver:

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


    def _get_H(self) -> np.array:
        if self.params.H is not False:
            return self.params.H
        else:
            h: np.array = sum([(self.params.v if i!=0 else 1) * np.diag(
                np.random.randn(self.params.size-i), i)
                for i in range(1, self.params.band)]
                )
            h += np.diag(self.diagonal)
            if self.params.check:
                assert all([d1 == d2 for d1, d2 in zip(h.diagonal(), self.diagonal)])
                for i in range(self.params.band):
                    assert all([np.sum(d)==0 for d in h.diagonal(self.params.size - i)])
            return h
