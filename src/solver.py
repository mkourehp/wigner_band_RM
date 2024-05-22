from tqdm import tqdm
from typing import Optional, List
import numpy as np
from scipy import linalg
from tools.models import Result, Params
from src.time_evolution import TimeEvolution
from src.initializer import Initialize

class Solver(TimeEvolution):
    H = np.array
    def __init__(self, params: Params, init_obj) -> None:
        self.p = params
        self.init_obj: Initialize = init_obj
        self._results: Optional[List[Result]] = None


    def get_results(self) -> List[Result]:
        f'''{100 * "%"}'''
        self._results = np.zeros(self.p.iterate, dtype=Result)
        if self.p.iterate != 0:
            for i in tqdm(range(self.p.iterate)):
                self.H = self.init_obj._get_H()
                self._results[i] = self._solve()
            return self._results

    @property
    def results(self):
        f'''Here is some docs from {__name__}'''
        if self._results != None: return self._results
        return self.get_results()

    def _solve(self) -> Result:
        f'''Here is some docs from {__name__}'''
        eigenvalues, self.H = np.linalg.eigh(a=self.H, UPLO="U")
        return Result(
            eigenvalues=eigenvalues, 
            eigenvectors=self.H if self.p.eigfunctions else None)


