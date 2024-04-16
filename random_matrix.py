
from matplotlib import pyplot as plt
import numpy as np
from typing import List
from src.initializer import Initialize
from src.solver import Solver
from physical_functions import PhysicalFunctions as PF
from models.models import Params, Result

class RM:
    _results = None
    def __init__(self, params : Params) -> None:
        self.params = params
        self.init = Initialize(params=params)
        self.solver = Solver(params=params, init_obj=self.init)
        self.pf = PF(params=params, results=self.solver.results)


    @property
    def results(self)-> List[Result]:
        if not self._results:
            self._results = self.solver.get_results() 
            return self._results
        return self._results




if __name__ == "__main__":
    params = Params(size=5, v=1., band=4,iterate=1,eigfunctions=True, check=True)
    obj = RM(params=params)
    psis_t, ss_t = params.iterate*[None], params.iterate*[None]
    for i, res in enumerate(obj.results):
        psi_0 = obj.pf.get_vector_from_eigenvalue(evals=res.eigenvalues, evecs=res.eigenvectors, value=0.0)
        psis_t[i] = np.array([obj.solver.get_psi_t(
            t=t,psi_0=psi_0,eigenvalues=res.eigenvalues)
            for t in np.linspace(0,1,2)])
        ss_t[i] = np.array([obj.pf.entropy(psi_t, psi0=psi_0) for psi_t in psis_t])
    
    [plt.plot(s_t, "-o", c=f"C0{i}")
     for i, s_t in enumerate(ss_t)]
    plt.show()