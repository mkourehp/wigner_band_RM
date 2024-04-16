
from matplotlib import pyplot as plt
import numpy as np
from typing import List
from src.initializer import Initialize
from src.solver import Solver
from physical_functions import PhysicalFunctions as PF
from models.models import Params, Result

class RM:
    _results = None
    pf = None
    def __init__(self, params : Params) -> None:
        self.params = params
        self.init = Initialize(params=params)
        self.solver = Solver(params=params, init_obj=self.init)
        self.pf = PF(params=params, results=self.results)


    @property
    def results(self)-> List[Result]:
        if self._results is None:
            self._results = self.solver.get_results() 
            return self._results
        return self._results



if __name__ == "__main__":
    params = Params(size=50, v=1., band=40,iterate=100,eigfunctions=True, check=True)
    diag = np.array(range(-params.size//2, params.size//2))
    params.diagonal = diag
    obj = RM(params=params)

    #############################################
    ## Shannon Entropy
    # for i, res in enumerate(obj.results): 
    #     psi0=obj.pf.get_vector_from_eigenvalue(
    #         evecs=res.eigenvectors,
    #         evals=res.eigenvalues,
    #         value=0.)
    #     s_t = obj.pf.entropy(psi0=psi0,res=res,t0=0,tf=2)
    #     plt.plot(s_t)
    #############################################
    ## Participaton Ratio
    # t_array = np.linspace(0, 1, 100)
    # for i, res in enumerate(obj.results): 
    #     psi_0 = obj.pf.get_vector_from_eigenvalue(
    #         evecs=res.eigenvectors,
    #         evals=res.eigenvalues,
    #         value=0.0)
    #     psi_0[np.argmin(np.abs(res.eigenvalues))] = 1.0
    #     ipr_t =  [obj.pf.ipr(
    #                 psi=obj.pf.get_psi_t(
    #                 t=t,psi_0=psi_0, 
    #                 eigenvalues=res.eigenvalues),
    #                 evecs=res.eigenvectors
    #             ) for t in t_array]
    #     plt.plot(t_array, ipr_t)
    #############################################
    

    plt.show()
