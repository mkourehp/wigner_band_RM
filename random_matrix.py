
from matplotlib import pyplot as plt
import numpy as np
from src.initializer import Initialize
from src.solver import Solver
from physical_functions import PhysicalFunctions as PF
from models.models import Params, Result, Results

class RM:
    _results = None
    def __init__(self, params : Params) -> None:
        self.params = params
        self.init = Initialize(params=params)
        self.solver = Solver(params=params, init_obj=self.init)
        self.pf = PF(params=params)


    @property
    def results(self)-> Results:
        if not self._results:
            self._results = self.solver.get_results() 
            return self._results
        return self._results




if __name__ == "__main__":
    params = Params(size=10, band=3,iterate=1,eigfunctions=True, check=True)
    obj = RM(params=params)
    for res in obj.results:
        psi_0 = np.zeros(params.size)
        psi_0[0] = 1.0
        for t in np.linspace(0,10,1000):
            plt.plot(t, obj.solver.get_psi_t(t=t, eigenvalues=res.eigenvalues, psi_0=psi_0)[0], "o")
    plt.show()
    
