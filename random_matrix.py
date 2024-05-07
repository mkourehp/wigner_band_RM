
from matplotlib import pyplot as plt
import numpy as np
from typing import List
from src.initializer import Initialize
from src.solver import Solver
from physical_functions import PhysicalFunctions as PF
from models.models import Params, Result
plt.rcParams["figure.figsize"] = (8,6)

class RM:
    _results = None
    pf = None
    def __init__(self, params : Params) -> None:
        self.params = params
        self.init = Initialize(params=params)
        self.params.h0 = self.init.h0
        self.solver = Solver(params=params, init_obj=self.init)
        self.pf = PF(params=params, results=self.results)


    @property
    def results(self)-> List[Result]:
        if self._results is None:
            self._results = self.solver.get_results() 
            return self._results
        return self._results


if __name__ == "__main__":
    size = 200
    for band in [0, 2, 5 , 10, 50]:    
        params = Params(size=size, band=band, iterate=10,
                            eigfunctions=True, 
                            v=.1, fixed_diagonal=True)
        rm_obj = RM(params=params)
        a = rm_obj.pf.ldos(energy=0.0)
        plt.plot(a["energies"], a["ldos"], ".-", alpha=0.7, label=f"band={band}", zorder=-band)
        plt.ylabel("LDoS")
        plt.xlabel("Energy")
        plt.legend()
        # plt.yscale("log")
    # plt.savefig(f"12ldos".replace(".", "-"))
    plt.show()


