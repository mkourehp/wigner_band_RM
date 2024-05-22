from matplotlib import pyplot as plt
import numpy as np
from typing import List
from src.initializer import Initialize
from src.solver import Solver
from src.physical_functions import PhysicalFunctions as PF
from tools.models import Params, Result
from tools.utils import fit_line
plt.rcParams["figure.figsize"] = (6,4)


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
