
from matplotlib import pyplot as plt
import numpy as np
from src.initializer import Initialize
from src.solver import Solver
from src.time_evolution import TimeEvolution

from physical_functions import PhysicalFunctions as PF
from tqdm import tqdm
from unfolding import Unfolding 
from models.models import Params, Result, Results

class RM(Initialize, Solver, TimeEvolution, PF):
    def __init__(self, params : Params) -> None:
        self.params = params
        self.params_check(params=params)
        self.resutls = self.get_results()




if __name__ == "__main__":
    params = Params(size=5, band=2,iterate=1,eigfunctions=True, check=True)
    obj = RM(params=params)
    
