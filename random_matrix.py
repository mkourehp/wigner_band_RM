
from matplotlib import pyplot as plt
import numpy as np
from src.initializer import Initialize
from src.solver import Solver
from physical_functions import PhysicalFunctions as PF
from tqdm import tqdm
from unfolding import Unfolding 
from models.models import Params, Result, Results

class RM(Initialize, Solver, PF):
    def __init__(self, params : Params) -> None:
        self.params = params
        self.params_check(params=params)
        self.resutls = self.get_results()




if __name__ == "__main__":
    params = Params(size=10, band=3,iterate=1,eigfunctions=True, check=True)
    obj = RM(params=params)
    for res in obj.resutls:
        for t in np.linspace(0,1,100):
            print(obj.get_psi_t(t=t, ))
            plt.plot(obj.get_psi_t(t=t, psi_0=[0.5, 0.5], energies=res.eigenvalues)[0], "o")
    plt.show()
    
