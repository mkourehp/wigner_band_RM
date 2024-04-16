import numpy as np
import utils
from typing import List
from models.models import Params, Result
import matplotlib.pyplot as plt
from src.time_evolution import TimeEvolution

class PhysicalFunctions(TimeEvolution):
    def __init__(self, params: Params, results: List[Result]) -> None:
        self.p: Params = params
        self.res: List[Result] = results
   
    e_min = property((lambda self : np.min([r.eigenvalues for r in self.results])))
    e_max = property((lambda self : np.max([r.eigenvalues for r in self.results])))
    energies = property(lambda self : np.array([r.eigenvalues for r in self.results]))
    eigfuncs = property(lambda self : np.array([r.eigenvectors for r in self.results]))

    @staticmethod
    def get_vector_from_eigenvalue(evecs: np.array, evals: np.array, value: float=0.0):
        """Returns the eigenvector with the smallest difference between 'value' and its eigenvalue""" 
        indx = np.argmin(np.abs(evals-value))
        return evecs[:, indx]

    @staticmethod
    def entropy(psi: np.array, psi0: np.array) -> np.array:
        if psi.shape == psi.size:
            if any([c==1 for c in np.real(psi)]): return 0.0
            return np.sum(np.abs(psi**2) * np.log(np.abs(psi**2)))
        else:
            res = np.array(psi.shape[0]*[0.])
            for i, ps in enumerate(psi):
                if any([c==1 for c in np.real(ps)]): continue
                res[i] = -sum([np.abs(p)**2 * np.log(np.abs(p)**2) for p in ps])
            return res

