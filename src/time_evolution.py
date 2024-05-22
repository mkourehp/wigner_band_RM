from tools.models import Params
import numpy as np

class TimeEvolution:  
    def __init__(self, params) -> None:
        self.p =params
    
    @staticmethod
    def get_psi_t(t: float, eigenvalues: np.array, psi0: np.array) -> np.array:
        psi0 = psi0 / np.sqrt(np.sum(np.abs(psi0)**2))
        return np.exp(-1j*eigenvalues*t) * (psi0)