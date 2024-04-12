from models.models import Params
import numpy as np

class TimeEvolution:  
    def __init__(self, params) -> None:
        self.p =params
    
    @staticmethod
    def get_psi_t(t: float, eigenvalues: np.array, psi_0: np.array) -> np.array:
        psi0 = psi_0 / np.sqrt(np.sum(np.abs(psi_0)**2))
        return np.exp(-1j*eigenvalues*t) * (psi0)