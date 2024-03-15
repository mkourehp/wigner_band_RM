from dataclasses import dataclass
import numpy as np
from typing import Optional, List


@dataclass
class Params:
    size : int = 2,
    v: float = 1.0,
    iterate: int = 1,
    eigfunctions : bool = False,
    ldos: bool = False,
    band: int = False,
    unfold: bool = False,
    check: bool = False



@dataclass
class Result:
    eigenvalues: np.array
    eigenvectors: Optional[np.array] = None
    ldos: Optional[np.array] = None


@dataclass
class Results:
    r: List[Optional[Result]]
    
    @property
    def energies(self):
        return np.array([res.eigenvalues for res in self.r])   

    @property
    def eigfuncs(self):
        return np.array([res.eigenvectors for res in self.r]) 