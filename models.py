from dataclasses import dataclass, field
import numpy as np
from typing import Optional, List


@dataclass
class Params:
    size : int = 2
    v: float = 1.0
    iterate: int = 1
    band: int = 2
    eigfunctions : bool = False
    ldos: bool = False
    unfold: bool = False
    check: bool = False
    H: np.array = False
    diagonal: np.array = np.array([])



@dataclass
class Result:
    v: float
    eigenvalues: np.array
    eigenvectors: Optional[np.array] = None


@dataclass
class Results:
    r: List[Optional[Result]]