from dataclasses import dataclass, field
import numpy as np
from typing import Optional, List



class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass
class Params:
    size : int = 2
    v: float = 1.0
    iterate: int = 1
    band: int = 2
    eigfunctions : bool = False
    check: bool = False
    h: np.array = False
    h0: np.array = np.array([])
    fixed_diagonal: bool = True



@dataclass
class Result:
    eigenvalues: np.array
    eigenvectors: Optional[np.array] = None