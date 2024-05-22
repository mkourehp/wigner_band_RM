import numpy as np
import pytest
from tools.models import Params



def get_params(
    size : int,
    v: float,
    iterate: int = 1,
    band: int = 0,
    eigfunctions : bool = True,
    h: np.array = False,
    h0: np.array = False,
    fixed_diagonal: bool = True
    ):
    return Params(
            size=size,v=v,iterate=iterate,
            band=band,
            eigfunctions=eigfunctions,
            check=True,
            h=h, h0=np.array([]),
            fixed_diagonal=fixed_diagonal)