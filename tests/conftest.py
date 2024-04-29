import numpy as np
import pytest
from models.models import Params



def get_params(
    size : int,
    v: float,
    iterate: int = 1,
    band: int = 0,
    eigfunctions : bool = True,
    check: bool = True,
    h: np.array = False,
    fixed_diagonal: bool = True
    ):
    return Params(
            size=size,v=v,iterate=iterate,
            band=band,
            eigfunctions=eigfunctions,
            check=check,
            h=h,
            fixed_diagonal=fixed_diagonal)