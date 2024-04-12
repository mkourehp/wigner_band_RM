import numpy as np
from scipy import linalg
from models.models import Params
from utils import print_h

class Initialize:
    def __init__(self, params:Params) -> None:
        self.p = params
        self.diagonal = np.sort(np.random.randn(self.p.size)) if self.p.diagonal.size==0 else self.p.diagonal
    def params_check(self):
        assert self.p.size >= 2 , ValueError(f"Matrix size must be greater that 1. it is {self.p.size}")
        assert  self.p.band <= self.p.size , ValueError(
            f"Band must be greater that matrix size!  matrix_size: {self.p.size}, band: {self.p.band}")
        print("Check Passed!")

    def _get_H(self) -> np.array:
        if not self.p.H is not False:
            h: np.array = sum([(self.p.v if i!=0 else 1) * np.diag(
                np.random.randn(self.p.size-i), i)
                for i in range(1, self.p.band+1)]
                )
            h += np.diag(self.diagonal)
            if self.p.check:
                assert all([d1 == d2 for d1, d2 in zip(h.diagonal(), self.diagonal)]), "DiagonalError!"
                for i in range(self.p.band+1):
                    assert all(h.diagonal(i)), "Non Zero Elements Error!"
                    if i==0: continue
                    assert not all(h.diagonal(self.p.size-i)), "Zero Elements Error!"
            return h
        return self.p.H