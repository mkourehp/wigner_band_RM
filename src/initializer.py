import numpy as np
from scipy import linalg
from models.models import Params
from utils import print_h

class Initialize:
    def __init__(self, params:Params) -> None:
        self.p = params
        self.h0 = self._get_h0() if self.p.h0.size==0 else self.p.h0
        self.params_check()
    
    def _get_h0(self):
        return np.arange(self.p.size) - (self.p.size-1)/2

    def params_check(self):
        assert self.p.size >= 2 , ValueError(f"Matrix size must be greater that 1. it is {self.p.size}")
        assert  self.p.band <= self.p.size , ValueError(
            f"Band must be greater that matrix size!  matrix_size: {self.p.size}, band: {self.p.band}")
        print("Check Passed!")

    def _get_H(self) -> np.array:
        if self.p.h is False:
            h: np.array = sum([(self.p.v if i!=0 else 1) * np.diag(
                np.random.randn(self.p.size-i), i)
                for i in range(1, self.p.band+1)]
                )
            h += np.diag(self.h0) if self.p.fixed_diagonal else np.diag(np.random.randn(self.p.size))
            if self.p.check:
                for i in range(0, self.p.size):
                    if not self.p.v > 0:
                        continue
                    if i>self.p.band:
                        assert not all(h.diagonal(i)), "Zero Elements Error!"
                    else:
                        assert all(h.diagonal(self.p.size-i)), "Non Zero Elements Error!"
            return h
        return self.p.h