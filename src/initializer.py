import numpy as np
from scipy import linalg
from tools.models import Params
from tools.utils import print_h

class Initialize:
    def __init__(self, params:Params) -> None:
        self.p = params
        self.h0 = self._get_h0() if self.p.h0.size==0 else self.p.h0
        self.h = self._get_H()
        self.params_check() if self.p.check else None
    
    def _get_h0(self):
        return np.random.randn(self.p.size)*self.p.size/2

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
                for b in range(1,self.p.size):
                    if b<=self.p.band :assert all([True for p in h.diagonal(b) if p!=0.0]), "Non Zero Element Error"
                    if b>self.p.band  :assert all([True for p in h.diagonal(b) if p==0.0]), "Zero Element Error"
            return h
        return self.p.h