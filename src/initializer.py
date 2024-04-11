import numpy as np
from scipy import linalg
from models.models import Params
from utils import print_h

class Initialize:        
    def params_check(self, params: Params):
        assert params.size >= 2 , ValueError(f"Matrix size must be greater that 1. it is {params.size}")
        assert  params.band <= params.size , ValueError(
            f"Band must be greater that matrix size!  matrix_size: {params.size}, band: {params.band}")
        self.diagonal = np.sort(np.random.randn(params.size)) if params.diagonal.size==0 else params.diagonal
        print("Check Passed!")

    def _get_H(params:Params) -> np.array:
        if not params.H is not False:
            h: np.array = sum([(params.v if i!=0 else 1) * np.diag(
                np.random.randn(params.size-i), i)
                for i in range(1, params.band+1)]
                )
            h += np.diag(params.diagonal)
            if params.check:
                assert all([d1 == d2 for d1, d2 in zip(h.diagonal(), params.diagonal)])
                for i in range(params.band+1):
                    assert all([all(d) for d in h.diagonal(i)])
            params.H = h
            return h
        return self.params.H

