import numpy as np
from scipy import linalg
from models.models import Params


class Initialize:
    def params_check(self, params: Params):
        assert params.size >= 2 , ValueError(f"Matrix size must be greater that 1. it is {params.size}")
        assert  params.band <= params.size , ValueError(
            f"Band must be greater that matrix size!  matrix_size: {params.size}, band: {params.band}")
        self.diagonal = np.sort(np.random.randn(params.size)) if params.diagonal.size==0 else params.diagonal
        print("Check Passed!")

