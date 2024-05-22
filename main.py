from src.random_matrix import RM
from tools.models import Params
import numpy as np
import matplotlib.pyplot as plt

markers = ["o","x","^", "d", "p", "*", "s", "v", "H"]
if __name__ == "__main__":
    t_array = np.linspace(0,1,50)
    params = Params(size=100, 
        band=50,
        iterate=100,
        v=1, 
        fixed_diagonal=True, 
        eigfunctions=True)
    rm_obj = RM(params=params)
    psi0 = rm_obj.pf.get_vector_from_energy(E0=0.0, dE=500)
    for i in range(params.iterate):
        for it, t in enumerate(t_array):
            psi=rm_obj.pf.get_psi_t(t,psi0=psi0[i],
                eigenvalues=rm_obj.results[i].eigenvalues)
            ent = rm_obj.pf.entropy(psi=psi, psi0=psi0[i])