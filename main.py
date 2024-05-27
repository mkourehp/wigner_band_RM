from src.random_matrix import RM
from tools.models import Params
from tools.utils import fit_line
import numpy as np
import matplotlib.pyplot as plt

markers = ["o","x","^", "d", "p", "*", "s", "v", "H"]
if __name__ == "__main__":
    size = 1000
    t_array = np.linspace(0,0.1,300)
    dE_array = np.linspace(0.05,size, 4)
    v_array = np.linspace(0,1, 4)
    b_array = np.linspace(5,size, 4)
    
    for iv, v in enumerate(v_array):
        params = Params(size=size, 
            band=size//2,
            iterate=10,
            v=v,
            fixed_diagonal=True, 
            eigfunctions=True)
        rm_obj = RM(params=params)
        for dE in dE_array:
            psi0 = rm_obj.pf.get_vector_from_energy(E0=0.0, dE=dE)
            fitted_params = []
            ent = np.zeros((t_array.size, params.iterate))
            for i, res in enumerate(rm_obj.results):
                for it, t in enumerate(t_array):
                    psi = rm_obj.pf.get_psi_t(psi0=psi0[i], t=t, eigenvalues=res.eigenvalues)
                    ent[it, i] = rm_obj.pf.entropy(psi0=psi0[i], psi=psi)
            # plt.plot(t_array, np.average(ent, axis=1), label=f"{dE:.2f}")
            fitted_params.append(fit_line(x=t_array, y=np.average(ent, axis=1), plot=True, dE=dE))
            # l = [plt.errorbar(x=dE, y=f["slope"], yerr=f["residual"],c=f"C{iv}", alpha=0.6,fmt="o", capsize=3, ecolor="C01",
            #                   label=f'v: {v:.2f}' if dE==dE_array[0] else None) for f in fitted_params]

        # plt.ylabel("Slope",  size=15)
        # plt.xlabel("dE",  size=15)           
        plt.legend()
        plt.savefig("slope.png", dpi=300)
        plt.xlim(0,0.03)
        plt.show()
