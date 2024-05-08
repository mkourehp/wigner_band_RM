
from matplotlib import pyplot as plt
import numpy as np
from typing import List
from src.initializer import Initialize
from src.solver import Solver
from physical_functions import PhysicalFunctions as PF
from models.models import Params, Result
plt.rcParams["figure.figsize"] = (6,4)


class RM:
    _results = None
    pf = None
    def __init__(self, params : Params) -> None:
        self.params = params
        self.init = Initialize(params=params)
        self.params.h0 = self.init.h0
        self.solver = Solver(params=params, init_obj=self.init)
        self.pf = PF(params=params, results=self.results)


    @property
    def results(self)-> List[Result]:
        if self._results is None:
            self._results = self.solver.get_results() 
            return self._results
        return self._results


if __name__ == "__main__":
    size = 200
    t_array = np.linspace(0,5,500)
    for v in [0.05, 0.2, 0.35, 0.5]:
        params = Params(size=size, band=20, iterate=20, v=v, fixed_diagonal=True, eigfunctions=True)
        rm_obj = RM(params=params)
        psi0 = rm_obj.pf.get_vector_from_energy(energy=0.0)[0]
        ent_tot = []
        fig, ax = plt.subplots()
        for i in range(params.iterate):
            ent = []
            for t in t_array:
                psi=rm_obj.pf.get_psi_t(t,psi0=psi0,
                    eigenvalues=rm_obj.results[i].eigenvalues)
                ent.append(rm_obj.pf.entropy(psi=psi, psi0=psi0))
            ax.plot(t_array, ent, "--",c="C00", alpha=0.5)
            ent_tot.append(ent)
        ax.plot(t_array, np.average(ent_tot, axis=0), "-", c="C01", label="average")
        fig.suptitle(fr"$v$={v:.2f}")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$S(t)$")
        plt.legend()
        # plt.xscale("log")
        plt.show()
        # fig.savefig(f"{v}".replace(".","_"))


