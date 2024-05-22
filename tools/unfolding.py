import numpy as np
import scipy as sp
import scipy.optimize as spopt
from matplotlib import pyplot as plt
#################    CLASS FOR UNFOLDING A ENERGY SPECTRUM  ###################


class Unfolding:
    unfolded_energies = np.array([])
    def __init__(self,file_address,fit_poly_order,discard_percentage):
        self.address =   file_address      
        self.polyfit =   fit_poly_order
        self.discard_percentage =   discard_percentage

        if isinstance(self.address, str):
            self.folded_energies = np.load(self.address) if self.address.endswith("npy") else np.loadtxt(self.address)
        else:
            self.folded_energies = self.address

        self.folded_energies = np.sort(self.folded_energies)
        self.unfolded_energies = self._unfold()
      
    def _unfold(self):
        polyorder  =self.polyfit
        r = int(self.discard_percentage*1.0*self.folded_energies.size/100.0)
        E = self.folded_energies[r:self.folded_energies.size-r]
 
        x = np.sort(self.folded_energies)
        y = np.arange(np.array(self.folded_energies).size)
        
        eta_av = np.polyfit(x,y,polyorder)        
        E_uf = np.polyval(eta_av,E)
        E_uf = np.sort(E_uf)
        return E_uf


if __name__ == "__main__":
    f = "energies.npy"
    obj = Unfolding(file_address=f,fit_poly_order=10,discard_percentage=10)
    plt.plot(obj.unfolded_energies, "o")
    # plt.hist(np.diff(obj.unfolded_energies), density=True, bins="auto")
    plt.show()


