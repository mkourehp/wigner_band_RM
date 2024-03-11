import numpy as np
import scipy as sp
import scipy.optimize as spopt
from matplotlib import pyplot as plt
#################    CLASS FOR UNFOLDING A ENERGY SPECTRUM  ###################
class Unfolding:
    unfolded_energies = np.array([])
    def __init__(self,file_address,fit_poly_order,discard_percentage,err = False):
        self.address =   file_address      
        self.polyfit =   fit_poly_order
        self.dis_per =   discard_percentage
        self.err     =   err
        self._unfold()
      
    def _unfold(self):
        pi = np.pi
        def __Exp(x):
            return np.exp(x)         
        def __NcBrody(x,b):
            al = sp.special.gamma((b+2.0)/(b+1.0))**(b+1.0)
            n = 1.0 - __Exp(-al*x**(b+1.0))
            return n         
        def __NcBR(x,q):
            b = sp.special.erfc(np.sqrt(pi)*q*x/2.0)
            ans = 1. - __Exp(x*(q-1))* (( q * __Exp(-pi*q*q*x*x/4.) + (1.-q)*b  ))
            return ans
        
        if isinstance(self.address, str):
            E1 = np.load(self.address) if self.address.endswith("npy") else np.loadtxt(self.address)
        else:
            E1 = self.address
        E1 = np.sort(E1)
        percentage = self.dis_per
        polyorder  =self.polyfit
        NN = E1.size
        r = int(percentage*1.0*NN/100.0)
        E = E1[r:NN-r]
        N = E.size
        Nc = np.zeros(100*N)
        rho , _ = np.histogram(E,bins=Nc.size-1)
        for i in range(Nc.size-1):
            Nc[i+1] = Nc[i] + rho[i]
        EE_x = np.linspace(E[0],E[-1],Nc.size)
        eta_av = np.polyfit(EE_x,Nc,polyorder)        
        E_uf = np.polyval(eta_av,E)
        E_uf = np.sort(E_uf)
        self.unfolded_energies = E_uf
        return E_uf
        ##########return E_uf            
        s = np.zeros(E.size)
        for i in range(N-1):
            s[i] = E_uf[i+1] - E_uf[i]          
        s = s / np.average(s)
        N_s = s.size
        Nc_s = np.zeros(N_s)
        h , _ = np.histogram(s,bins=Nc_s.size-1)
        x = np.linspace(np.min(s),np.max(s),Nc_s.size)
        for i in range(Nc_s.size-1):
            Nc_s[i+1] = Nc_s[i] + h[i] 
        Nc = Nc_s / np.max(Nc_s)
        b , err_b = spopt.curve_fit(__NcBrody,x,Nc)
        br , err_br = spopt.curve_fit(__NcBR,x,Nc)

        if self.err is True:
            b  = spopt.curve_fit(__NcBrody,x,Nc)
            br = spopt.curve_fit(__NcBR,x,Nc)
            return E_uf , [b[0][0] , b[1][0][0]] , [br[0][0] , br[1][0][0]]
        else:
            b , _ = spopt.curve_fit(__NcBrody,x,Nc)
            br , _ = spopt.curve_fit(__NcBR,x,Nc)
            return E_uf , b, br
        
    

if __name__ == "__main__":
    f = "eigenenergies.npy"
    obj = Unfolding(file_address=f,fit_poly_order=10,discard_percentage=10)
    # plt.plot(obj.unfolded_energies, "o")
    plt.hist(np.diff(obj.unfolded_energies), density=True, bins="auto")
    plt.show()


