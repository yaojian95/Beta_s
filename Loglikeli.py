from scipy.stats import rice

from scipy.stats.mstats import gmean
import numpy as np

class logLike(object):
    
    def __init__(self, nside, fres, fre_list, P_maps, sigma_maps, index) :
        
        '''
        log-likelihood of P_maps
        
        Input:        
        -----------------------------------------
        nside;        
        fres: np.array; contains the all the frequencies of the input maps
        fre_list: list that contain the target frequencies (ordered number) 
        P_maps: input maps (signal + noise), with the shape of (fres, 12*nside**2)
        sigma_maps: noise std, corresponding to the noise generation in the P_maps
        index: a number, pixel index of the pixel to be analyzed.
        
        Note
        ----
        Set the geometric mean of the frequencies as the reference frequency. Then if 
        power law is statisfied, A0 = P_nu0.
        
        '''
        
        self.nside = nside; 
        
        self.fre_list = fre_list; self.fres = fres; 
        
        self.nu0 = gmean(fres[fre_list]) 
        self.P = P_maps; self.sigma = sigma_maps
        self.index = index
    
    def Pro(self, fre, A0, beta):
        
        '''
        pdf for Rice distribution at some singel frequency.
        
        Input
        ----------------------------
        fre_index: a ordered number, at which frequency of input frequency array.
        
        A0: amplitude of the P.
        beta: spectral index of synchrotron.
        '''
        nu1 = self.fres[fre]
        P = self.P[fre][self.index]; 
        
        P0 = A0*(nu1/self.nu0)**(beta)
        sigma = self.sigma[fre][self.index]

        pro = rice.pdf(P, P0/sigma, scale=sigma)
        return pro

    def loglike(self, cube):
        '''
        log-likelihood function
        
        Input 
        -----------------------------------
        
        '''
        A0, beta = cube[0], cube[1]
        Nf = len(self.fre_list)
        pro = 1
        for i in range(Nf):
            pro *= self.Pro(self.fre_list[i], A0, beta)
            
        if pro == 0:
            logL = -1e30
            return logL
        else:
            return np.log(pro)
        
        