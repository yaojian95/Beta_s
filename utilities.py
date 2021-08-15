import healpy as hp
import numpy as np

def Mask(m, mask):
    masked = hp.ma(m)
    masked.mask = np.logical_not(mask)
    
    return masked

def Bp2f(m, mask_index, nside):
    '''
    from mask area to full sky.
    
    Return:
    --------
    A0, sigma_A0; beta_s, sigma_beta_s
    '''
    
    Betas = np.zeros(12*nside**2); sig_B = np.zeros(12*nside**2)
    Betas[mask_index] = m[2:7019:4]; sig_B[mask_index] = m[3:7020:4]
    Betas[Betas==0] = hp.UNSEEN; sig_B[sig_B == 0] = hp.UNSEEN;
    
    As = np.zeros(12*nside**2); sig_A = np.zeros(12*nside**2);
    As[mask_index] = m[0:7017:4]; sig_A[mask_index] = m[1:7018:4]
    As[As==0] = hp.UNSEEN; sig_A[sig_A == 0] = hp.UNSEEN;

    return As, sig_A, Betas, sig_B

def prior_H(cube):
    A0 = cube[0]*1000 # 0-100
    beta = cube[1]*2 - 4
    
    return [A0, beta]

def prior_L(cube):
    A0 = cube[0]*300 # 0-100
    beta = cube[1]*2 - 4
    
    return [A0, beta]