import healpy as hp
import numpy as np

from scipy.stats.mstats import gmean

import emcee
# from pathos.pools import ProcessPool as Pool
from multiprocessing import Pool ## much faster
from schwimmbad import MPIPool

import time
import Loglikeli

fres = np.array([2.3, 5, 23, 28, 33]); nu0 = gmean(fres); nside = 32

#----------------------------------------------------#

total_P = np.load('/global/cscratch1/sd/jianyao/CBASS/Observations/homo_noise/totalP_s0_%s_uK_RJ_000.npy'%nside)#/1000 ## from uK to mK
total_sigma = np.load('/global/cscratch1/sd/jianyao/CBASS/Noise/homo_noise/5_fre_sigma_P_%s_uK_RJ.npy'%nside)#/1000 ## from uK to mK
# P_nu0 = np.load('/global/cscratch1/sd/jianyao/CBASS/Foreground/P_14.92_s0_32_uK_RJ.npy')

mask_both = np.load('/global/cscratch1/sd/jianyao/CBASS/mask_both_%s.npy'%nside)
mask_index = np.load('/global/cscratch1/sd/jianyao/CBASS/masked_index.npy')

### pixels with high s/n ratios. ###
high_snr = np.load('/global/cscratch1/sd/jianyao/CBASS/Results/s0_only_homo_noise/High_SNR_pixels_300.npy')

frelist = [0,2,3,4]; index = 0

logL = Loglikeli.logLike(nside, fres,frelist, total_P, total_sigma, index)

npara = 2; 

def log_prior(theta):
    A0, beta = theta
    if 0 < A0 < 2000 and -4 < beta < -2:
        return 0.0
    return -np.inf

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + logL.loglike(theta)

pos = np.array((100, -4)) + 1e-4 * np.random.randn(64, 2)
nwalkers, ndim = pos.shape

start = time.time()
A0 = []; Beta = []

for n in high_snr:
    logL.index = n

    with Pool(64) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
#         start = time.time()
        sampler.run_mcmc(pos, 500, progress=False)
#         end = time.time()
#         multi_time = end - start
#         print("Multiprocessing took {0:.1f} seconds".format(multi_time))

        A, beta = np.median(sampler.flatchain, axis = 0)
        A0.append(A); Beta.append(beta)

np.save('/global/cscratch1/sd/jianyao/CBASS/Results/s0_only_homo_noise/Emcee_As_betas_masked_both_32_with_SPASS_v5.npy', np.array((A0, Beta)))
end = time.time()

cost = (end - start)/60.0
print('time cost is %s mins'%cost)