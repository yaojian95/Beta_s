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

mask_both = np.load('/global/cscratch1/sd/jianyao/CBASS/mask_both_%s.npy'%nside)
mask_index = np.load('/global/cscratch1/sd/jianyao/CBASS/masked_index.npy')

frelist = [1,2,3,4]; index = 0

logL = Loglikeli.logLike(nside, fres,frelist, total_P, total_sigma, index)

npara = 2; 

def log_prior(theta):
    A0, beta = theta
    if 0 < A0 < 1000 and -4 < beta < -2: ## change here
        return 0.0
    return -np.inf

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + logL.loglike(theta)

pos = np.array((100, -4)) + 1e-4 * np.random.randn(64, 2)
nwalkers, ndim = pos.shape

A0 = []; Beta = []

# with MPIPool() as pool:
#     if not pool.is_master():
#         pool.wait()
#         sys.exit(0)
start = time.time()
for n in mask_index:
    logL.index = n

    with Pool(64) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
#         
        sampler.run_mcmc(pos, 500, progress=False)

        A, beta = np.median(sampler.flatchain, axis = 0)
        A0.append(A); Beta.append(beta)

np.save('/global/cscratch1/sd/jianyao/CBASS/Results/s0_only_homo_noise/Emcee_As_betas_masked_both_32_with_CBASS_only_v4.npy', np.array((A0, Beta)))
end = time.time()

multi_time = (end - start)/60
print("Multiprocessing took {0:.2f} mins".format(multi_time))