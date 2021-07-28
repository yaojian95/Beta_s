from scipy.stats import rice
from scipy.stats.mstats import gmean

import time
import numpy as np

import dynesty
from dynesty import utils as dyfunc
from dynesty.utils import quantile

from multiprocessing import Pool
cpu_num = 32
import Loglikeli


## configuration 

fres = np.array([2.3, 5, 23, 28, 33]);Nside = 64; npix = 12*Nside**2
As = np.ones(npix); betas = np.ones(npix)

## import data

total_P = np.load('/global/cscratch1/sd/jianyao/CBASS/Observations/homo_noise/totalP_s0_64_uK_RJ_000.npy')
P_nu0 = np.load('/global/cscratch1/sd/jianyao/CBASS/Foreground/P_nu0_s0_64_uK_RJ.npy')

total_sigma = np.load('/global/cscratch1/sd/jianyao/CBASS/Noise/homo_noise/5_fre_sigma_P_64_uK_RJ.npy')

npara = 2; 

def prior(cube):
    A0 = cube[0]*100 # 0-100
    beta = cube[1]*2 - 4
    
    return [A0, beta]

start = time.time()
for n in range(500):

    logL = Loglikeli.logLike(Nside, fres,[0,1,2,3,4], total_P, total_sigma, n)
    
#     with Pool(cpu_num-1) as executor:
#         sampler = dynesty.NestedSampler(logL.loglike, prior, npara, nlive=400, pool=executor, queue_size=cpu_num, bootstrap = 0)
#         sampler.run_nested(dlogz = 0.1)
#         results = sampler.results

    sampler = dynesty.NestedSampler(logL.loglike, prior, npara, nlive=400, bootstrap = 0)
    sampler.run_nested(dlogz = 0.1, print_progress=False)
    results = sampler.results

    samples, weights = results.samples, np.exp(results.logwt - results.logz[-1])

    As[n] = quantile(samples[:,0], [0.5], weights)[0]
    betas[n] = quantile(samples[:,1], [0.5], weights)[0]
    
    if n%10 == 0:
        end = time.time()
        cost = (end - start)
        print('Time cost is %s seconds for 10 pixels'%cost)
        print(n, As, betas)
        start = time.time()
np.save('/global/cscratch1/sd/jianyao/CBASS/Results/s0_only_homo_noise/As.npy', As)
np.save('/global/cscratch1/sd/jianyao/CBASS/Results/s0_only_homo_noise/Beta.npy', betas)

