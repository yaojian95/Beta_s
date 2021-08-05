from scipy.stats import rice
from scipy.stats.mstats import gmean

import time
import numpy as np

import dynesty
from dynesty import utils as dyfunc
from dynesty.utils import quantile

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

import Loglikeli

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--frelist', type=str, required=True)
parser.add_argument('--npix', type=str, required=True, help='the number of pixels for each process to analysis')

args = parser.parse_args()
a = np.array((0,1,2,3,4))

if args.frelist == 'spass_only':
    fres_list = [0,2,3,4]

if args.frelist == 'cbass_only':
    fres_list = [1,2,3,4]
    
if args.frelist == 'both':
    fres_list = [0,1,2,3,4]
    
if rank == 0:
    start = time.time()

## configuration 

fres = np.array([2.3, 5, 23, 28, 33]);Nside = 32; 

npix = 12*Nside**2
As = np.ones(npix); betas = np.ones(npix)

## import data

total_P = np.load('/global/cscratch1/sd/jianyao/CBASS/Observations/homo_noise/totalP_s0_%s_uK_RJ_000.npy'%Nside)
# P_nu0 = np.load('/global/cscratch1/sd/jianyao/CBASS/Foreground/P_nu0_s0_%s_uK_RJ.npy'%Nside)

total_sigma = np.load('/global/cscratch1/sd/jianyao/CBASS/Noise/homo_noise/5_fre_sigma_P_%s_uK_RJ.npy'%Nside)
masked_index = np.load('/global/cscratch1/sd/jianyao/CBASS/masked_index.npy')
npara = 2; 

def prior(cube):
    A0 = cube[0]*200 # 0-100
    beta = cube[1]*2 - 4
    
    return [A0, beta]

def log_run(subset_pixels):
#     start = time.time()
    As = []; betas = []
    for n in subset_pixels:
        start = time.time()
        logL = Loglikeli.logLike(Nside, fres,fres_list, total_P, total_sigma, n) ### need to be put outside the loop

        sampler = dynesty.NestedSampler(logL.loglike, prior, npara, nlive=200, bootstrap = 0)
        sampler.run_nested(dlogz = 0.1, print_progress=False) #
        results = sampler.results

        samples, weights = results.samples, np.exp(results.logwt - results.logz[-1])

        As.append(quantile(samples[:,0], [0.5], weights)[0])
        betas.append(quantile(samples[:,1], [0.5], weights)[0])

        if n%20 == 0:
            end = time.time()
            cost = (end - start)/60
            print('Time cost is %s mins for pixel %s at rank %s'%(cost, n, rank))
        
    #         print(n, As, betas)
    return np.array((As, betas))

## 45 ranks, 1755 pixels for the masked region, each rank has 39 pixels

N = int(args.npix) #int(len(masked_index)/size) ## size = 45
subset = masked_index[np.arange((rank)*N, (rank+1)*N)]

sendbuf = log_run(subset)

# print(As_beta)

recvbuf = None
if rank == 0:
    recvbuf = np.ones((size,2*N))  

comm.Gather(sendbuf, recvbuf, root=0)

if rank == 0:
    if args.frelist == 'spass_only':
        np.save('/global/cscratch1/sd/jianyao/CBASS/Results/s0_only_homo_noise/As_betas_masked_both_32_with_SPASS_only.npy', recvbuf)
    if args.frelist == 'cbass_only':
        np.save('/global/cscratch1/sd/jianyao/CBASS/Results/s0_only_homo_noise/As_betas_masked_both_32_with_CBASS_only_2.npy', recvbuf)
    if args.frelist == 'both':
        np.save('/global/cscratch1/sd/jianyao/CBASS/Results/s0_only_homo_noise/As_betas_masked_both_32_with_SPASS_CBASS.npy', recvbuf)
    
    end = time.time()
    print('Time cost is %s mins.'%((end-start)/60))

# print(As)
# print(betas)
# np.save('/global/cscratch1/sd/jianyao/CBASS/Results/s0_only_homo_noise/As.npy', As)
# np.save('/global/cscratch1/sd/jianyao/CBASS/Results/s0_only_homo_noise/Beta.npy', betas)

