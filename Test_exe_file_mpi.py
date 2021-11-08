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

parser = argparse.ArgumentParser()
parser.add_argument('--frelist', type=str, required=True)
parser.add_argument('--npix', type=str, required=True, help='the number of pixels for each process to analysis')
parser.add_argument('--seed', type=str, required=True, help = 'realization of the noise')

args = parser.parse_args()
a = np.array((0,1,2,3,4))

if args.frelist == 'spass_only':
    fres_list = [0, 1, 2, 3]; const = 60
    P_nu0 = np.load('/global/cscratch1/sd/jianyao/CBASS/Foreground/P_beamed_14.92_s0_128_uK_RJ.npy')

if args.frelist == 'cbass_only':
    fres_list = [1,2,3,4]; const = 50
    P_nu0 = np.load('/global/cscratch1/sd/jianyao/CBASS/Foreground/P_18.12_s0_32_uK_RJ.npy')
    
if args.frelist == 'both':
    fres_list = [0,1,2,3,4]; const = 100
    P_nu0 = np.load('/global/cscratch1/sd/jianyao/CBASS/Foreground/P_11.99_s0_32_uK_RJ.npy')
    
if rank == 0:
    start = time.time()

## configuration 

fres = np.array([2.3, 23, 28.4, 33]);Nside = 128;  ### 28 -> 28.4

## import data
total_P = np.load('/global/cscratch1/sd/jianyao/Data/total_P_smoothed_128.npy')
total_sigma = np.load('/global/cscratch1/sd/jianyao/Data/total_sigma_smoothed_128.npy')

mask_index = np.load('/global/cscratch1/sd/jianyao/Data/mask_com_smo_128_index.npy')
mask_index = np.append(mask_index, np.arange(142)) ## to fill the data so that every rank handles with the equal amount of pixels.

## likelihood analysis
npara = 2; 

def prior_o(cube):
    
    As = cube[0]*400
    beta = cube[1]*4 - 4
    return [As, beta]

def prior(cube, a):
    A0 = cube[0]*a 
    beta = cube[1]*2 - 4
    
    return [A0, beta]

def prior_mid(cube, a):
    A0 = cube[0]*a 
    beta = cube[1]*0.2 - 3.1
    
    return [A0, beta]

def prior_flex(cube, A0):
    
    As = cube[0]*120 + (A0 - const) # -60, +60
#     beta = cube[1]*2 - 4
    beta = cube[1]*0.2 - 3.1
    
    return [As, beta]

logL = Loglikeli.logLike(Nside, fres,fres_list, total_P, total_sigma, 0)
 
def log_run(logL, index):
    
    logL.index = index
    
    sampler = dynesty.NestedSampler(logL.loglike, prior_o, npara, nlive=400, bootstrap = 0, sample = 'rslice')
        
    sampler.run_nested(dlogz = 0.1, print_progress=False) #
    results = sampler.results
    
    if index%250 == 0:
        results_reduced = {'samples':results['samples'], 'logwt':results['logwt'], 'logz': results['logz']}
        np.save('/global/cscratch1/sd/jianyao/Data/Results/chains/pixel_%s.npy'%index, results_reduced)
    
    samples, weights = results.samples, np.exp(results.logwt - results.logz[-1])
    mean, cov = dyfunc.mean_and_cov(samples, weights)
    
#     As = quantile(samples[:,0], [0.5], weights)[0];
#     beta_s = quantile(samples[:,1], [0.5], weights)[0]
    As, beta_s = mean[0], mean[1]
    sig_A, sig_B = np.sqrt(cov[0,0]), np.sqrt(cov[1,1])
    
    return np.array((As, sig_A, beta_s, sig_B))

nid = int(args.seed) ## node id used; 0-62
N = int(args.npix) # 16 pixel for each rank; 62 ranks each node; 42 nodes in total.

subset_pixels = mask_index[nid*992:(nid+1)*992][np.arange((rank)*N, (rank+1)*N)]    
# print(subset_pixels)
paras = np.zeros((N, 4)) ## mean value and uncertainty for As and beta_s
j = 0
start_all = time.time()
for n in subset_pixels:
    start = time.time()
    
    try:
        paras[j] = log_run(logL, n)
        
    except Exception as e: 
        print(e)
        print('index is:', n)
        paras[j] = np.array((1024, 1024, 1024, 1024))
    j += 1
                    
sendbuf = paras

recvbuf = None
if rank == 0:
    recvbuf = np.ones(size*4*N, dtype='d')

comm.Gather(sendbuf, recvbuf, root=0)

if rank == 0:
    print(dynesty.__path__)
    if args.frelist == 'spass_only':
        np.save('/global/cscratch1/sd/jianyao/Data/Results/Dyne_As_betas_SPASS_128_%03d.npy'%(int(args.seed)), recvbuf)
    if args.frelist == 'cbass_only':
        np.save('/global/cscratch1/sd/jianyao/CBASS/Results/s0_only_homo_noise/Dyne_As_betas_masked_both_32_with_CBASS_only_%03d.npy'%(int(args.seed)), recvbuf)
    if args.frelist == 'both':
        np.save('/global/cscratch1/sd/jianyao/CBASS/Results/s0_only_homo_noise/Dyne_As_betas_masked_both_32_with_SPASS_CBASS_%03d.npy'%(int(args.seed)), recvbuf)
    
    end_all = time.time()
    print('Time cost is %s mins.'%((end_all-start_all)/60))

