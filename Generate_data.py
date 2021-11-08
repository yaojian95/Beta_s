import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

import pysm3
import pysm3.units as u

import sys
sys.path.append('/global/cscratch1/sd/jianyao/PySM_public/')
from pysm.common import convert_units

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

wmap_k = fits.open('/global/cscratch1/sd/jianyao/Data/wmap_band_iqumap_r9_9yr_K_v5.fits', field = None, h = True) # nside = 512, mK_CMB
wmap_ka = fits.open('/global/cscratch1/sd/jianyao/Data/wmap_band_iqumap_r9_9yr_Ka_v5.fits', field = None, h = True)

lfi = fits.open('/global/cscratch1/sd/jianyao/Data/LFI_SkyMap_030_1024_R4.00_full.fits', field = None, h = True) # nside = 1024, K_CMB

spass_noise = fits.open('/global/cscratch1/sd/jianyao/Data/spass_dr1_1902_healpix_Tb_sens.fits', field = None, h = True) #nside_1024, K 
spass_Q = fits.open('/global/cscratch1/sd/jianyao/Data/spass_dr1_1902_healpix_Tb.q.fits',field = None, h = True)
spass_U = fits.open('/global/cscratch1/sd/jianyao/Data/spass_dr1_1902_healpix_Tb.u.fits',field = None, h = True)

f_k = convert_units("uK_CMB", 'uK_RJ', 23)
f_ka = convert_units("uK_CMB", 'uK_RJ', 33)
f_30 = convert_units("uK_CMB", 'uK_RJ', 28.4)

sen_spass = hp.read_map(spass_noise)*1e6; 
spass = np.zeros((3, 12*1024**2))
spass_Q = hp.read_map(spass_Q)*1e6; spass_Q[np.where(sen_spass < 0)] = 0
spass_U = hp.read_map(spass_U)*1e6; spass_U[np.where(sen_spass < 0)] = 0
spass[1] = spass_Q; spass[2] = -1*spass_U # IAU convention
sen_spass[np.where(sen_spass < 0)] = 0; 

k_data = hp.read_map(wmap_k, field = None)
k_iqu = np.ones((3, 12*512**2))
k_iqu[1] = k_data[1]*1e3*f_k # from mK_CMB to uK_RJ
k_iqu[2] = k_data[2]*1e3*f_k
sen_k = 1.435/np.sqrt(k_data[3])*1e3*f_k

# https://lambda.gsfc.nasa.gov/product/map/dr5/skymap_info.cfm
ka_data = hp.read_map(wmap_ka, field = None)
ka_iqu = np.ones((3, 12*512**2))
ka_iqu[1] = ka_data[1]*1e3*f_ka
ka_iqu[2] = ka_data[2]*1e3*f_ka
sen_ka = 1.472/np.sqrt(ka_data[3])*1e3*f_ka

lfi_data = hp.read_map(lfi, field = None)
lfi_iqu = np.ones((3, 12*1024**2))
lfi_iqu[1] = lfi_data[1]*1e6*f_30
lfi_iqu[2] = lfi_data[2]*1e6*f_30

sen = [sen_spass, sen_k, sen_ka]; signal = [spass, k_iqu, ka_iqu, lfi_iqu]

nside_out = 128; beam_fwhm = [8.9, 52.8, 39.6, 32.34] # spass, wmap_k, wmap_ka, LFI
nsides = [1024, 512, 512, 1024]; names = ['SPASS','WMAP_K','WMAP_Ka', 'LFI']; 

nset = np.arange((rank)*5, (rank+1)*5)

for fre in range(0,4):
    
    beam_re = np.sqrt(300**2 - beam_fwhm[fre]**2)
    
    ##-------- signal ----------##    
    if rank == 0:
        iqu_smoothed = pysm3.apply_smoothing_and_coord_transform(signal[fre], fwhm= beam_re*u.arcmin)
        P_smoothed = np.sqrt(iqu_smoothed[1]**2 + iqu_smoothed[2]**2)
        P_smoothed_128 = hp.ud_grade(P_smoothed, nside_out = nside_out)
        np.save('/global/cscratch1/sd/jianyao/Data/%s_P_smoothed_5deg_128.npy'%names[fre], P_smoothed_128)
    
    if fre==3:
        break
    ##-------- noise ----------##   
    npix = 12*nsides[fre]**2
    
    noise_P_smoothed = np.zeros((100, 12*nside_out**2))
    
    for i in nset:
        # noise = np.random.randn(3, npix)
        # noise[1, :] *= sen[fre]
        # noise[2, :] *= sen[fre]

        noise = hp.read_map('/global/cscratch1/sd/jianyao/Data/LFI_Noise/noise_030_full_map_mc_%05d.fits'%i, field = None)
        noise_smoothed = pysm3.apply_smoothing_and_coord_transform(noise*1e6*f_30, fwhm= beam_re*u.arcmin)
        noise_smoothed_128 = hp.ud_grade(noise_smoothed, nside_out =nside_out)
        
        np.save('/global/cscratch1/sd/jianyao/Data/%s_Noise/Noise_%s_uK_RJ_%03d.npy'%(names[fre],nside_out, i), noise_smoothed_128)
        
        noise_P_smoothed[i] = np.sqrt(noise_smoothed_128[1]**2 + noise_smoothed_128[2]**2)

    noise_sigmaP_smoothed = np.std(noise_P_smoothed, axis = 0)
    np.save('/global/cscratch1/sd/jianyao/Data/%s_Noise/sigma_P_%s_smoothed.npy'%(names[fre],nside_out), noise_sigmaP_smoothed)

print('Done!')