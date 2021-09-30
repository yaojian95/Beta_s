import pymaster as nmt
import healpy as hp
import numpy as np


class BPE(object):
    
    def __init__(self, mask_in, nside, bin_w, lmax, beam = None, wsp = True):
        
        '''
        class for Band-Power-Estimation;
        
        Define the **apodized mask**, **beam weights**, **nside**, **bin-scheme**, **ell**
        
        Needs to be revised for the beam correction. Different frequency have different sigma,
        which may lead to different wsp...
        '''
        self.mask = nmt.mask_apodization(mask_in, 6, apotype='C2')
        
        self.nside = nside; self.lmax = lmax
        
#         self.beam = hp.gauss_beam(beam/60/180*np.pi, lmax = 3*self.nside); 
        
        self.b = nmt.NmtBin(self.nside, nlb=bin_w, lmax=self.lmax, is_Dell = True)
        
        self.ell_n = self.b.get_effective_ells(); self.lbin = len(self.ell_n)
        
        
        # - To construct a empty template with a mask to calculate the **coupling matrix**
        
        if wsp is True:
            
            qu = np.ones((2, 12*self.nside**2))
            m0 = nmt.NmtField(self.mask,[qu[0]])
            m2 = nmt.NmtField(self.mask, qu, purify_e=False, purify_b=True)#, beam=bl)

            # construct a workspace that calculate the coupling matrix first.
            _w00 = nmt.NmtWorkspace()
            _w00.compute_coupling_matrix(m0, m0, self.b)  ## spin-0 with spin-0
            
            _w02 = nmt.NmtWorkspace()
            _w02.compute_coupling_matrix(m0, m2, self.b)  ## spin-0 with spin-2
            
            _w22 = nmt.NmtWorkspace()
            _w22.compute_coupling_matrix(m2, m2, self.b)  ## spin-2 with spin-2
            
            self.w00 = _w00
            self.w02 = _w02
            self.w22 = _w22
        
    def compute_master(self, f_a, f_b, wsp):
        
        cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
        cl_decoupled = wsp.decouple_cell(cl_coupled)
        
        return cl_decoupled
    
    
    def Reshape(self, cl):
        '''
        reshape the cross power spectra;
        --------------------------------
        Input : cl, (Nf*Nf, lbin)
        Output : Cl, (lbin, Nf, Nf)
        
        '''
        n_f = int(np.sqrt(cl.shape[0]));
        Cl = np.zeros((self.lbin, n_f, n_f));
        for l in range(self.lbin):
            
            Cl[l, : , :] = cl[:,l].reshape(n_f, n_f);
            Cl[l] += Cl[l].T - np.diag(Cl[l].diagonal())
        
        return Cl
    
    
    def Auto_T(self, maps):
        
        '''
        auto power spectum.
        '''
        t = nmt.NmtField(self.mask, [maps[0]])
        
        return self.compute_master(t, t, self.w00)
    
    def Auto_TEB(self, maps):
        '''
        Calculate the auto-power spectra; 6 kinds of PS for each l-bin;
        
        Output
        ------------------------
        cls_all, with order TT TE TB EE EB BB.
        '''
        
        cls_all = np.ones((6, self.lbin))
        
        t = nmt.NmtField(self.mask, [maps[0]])
        qu = nmt.NmtField(self.mask, maps[1:3], purify_e=False, purify_b=True)
        
        cls_all[0] = self.compute_master(t, t, self.w00); #TT
        
        cls_all[1:3] = self.compute_master(t, qu, self.w02); #TE, TB

        cls_EB = self.compute_master(qu,qu,self.w22);
        
        cls_all[3] = cls_EB[0]; #EE
        cls_all[4] = cls_EB[1]; #EB
        cls_all[5] = cls_EB[3]; #BB
        
        return cls_all
    
    def Cross_EB(self, maps):

        '''
        Calculate the E- and B-mode power spectrum utilize Namaster purify_B method.

        Parameters:
        ----------------
        maps : input maps with QU component. Only Q and U are needed in this EB estimation. maps[i]
        ell_n : the effective number of l_bins
        mask : apodized mask 
        beam : the gaussian beam weights for each multipole

        '''

        n_f = len(maps); 
        cl = np.ones((3, n_f*n_f, self.lbin)); Cl = np.zeros((3, self.lbin, n_f, n_f))
        k = 0
        for i in range(n_f):
            for j in range(n_f):
                
                map_i = np.copy(maps[i]);
                map_j = np.copy(maps[j]); # make sure that the change of the copied one will note affect the original one.

                if i >= j :
                    
                    m_i = nmt.NmtField(self.mask, map_i, purify_e=False, purify_b=True)#beam=bl); #Q and U maps at i-th fre
                    # maps[0] will change after first call using the purification, and when maps[j] =maps[0] problems occur. 
                    m_j = nmt.NmtField(self.mask, map_j, purify_e=False, purify_b=True)#beam=bl); #Q and U maps at j-th fre

                    cross_ps = self.compute_master(m_i, m_j, self.w22) ## EE, EB, BE, BB
                   
                else:
                    cross_ps = np.zeros((4, self.lbin)) 

                cl[0][k] = cross_ps[0]; cl[1][k] = cross_ps[1]; cl[2][k] = cross_ps[3]  ## assign the EE, EB and BB power spectrum 
                k += 1
                
        Cl[0] = self.Reshape(cl[0]); Cl[1] = self.Reshape(cl[1]); Cl[2] = self.Reshape(cl[2])

        return Cl