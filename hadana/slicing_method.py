from .packages import *

def get_sliceID_histograms(f_KEi, f_KEf, f_int_type, f_containing, KEbins):
    f_SIDini = []
    f_SIDend = []
    f_SIDint_ex = []
    f_Nevts = len(f_KEi)
    Nbins = len(KEbins)
    for ievt in range(f_Nevts):
        KE_ini = f_KEi[ievt]
        KE_end = f_KEf[ievt]
        intrcn = f_int_type[ievt]
        containing = f_containing[ievt]
        
        ## initial slice ID
        for SID_ini in range(Nbins-1):
            if KEbins[SID_ini] < KE_ini:
                break
                
        ## end slice ID
        for SID_end in range(Nbins-1):
            if KEbins[SID_end+1] < KE_end:
                break
        
        ## remove incomplete slices
        if SID_end < SID_ini: # SID_end == SID_ini - 1
            SID_ini = -1
            SID_end = -1
            
        ## interaction slice ID
        SID_int_ex = SID_end
        
        ## non-signal interaction
        if intrcn != 1:
            SID_int_ex = -1

        ## non-containing tracks
        if not containing:
            SID_int_ex = -1
            
        f_SIDini.append(SID_ini)
        f_SIDend.append(SID_end)
        f_SIDint_ex.append(SID_int_ex)
    return f_SIDini, f_SIDend, f_SIDint_ex

def derive_energy_histograms(f_SIDini, f_SIDend, f_SIDint_ex, Nbins, f_evtweight=None):
    if f_evtweight is None:
        f_evtweight = np.ones(len(f_SIDini))
    ### derive the initial, end, interaction histogram (do not use the null bin)
    f_Nini,_ = np.histogram(f_SIDini, bins=range(-1, Nbins), weights=f_evtweight)
    f_Nend,_ = np.histogram(f_SIDend, bins=range(-1, Nbins), weights=f_evtweight)
    f_Nint_ex,_ = np.histogram(f_SIDint_ex, bins=range(-1, Nbins), weights=f_evtweight)
    f_Nini = f_Nini[1:]
    f_Nend = f_Nend[1:]
    f_Nint_ex = f_Nint_ex[1:]
    
    ### derive the incident histogram
    f_Ninc = np.zeros_like(f_Nini)
    for ibin in range(Nbins-1): # again, the formula can be found in the reference below
        ## two equivalent way to calculate the incident histogram
        for itmp in range(0, ibin+1):
            f_Ninc[ibin] += f_Nini[itmp]
        for itmp in range(0, ibin):
            f_Ninc[ibin] -= f_Nend[itmp]
        '''for itmp in range(ibin, Nbins-1):
            f_Ninc[ibin] += f_Nend[itmp]
        for itmp in range(ibin+1, Nbins-1):
            f_Ninc[ibin] -= f_Nini[itmp]'''
    return f_Nini, f_Nend, f_Nint_ex, f_Ninc

def get_3D_histogram(f_SIDini, f_SIDend, f_SIDint_ex, Nbins, f_evtweight=None):
    f_Nevts = len(f_SIDini)
    if f_evtweight is None:
        f_evtweight = np.ones(f_Nevts)
    Nbins_3D = Nbins**3
    f_N3D = np.zeros(Nbins_3D)
    f_N3D_errsq = np.zeros(Nbins_3D)
    f_SID3D = np.zeros(f_Nevts, dtype=np.int32)
    
    for ievt in range(f_Nevts): # fill in the combined variable
        SID_ini = f_SIDini[ievt] + 1 # so that null bin moves to 0 (covenient for integer division and modulus below)
        SID_end = f_SIDend[ievt] + 1
        SID_int_ex = f_SIDint_ex[ievt] + 1
        weight = f_evtweight[ievt]
        
        SID3D = SID_ini + Nbins * SID_end + Nbins*Nbins * SID_int_ex # definition of the combined variable
        f_SID3D[ievt] = SID3D
        
        f_N3D[SID3D] += weight
        f_N3D_errsq[SID3D] += weight*weight
        
    f_N3D_Vcov = np.diag(f_N3D_errsq) # this is a fill process. Each bin is independent, so the covariance matrix is diagonal
    return f_SID3D, f_N3D, f_N3D_Vcov

def get_Cov_3SID_from_N3D(f_N3D_Vcov, Nbins):
    Nbins_3D = Nbins**3
    Jac_N3D_3SID = np.zeros([3*Nbins, Nbins_3D])
    for jbin in range(Nbins_3D):
        ## use integer division and modulus to project back to 1D histograms
        ibx = jbin % Nbins # get SID_ini
        iby = (jbin // Nbins) % Nbins # get SID_end
        ibz = (jbin // Nbins // Nbins) % Nbins # get SID_int_ex
        Jac_N3D_3SID[ibx, jbin] = 1
        Jac_N3D_3SID[Nbins+iby, jbin] = 1
        Jac_N3D_3SID[2*Nbins+ibz, jbin] = 1
    
    ### derive the covariance matrix 3SID_Vcov
    #f_3SID_Vcov = np.einsum("ij,jk,lk->il", Jac_N3D_3SID, f_N3D_Vcov, Jac_N3D_3SID) # computation too slow
    f_3SID_Vcov = np.einsum("ij,jk->ik", Jac_N3D_3SID, f_N3D_Vcov)
    f_3SID_Vcov = np.einsum("ij,kj->ik", f_3SID_Vcov, Jac_N3D_3SID)
    return f_3SID_Vcov

def get_Cov_3N_from_3SID(f_3SID_Vcov, Nbins):
    Jac_3SID_3N = np.zeros([3*(Nbins-1), 3*Nbins])
    for ibin in range(Nbins-1): # for Ninc
        ## two equivalent way to calculate the incident histogram
        for jbin in range(ibin+2, Nbins):
            Jac_3SID_3N[ibin, jbin] = -1
        for jbin in range(ibin+1+Nbins, 2*Nbins):
            Jac_3SID_3N[ibin, jbin] = 1
        '''for jbin in range(1, ibin+2):
            Jac_3SID_3N[ibin, jbin] = 1
        for jbin in range(1+Nbins, ibin+1+Nbins):
            Jac_3SID_3N[ibin, jbin] = -1'''
    for ibin in range(Nbins-1, 2*(Nbins-1)): # for Nend
        Jac_3SID_3N[ibin, ibin+2] = 1
    for ibin in range(2*(Nbins-1), 3*(Nbins-1)): # for Nint_ex
        Jac_3SID_3N[ibin, ibin+3] = 1
    
    ### derive the covariance matrix 3N_Vcov
    f_3N_Vcov = np.einsum("ij,jk,lk->il", Jac_3SID_3N, f_3SID_Vcov, Jac_3SID_3N)
    return f_3N_Vcov

def calculate_XS_Cov_from_3N(f_Ninc, f_Nend, f_Nint_ex, f_3N_Vcov, KEbins, bb):
    NA = 6.02214076e23 # Avogadro constant (mol^{-1})
    rho_tar = 1.396 # density (g/cm^3) # 1.401
    M_tar = 39.95 # molar mass (g/mol)
    n_tar = rho_tar*NA/M_tar # number density (cm^{-3})

    Nbins, _, cKE, wKE = utils.set_bins(KEbins)
    f_XS = np.zeros_like(f_Ninc)
    Jac_3N_XS = np.zeros([Nbins-1, 3*(Nbins-1)])
    for ibin in range(Nbins-1):
        prefact = bb.mean_dEdx(cKE[ibin]) / (n_tar*wKE[ibin]*2) * 1e27 # pre-factor
        Ninc = f_Ninc[ibin]
        Nend = f_Nend[ibin]
        Nint_ex = f_Nint_ex[ibin]
        
        if Nend <= 0 or Ninc-Nend <= 0: # no track ends in this bin, or all tracks end in this bin
            f_XS[ibin] = 0
            Jac_3N_XS[ibin, ibin] = 0
            Jac_3N_XS[ibin, ibin+Nbins-1] = 0
            Jac_3N_XS[ibin, ibin+2*(Nbins-1)] = 0
        else:
            f_XS[ibin] = prefact * Nint_ex/Nend * np.log(Ninc/(Ninc-Nend)) # same as above when we calculate the signal cross-section σ 
            Jac_3N_XS[ibin, ibin] = prefact * Nint_ex / Ninc / (Nend-Ninc) # ∂σ/∂Ninc
            Jac_3N_XS[ibin, ibin+Nbins-1] = prefact * Nint_ex/Nend * (1/(Ninc-Nend) - 1/Nend*np.log(Ninc/(Ninc-Nend))) # ∂σ/∂Nend
            Jac_3N_XS[ibin, ibin+2*(Nbins-1)] = prefact * 1 / Nend *np.log(Ninc/(Ninc-Nend)) # ∂σ/∂Nint_ex
    
    ### derive the covariance matrix XS_Vcov
    f_XS_Vcov = np.einsum("ij,jk,lk->il", Jac_3N_XS, f_3N_Vcov, Jac_3N_XS)
    return f_XS, f_XS_Vcov


if __name__ == "__main__":
    pass