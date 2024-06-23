from packages import *
import get_hists
from BetheBloch import BetheBloch

def get_sliceID_histograms(f_KEi, f_KEf, f_int_type, KEbins):
    f_SIDini = []
    f_SIDend = []
    f_SIDint_ex = []
    f_Nevts = len(f_KEi)
    Nbins = len(KEbins)
    for ievt in range(f_Nevts):
        KE_ini = f_KEi[ievt]
        KE_end = f_KEf[ievt]
        intrcn = f_int_type[ievt]
        
        ## derive the initial slice ID
        for SID_ini in range(Nbins-1):
            if KEbins[SID_ini] < KE_ini:
                break
                
        ## derive the final slice ID
        for SID_end in range(Nbins-1):
            if KEbins[SID_end+1] < KE_end:
                break
        
        ## remove incomplete slices
        if SID_end < SID_ini:
            SID_ini = -1
            SID_end = -1
            
        ## derive the signal interaction slice ID
        SID_int_ex = SID_end
        if intrcn != 1:
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
    f_N3D_err = np.zeros(Nbins_3D)
    f_SID3D = np.zeros(f_Nevts, dtype=np.int32)
    
    for ievt in range(f_Nevts): # fill in the combined variable
        SID_ini = f_SIDini[ievt] + 1 # so that null bin moves to 0 (covenient for integer division and modulus below)
        SID_end = f_SIDend[ievt] + 1
        SID_int_ex = f_SIDint_ex[ievt] + 1
        weight = f_evtweight[ievt]
        
        SID3D = SID_ini + Nbins * SID_end + Nbins*Nbins * SID_int_ex # definition of the combined variable
        f_SID3D[ievt] = SID3D
        
        f_N3D[SID3D] += weight
        f_N3D_err[SID3D] += weight*weight
        
    f_N3D_Vcov = np.diag(f_N3D_err) # this is a fill process. Each bin is independent, so the covariance matrix is diagonal
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
    rho_tar = 1.4 # density (g/cm^3)
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
    with open('processedVars.pkl', 'rb') as procfile:
        processedVars = pickle.load(procfile)

    mask_TrueSignal = processedVars["mask_TrueSignal"]
    true_initial_energy = processedVars["true_initial_energy"]
    true_end_energy = processedVars["true_end_energy"]
    true_sigflag = processedVars["true_sigflag"]
    particle_type = processedVars["particle_type"]
    reweight = processedVars["reweight"]

    divided_trueEini, divided_weights = get_hists.divide_vars_by_partype(true_initial_energy, particle_type, mask=mask_TrueSignal, weight=reweight)
    divided_trueEend, divided_weights = get_hists.divide_vars_by_partype(true_end_energy, particle_type, mask=mask_TrueSignal, weight=reweight)
    divided_trueflag, divided_weights = get_hists.divide_vars_by_partype(true_sigflag, particle_type, mask=mask_TrueSignal, weight=reweight)
    true_Eini = divided_trueEini[0]
    true_Eend = divided_trueEend[0]
    true_flag = divided_trueflag[0]
    true_weight = divided_weights[0]
    print(len(true_Eini), true_Eini, true_Eend, true_flag, true_weight, sep='\n')

    true_bins = np.array([1000,950,900,850,800,750,700,650,600,550,500,450,400,350,300,250,200,150,100,50,0])
    Ntruebins, Ntruebins_3D, true_cKE, true_wKE = utils.set_bins(true_bins)
    true_SIDini, true_SIDend, true_SIDint_ex = get_sliceID_histograms(true_Eini, true_Eend, true_flag, true_bins)
    true_Nini, true_Nend, true_Nint_ex, true_Ninc = derive_energy_histograms(true_SIDini, true_SIDend, true_SIDint_ex, Ntruebins, true_weight)
    true_SID3D, true_N3D, true_N3D_Vcov = get_3D_histogram(true_SIDini, true_SIDend, true_SIDint_ex, Ntruebins, true_weight)
    true_3SID_Vcov = get_Cov_3SID_from_N3D(true_N3D_Vcov, Ntruebins)
    true_3N_Vcov = get_Cov_3N_from_3SID(true_3SID_Vcov, Ntruebins)
    true_XS, true_XS_Vcov = calculate_XS_Cov_from_3N(true_Ninc, true_Nend, true_Nint_ex, true_3N_Vcov, true_bins, BetheBloch(211))

    plt.figure(figsize=[8,4.8])
    XS_x = true_cKE[1:-1] # the underflow and overflow bin are not used
    XS_y = true_XS[1:-1]
    XS_xerr = true_wKE[1:-1]
    XS_yerr = np.sqrt(np.diagonal(true_XS_Vcov))[1:-1] # get the uncertainty from the covariance matrix
    plt.errorbar(XS_x, XS_y, XS_yerr, XS_xerr, fmt=".", label="Extracted true signal cross section")
    #xx = np.linspace(0, 1100, 100)
    #plt.plot(xx,XS_gen_ex(xx), label="Signal cross section used in simulation")
    plt.xlabel("Kinetic energy (MeV)")
    plt.ylabel("Cross section (mb)") # 1 mb = 10^{-27} cm^2
    plt.xlim([0,1000])
    plt.ylim(bottom=0)
    plt.show()

    plt.imshow(utils.transform_cov_to_corr_matrix(true_XS_Vcov[1:-1, 1:-1]), origin="lower", cmap="RdBu_r", vmin=-1, vmax=1, extent = [950,50,950,50])
    plt.title(r"Correlation matrix for cross section")
    plt.xticks([950,850,750,650,550,450,350,250,150,50])
    plt.yticks([950,850,750,650,550,450,350,250,150,50])
    plt.xlabel(r"Kinetic energy (MeV)")
    plt.ylabel(r"Kinetic energy (MeV)")
    plt.colorbar()
    plt.show()
