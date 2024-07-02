from packages import *
import calcXS
import get_hists
import parameters

def map_index_to_combined_variable(f_N3D, f_N3D_err, Nbins):
    Nbins_3D = Nbins**3
    ### mapping index from 3D to 1D
    f_3D1D_map = np.zeros(Nbins_3D, dtype=np.int32)
    tmpidx = 0
    for ibin in range(Nbins_3D):
        if f_N3D[ibin] > 0:
            tmpidx += 1
            f_3D1D_map[ibin] = tmpidx
    
    f_N1D = f_N3D[f_N3D>0]
    f_N1D_err = f_N3D_err[f_N3D>0]
    Nbins_1D = len(f_N1D)
    return f_3D1D_map, f_N1D, f_N1D_err, Nbins_1D

def get_efficiency(f_true_N3D, f_true_N1D, f_true_SID3D, Nbins_3D, pass_selection, weight):
    f_true_SID3D_sel = f_true_SID3D[pass_selection]
    f_true_N3D_sel, _ = np.histogram(f_true_SID3D_sel, bins=np.arange(Nbins_3D+1), weights=weight)

    f_eff1D = f_true_N3D_sel[f_true_N3D>0]/f_true_N1D
    return f_eff1D, f_true_SID3D_sel

def get_response_matrix(f_Nmeasbins, f_Ntruebins, f_meas_hist, f_true_hist, weight):
    f_response = ROOT.RooUnfoldResponse(f_Nmeasbins, 1, f_Nmeasbins+1, f_Ntruebins, 1, f_Ntruebins+1) # 1D index starts from 1
    for ievt in range(len(f_meas_hist)):
        f_response.Fill(f_meas_hist[ievt], f_true_hist[ievt], weight[ievt])
    f_response_matrix = np.zeros([f_Ntruebins, f_Nmeasbins])
    for ibin in range(f_Ntruebins):
        for jbin in range(f_Nmeasbins):
            f_response_matrix[ibin, jbin] = f_response.Hresponse().GetBinContent(jbin+1, ibin+1)
    return f_response_matrix, f_response

def map_data_to_MC_bins(f_N3D, f_N3D_err, f_3D1D_map):
    Nbins_3D = len(f_3D1D_map)
    for ibin in range(Nbins_3D):
        if f_N3D[ibin] > 0 and f_3D1D_map[ibin] == 0:
            print("Not empty in data but empty in MC.")
    f_N1D = f_N3D[f_3D1D_map>0]
    f_N1D_err = f_N3D_err[f_3D1D_map>0]
    return f_N1D, f_N1D_err

def unfolding(f_data_meas_N1D, f_data_meas_V1D, f_response, niter=4, Nmeasbins=None, Ntruebins=None):
    if Nmeasbins is None:
        Nmeasbins = f_response.Hresponse().GetNbinsX()
    if Ntruebins is None:
        Ntruebins = f_response.Hresponse().GetNbinsY()
    hMeas = ROOT.TH1D ("hmeas", "", Nmeasbins, 0, Nmeasbins)
    for ibin in range(Nmeasbins):
        hMeas.SetBinContent(ibin+1, f_data_meas_N1D[ibin])
    uf = ROOT.RooUnfoldBayes(f_response, hMeas, niter=niter)
    data_meas_V1D_TM = ROOT.TMatrix(Nmeasbins, Nmeasbins)
    for ibin in range(Nmeasbins):
        for jbin in range(Nmeasbins):
            data_meas_V1D_TM[ibin, jbin] = f_data_meas_V1D[ibin, jbin]
    uf.SetMeasuredCov(data_meas_V1D_TM)
    
    hUnfold = uf.Hunfold()
    VUnfold = uf.Eunfold()
    
    f_data_unfold = np.zeros(Ntruebins)
    f_data_unfold_cov = np.zeros([Ntruebins, Ntruebins])
    for ibin in range(Ntruebins):
        f_data_unfold[ibin] = hUnfold.GetBinContent(ibin+1)
        for jbin in range(Ntruebins):
            f_data_unfold_cov[ibin, jbin] = VUnfold[ibin, jbin]
    return f_data_unfold, f_data_unfold_cov

def efficiency_correct_1Dvar(f_data_unfold, f_data_unfold_cov, f_eff1D, f_true_3D1D_map, f_Ntruebins_3D, f_true_N3D, f_true_N3D_Vcov, f_data_MC_scale):
    f_unfd_N3D = np.zeros(f_Ntruebins_3D)
    f_unfd_N3D_Vcov = np.zeros([f_Ntruebins_3D, f_Ntruebins_3D])
    for ibin in range(f_Ntruebins_3D):
        idx_1D_i = f_true_3D1D_map[ibin]
        if idx_1D_i > 0:
            if f_data_unfold[idx_1D_i-1] > 0:
                f_unfd_N3D[ibin] = f_data_unfold[idx_1D_i-1] / f_eff1D[idx_1D_i-1]
                for jbin in range(f_Ntruebins_3D):
                    idx_1D_j = f_true_3D1D_map[jbin]
                    if idx_1D_j > 0 and f_data_unfold[idx_1D_j-1] > 0:
                        f_unfd_N3D_Vcov[ibin, jbin] = f_data_unfold_cov[idx_1D_i-1, idx_1D_j-1] / (f_eff1D[idx_1D_i-1]*f_eff1D[idx_1D_j-1])
            elif f_eff1D[idx_1D_i-1] == 0:
                #print(data_unfold[true_3D1D_map[ibin]-1], true_N1D[true_3D1D_map[ibin]-1])
                f_unfd_N3D[ibin] = f_true_N3D[ibin]*f_data_MC_scale
                f_unfd_N3D_Vcov[ibin, ibin] = f_true_N3D_Vcov[ibin, ibin]*f_data_MC_scale*f_data_MC_scale
    #f_unfd_N3D_err = np.sqrt(np.diag(f_unfd_N3D_Vcov))
    return f_unfd_N3D, f_unfd_N3D_Vcov

def get_unfold_histograms(f_unfd_N3D, f_Ntruebins):
    f_unfd_N3D_real3D = np.reshape(f_unfd_N3D, [f_Ntruebins, f_Ntruebins, f_Ntruebins])
    f_unfd_Nini = f_unfd_N3D_real3D.sum((0,1))[1:]
    f_unfd_Nend = f_unfd_N3D_real3D.sum((0,2))[1:]
    f_unfd_Nint_ex = f_unfd_N3D_real3D.sum((1,2))[1:]
    f_unfd_Ninc = np.zeros_like(f_unfd_Nini)
    for ibin in range(f_Ntruebins-1):
        ## two equivalent way to calculate the incident histogram
        for itmp in range(0, ibin+1):
            f_unfd_Ninc[ibin] += f_unfd_Nini[itmp]
        for itmp in range(0, ibin):
            f_unfd_Ninc[ibin] -= f_unfd_Nend[itmp]
        '''for itmp in range(ibin, f_Ntruebins-1):
            f_unfd_Ninc[ibin] += f_unfd_Nend[itmp]
        for itmp in range(ibin+1, f_Ntruebins-1):
            f_unfd_Ninc[ibin] -= f_unfd_Nini[itmp]'''
    return f_unfd_Nini, f_unfd_Nend, f_unfd_Nint_ex, f_unfd_Ninc


if __name__ == "__main__":
    beamPDG = 2212
    with open('processedVars_pMC.pkl', 'rb') as procfile:
        processedVars = pickle.load(procfile)

    if beamPDG == 211:
        true_bins = parameters.true_bins_pionp
        meas_bins = parameters.meas_bins_pionp
    elif beamPDG == 2212:
        true_bins = parameters.true_bins_proton
        meas_bins = parameters.meas_bins_proton

    mask_TrueSignal = processedVars["mask_TrueSignal"]
    mask_SelectedPart = processedVars["mask_SelectedPart"]
    combined_true_mask = mask_SelectedPart & mask_TrueSignal
    mask_FullSelection = processedVars["mask_FullSelection"]
    beam_matched = processedVars["reco_beam_true_byE_matched"]
    combined_reco_mask = mask_FullSelection & np.array(beam_matched, dtype=bool)
    reco_initial_energy = processedVars["reco_initial_energy"]
    reco_end_energy = processedVars["reco_end_energy"]
    reco_sigflag = processedVars["reco_sigflag"]
    particle_type = processedVars["particle_type"]
    reweight = processedVars["reweight"]
    
    true_initial_energy = processedVars["true_initial_energy"]
    true_end_energy = processedVars["true_end_energy"]
    true_sigflag = processedVars["true_sigflag"]

    particle_type_bool = np.where(particle_type==0, 0, 1) # 0 for fake data, 1 for truth MC
    divided_trueEini, divided_weights = get_hists.divide_vars_by_partype(true_initial_energy, particle_type_bool, mask=combined_true_mask, weight=reweight)
    divided_trueEend, divided_weights = get_hists.divide_vars_by_partype(true_end_energy, particle_type_bool, mask=combined_true_mask, weight=reweight)
    divided_trueflag, divided_weights = get_hists.divide_vars_by_partype(true_sigflag, particle_type_bool, mask=combined_true_mask, weight=reweight)
    true_Eini = divided_trueEini[1]
    true_Eend = divided_trueEend[1]
    true_flag = divided_trueflag[1]
    true_weight = divided_weights[1]
    Ntruebins, Ntruebins_3D, true_cKE, true_wKE = utils.set_bins(true_bins)
    Nmeasbins = len(meas_bins)
    true_SIDini, true_SIDend, true_SIDint_ex = calcXS.get_sliceID_histograms(true_Eini, true_Eend, true_flag, true_bins)
    true_Nini, true_Nend, true_Nint_ex, true_Ninc = calcXS.derive_energy_histograms(true_SIDini, true_SIDend, true_SIDint_ex, Ntruebins, true_weight)
    true_SID3D, true_N3D, true_N3D_Vcov = calcXS.get_3D_histogram(true_SIDini, true_SIDend, true_SIDint_ex, Ntruebins, true_weight)

    divided_recoEini, divided_weights = get_hists.divide_vars_by_partype(reco_initial_energy, particle_type_bool, mask=combined_true_mask, weight=reweight)
    divided_recoEend, divided_weights = get_hists.divide_vars_by_partype(reco_end_energy, particle_type_bool, mask=combined_true_mask, weight=reweight)
    divided_recoflag, divided_weights = get_hists.divide_vars_by_partype(reco_sigflag, particle_type_bool, mask=combined_true_mask, weight=reweight)
    divided_FullSelection, divided_weights = get_hists.divide_vars_by_partype(mask_FullSelection, particle_type_bool, mask=combined_true_mask, weight=reweight)
    pass_selection = divided_FullSelection[1]
    reco_Eini = divided_recoEini[1][pass_selection]
    reco_Eend = divided_recoEend[1][pass_selection]
    reco_flag = divided_recoflag[1][pass_selection]
    reco_weight = divided_weights[1][pass_selection]
    #print(len(reco_Eini), reco_Eini, reco_Eend, reco_flag, reco_weight, pass_selection, sep='\n')

    meas_SIDini, meas_SIDend, meas_SIDint_ex = calcXS.get_sliceID_histograms(reco_Eini, reco_Eend, reco_flag, meas_bins)
    meas_SID3D, meas_N3D, meas_N3D_Vcov = calcXS.get_3D_histogram(meas_SIDini, meas_SIDend, meas_SIDint_ex, Nmeasbins, reco_weight)

    true_3D1D_map, true_N1D, true_N1D_err, Ntruebins_1D = map_index_to_combined_variable(true_N3D, np.sqrt(np.diag(true_N3D_Vcov)), Ntruebins)
    meas_3D1D_map, meas_N1D, meas_N1D_err, Nmeasbins_1D = map_index_to_combined_variable(meas_N3D, np.sqrt(np.diag(meas_N3D_Vcov)), Nmeasbins)
    #print(true_3D1D_map, true_N1D, true_N1D_err, Ntruebins_1D, sep='\n')
    #print(meas_3D1D_map, meas_N1D, meas_N1D_err, Nmeasbins_1D, sep='\n')

    eff1D, true_SID3D_sel = get_efficiency(true_N3D, true_N1D, true_SID3D, Ntruebins_3D, pass_selection, reco_weight)
    response_matrix, response = get_response_matrix(Nmeasbins_1D, Ntruebins_1D, meas_3D1D_map[meas_SID3D], true_3D1D_map[true_SID3D_sel], reco_weight)
    #print(eff1D, response_matrix, sep='\n')

    '''eff1D_upperr = []
    eff1D_lowerr = []
    for ii in range(len(eff1D)):
        # disable error calculation since ClopperPearson does not include weights
        eff1D_upperr.append(0)#ROOT.TEfficiency.ClopperPearson(true_N1D[ii], true_N3D_sel[true_N3D>0][ii], 0.6826894921, True) - eff1D[ii])
        eff1D_lowerr.append(0)#eff1D[ii] - ROOT.TEfficiency.ClopperPearson(true_N1D[ii], true_N3D_sel[true_N3D>0][ii], 0.6826894921, False))
    plt.figure(figsize=[8.6,4.8])
    plt.errorbar(np.arange(0, Ntruebins_1D), eff1D, yerr=[eff1D_lowerr, eff1D_upperr], fmt="r.", ecolor="g", elinewidth=1, label="Efficiency")
    plt.xlabel(r"${\rm ID_{rem}}$")
    plt.ylabel("Efficiency")
    plt.title(r"Efficiency for true ${\rm ID_{rem}}$")
    plt.ylim([0,1])
    plt.legend()
    plt.show()

    plt.imshow(np.ma.masked_where(response_matrix == 0, response_matrix), origin="lower")
    plt.title(r"Response matrix for ${\rm ID_{rem}}$")
    plt.xlabel(r"Measured ${\rm ID_{rem}}$")
    plt.ylabel(r"True ${\rm ID_{rem}}$")
    #plt.xlim([-10,Nmeasbins_1D+10])
    #plt.ylim([-10,Ntruebins_1D+10])
    plt.colorbar(label="Counts")
    plt.show()'''

    with open('response.pkl', 'wb') as respfile: # save the response modeled by MC
        responseVars = {}
        responseVars["response_matrix"] = response.Hresponse()
        responseVars["response_truth"] = response.Htruth()
        responseVars["response_measured"] = response.Hmeasured()
        responseVars["eff1D"] = eff1D
        responseVars["meas_3D1D_map"] = meas_3D1D_map
        responseVars["true_3D1D_map"] = true_3D1D_map
        responseVars["true_N3D"] = true_N3D
        responseVars["true_N3D_Vcov"] = true_N3D_Vcov
        responseVars["meas_N3D"] = meas_N3D
        responseVars["meas_N3D_Vcov"] = meas_N3D_Vcov
        pickle.dump(responseVars, respfile)
