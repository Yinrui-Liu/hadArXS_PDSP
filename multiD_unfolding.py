from packages import *
import calcXS
import get_hists

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

if __name__ == "__main__":
    with open('processedVars.pkl', 'rb') as procfile:
        processedVars = pickle.load(procfile)

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
    true_bins = np.array([1000, 900, 850, 800, 750, 700, 650, 600, 550, 500, 0])
    Ntruebins, Ntruebins_3D, true_cKE, true_wKE = utils.set_bins(true_bins)
    meas_bins = np.array([1000, 900, 850, 800, 750, 700, 650, 600, 550, 500, 0])
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
    print(eff1D, response_matrix, sep='\n')
    