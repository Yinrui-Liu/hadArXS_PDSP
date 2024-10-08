from hadana.packages import *
import hadana.slicing_method as slicing
import hadana.multiD_mapping as multiD
from hadana.BetheBloch import BetheBloch
import hadana.parameters as parameters
import hadana.MC_reweight as reweight


beamPDG = 211
datafilename = "processed_files/procVars_pidata.pkl"
MCfilename = "processed_files/procVars_piMC.pkl"
# types of systematic uncertainties to include
bkg_scale = [1, 1, 1, 1, 1, 1, 1] # should be imported from sideband fit  pionp [0.93, 1, 1.72, 1.43, 0.93, 1, 1]  proton [1, 1, 1, 1, 1, 1, 1]
bkg_scale_err = [0, 0, 0, 0, 0, 0, 0] # pionp [0.12, 0, 0.13, 0.11, 0.12, 0, 0]  proton [0, 0, 0, 0, 0, 0, 0]
inc_sys_bkg = False
inc_sys_MCstat = False
inc_sys_MCXS = False
inc_sys_reweiP = False
inc_sys_Eloss = False
inc_sys_AltSCE = False
niter = 47 # 47 for pion data, 16 for proton data
save_xs_for_sys = False
sysmode = 'Eloss'
use_external_cov = False
use_total_cov = False


if beamPDG == 211:
    true_bins = parameters.true_bins_pionp
    meas_bins = parameters.meas_bins_pionp
elif beamPDG == 2212:
    true_bins = parameters.true_bins_proton
    meas_bins = parameters.meas_bins_proton

### load data histograms
with open(datafilename, 'rb') as datafile: # either fake data or real data
    processedVars = pickle.load(datafile)

mask_SelectedPart = processedVars["mask_SelectedPart"]
mask_FullSelection = processedVars["mask_FullSelection"]
combined_mask = mask_SelectedPart & mask_FullSelection
reco_initial_energy = processedVars["reco_initial_energy"]
reco_end_energy = processedVars["reco_end_energy"]
reco_sigflag = processedVars["reco_sigflag"]
reco_containing = processedVars["reco_containing"]
particle_type = processedVars["particle_type"]
weight_dt = processedVars["reweight"]


### selection
print("### selection")
divided_recoEini, divided_weights = utils.divide_vars_by_partype(reco_initial_energy, particle_type, mask=combined_mask, weight=weight_dt)
divided_recoEend, divided_weights = utils.divide_vars_by_partype(reco_end_energy, particle_type, mask=combined_mask, weight=weight_dt)
divided_recoflag, divided_weights = utils.divide_vars_by_partype(reco_sigflag, particle_type, mask=combined_mask, weight=weight_dt)
divided_recoisct, divided_weights = utils.divide_vars_by_partype(reco_containing, particle_type, mask=combined_mask, weight=weight_dt)
data_reco_Eini = divided_recoEini[0]
data_reco_Eend = divided_recoEend[0]
data_reco_flag = divided_recoflag[0]
data_reco_isCt = divided_recoisct[0]
data_reco_weight = divided_weights[0]

Ndata = len(data_reco_Eini)
Ntruebins, Ntruebins_3D, true_cKE, true_wKE = utils.set_bins(true_bins)
Nmeasbins = len(meas_bins)
data_meas_SIDini, data_meas_SIDend, data_meas_SIDint_ex = slicing.get_sliceID_histograms(data_reco_Eini, data_reco_Eend, data_reco_flag, data_reco_isCt, meas_bins)
data_meas_SID3D, data_meas_N3D, data_meas_N3D_Vcov = slicing.get_3D_histogram(data_meas_SIDini, data_meas_SIDend, data_meas_SIDint_ex, Nmeasbins, data_reco_weight)
data_meas_N3D_err = np.sqrt(np.diag(data_meas_N3D_Vcov))


### background subtraction
print("### background subtraction")
with open(MCfilename, 'rb') as mcfile: # truth MC
    processedVars_mc = pickle.load(mcfile)

mask_SelectedPart_mc = processedVars_mc["mask_SelectedPart"]
mask_FullSelection_mc = processedVars_mc["mask_FullSelection"]
combined_mask_mc = mask_SelectedPart_mc & mask_FullSelection_mc
reco_initial_energy_mc = processedVars_mc["reco_initial_energy"]
reco_end_energy_mc = processedVars_mc["reco_end_energy"]
reco_sigflag_mc = processedVars_mc["reco_sigflag"]
reco_containing_mc = processedVars_mc["reco_containing"]
particle_type_mc = processedVars_mc["particle_type"]
weight_mc = processedVars_mc["reweight"]
if inc_sys_Eloss:
    if beamPDG == 211:
        rdm_Eloss_shift = np.random.normal(0, 2.0) # 2.0 MeV constant error for pion upstream E loss
    elif beamPDG == 2212:
        rdm_Eloss_shift = np.random.normal(0, 0.6) # 0.6 MeV constant error for proton upstream E loss
    reco_initial_energy_mc = np.where(reco_initial_energy_mc>0, reco_initial_energy_mc - rdm_Eloss_shift, reco_initial_energy_mc)
    reco_end_energy_mc = np.where(reco_end_energy_mc>0, reco_end_energy_mc - rdm_Eloss_shift, reco_end_energy_mc)
if inc_sys_reweiP:
    rdm_reweiP_radius = np.random.normal(0, 1)
    rdm_reweiP_angle = np.random.uniform(0, 2*np.pi)
    weight_mc = reweight.cal_bkg_reweight(processedVars_mc) * reweight.cal_momentum_reweight(processedVars_mc, rdm_reweiP_radius, rdm_reweiP_angle)
if inc_sys_MCXS:
    rdm_MCXS = np.random.normal(1, 0.15) # assign 15% uncertainty for MC XS model
    weight_mc *= reweight.cal_g4rw(processedVars_mc, rdm_MCXS)
divided_recoEini_mc, divided_weights_mc = utils.divide_vars_by_partype(reco_initial_energy_mc, particle_type_mc, mask=combined_mask_mc, weight=weight_mc)
divided_recoEend_mc, divided_weights_mc = utils.divide_vars_by_partype(reco_end_energy_mc, particle_type_mc, mask=combined_mask_mc, weight=weight_mc)
divided_recoflag_mc, divided_weights_mc = utils.divide_vars_by_partype(reco_sigflag_mc, particle_type_mc, mask=combined_mask_mc, weight=weight_mc)
divided_recoisct_mc, divided_weights_mc = utils.divide_vars_by_partype(reco_containing_mc, particle_type_mc, mask=combined_mask_mc, weight=weight_mc)
Ntruemc = len(reco_initial_energy_mc[combined_mask_mc]) - len(divided_recoEini_mc[0])
bkg_meas_N3D_list = []
bkg_meas_N3D_err_list = []
for ibkg in range(3, len(divided_recoEini_mc)):
    bkg_reco_Eini = divided_recoEini_mc[ibkg]
    bkg_reco_Eend = divided_recoEend_mc[ibkg]
    bkg_reco_flag = divided_recoflag_mc[ibkg]
    bkg_reco_isCt = divided_recoisct_mc[ibkg]
    bkg_reco_weight = divided_weights_mc[ibkg]
    bkg_meas_SIDini, bkg_meas_SIDend, bkg_meas_SIDint_ex = slicing.get_sliceID_histograms(bkg_reco_Eini, bkg_reco_Eend, bkg_reco_flag, bkg_reco_isCt, meas_bins)
    bkg_meas_SID3D, bkg_meas_N3D, bkg_meas_N3D_Vcov = slicing.get_3D_histogram(bkg_meas_SIDini, bkg_meas_SIDend, bkg_meas_SIDint_ex, Nmeasbins, bkg_reco_weight)
    bkg_meas_N3D_list.append(bkg_meas_N3D)
    bkg_meas_N3D_err_list.append(np.sqrt(np.diag(bkg_meas_N3D_Vcov)))

print(f"Bkg scale \t{bkg_scale}\nError \t\t{bkg_scale_err}")
sig_meas_N3D, sig_meas_N3D_err = utils.bkg_subtraction(data_meas_N3D, data_meas_N3D_err, bkg_meas_N3D_list, bkg_meas_N3D_err_list, mc2data_scale=Ndata/Ntruemc, bkg_scale=bkg_scale, bkg_scale_err=bkg_scale_err, include_bkg_err=inc_sys_bkg)


### unfolding
print("### model response")
mask_TrueSignal_mc = processedVars_mc["mask_TrueSignal"]
combined_true_mask_mc = mask_SelectedPart_mc & mask_TrueSignal_mc
beam_matched_mc = processedVars_mc["reco_beam_true_byE_matched"]
combined_reco_mask_mc = mask_FullSelection_mc & np.array(beam_matched_mc, dtype=bool)
true_initial_energy_mc = processedVars_mc["true_initial_energy"]
true_end_energy_mc = processedVars_mc["true_end_energy"]
true_sigflag_mc = processedVars_mc["true_sigflag"]
true_containing_mc = processedVars_mc["true_containing"]

particle_type_bool_mc = np.where(particle_type_mc==0, 0, 1) # 0 for fake data, 1 for truth MC
divided_trueEini_mc, divided_weights_mc = utils.divide_vars_by_partype(true_initial_energy_mc, particle_type_bool_mc, mask=combined_true_mask_mc, weight=weight_mc)
divided_trueEend_mc, divided_weights_mc = utils.divide_vars_by_partype(true_end_energy_mc, particle_type_bool_mc, mask=combined_true_mask_mc, weight=weight_mc)
divided_trueflag_mc, divided_weights_mc = utils.divide_vars_by_partype(true_sigflag_mc, particle_type_bool_mc, mask=combined_true_mask_mc, weight=weight_mc)
divided_trueisct_mc, divided_weights_mc = utils.divide_vars_by_partype(true_containing_mc, particle_type_bool_mc, mask=combined_true_mask_mc, weight=weight_mc)
mc_true_Eini = divided_trueEini_mc[1]
mc_true_Eend = divided_trueEend_mc[1]
mc_true_flag = divided_trueflag_mc[1]
mc_true_isCt = divided_trueisct_mc[1]
mc_true_weight = divided_weights_mc[1]

mc_true_SIDini, mc_true_SIDend, mc_true_SIDint_ex = slicing.get_sliceID_histograms(mc_true_Eini, mc_true_Eend, mc_true_flag, mc_true_isCt, true_bins)
mc_true_Nini, mc_true_Nend, mc_true_Nint_ex, mc_true_Ninc = slicing.derive_energy_histograms(mc_true_SIDini, mc_true_SIDend, mc_true_SIDint_ex, Ntruebins, mc_true_weight)
mc_true_SID3D, mc_true_N3D, mc_true_N3D_Vcov = slicing.get_3D_histogram(mc_true_SIDini, mc_true_SIDend, mc_true_SIDint_ex, Ntruebins, mc_true_weight)

divided_recoEini_mc, divided_weights_mc = utils.divide_vars_by_partype(reco_initial_energy_mc, particle_type_bool_mc, mask=combined_true_mask_mc, weight=weight_mc)
divided_recoEend_mc, divided_weights_mc = utils.divide_vars_by_partype(reco_end_energy_mc, particle_type_bool_mc, mask=combined_true_mask_mc, weight=weight_mc)
divided_recoflag_mc, divided_weights_mc = utils.divide_vars_by_partype(reco_sigflag_mc, particle_type_bool_mc, mask=combined_true_mask_mc, weight=weight_mc)
divided_recoisct_mc, divided_weights_mc = utils.divide_vars_by_partype(reco_containing_mc, particle_type_bool_mc, mask=combined_true_mask_mc, weight=weight_mc)
divided_FullSelection_mc, divided_weights_mc = utils.divide_vars_by_partype(mask_FullSelection_mc, particle_type_bool_mc, mask=combined_true_mask_mc, weight=weight_mc)
pass_selection_mc = divided_FullSelection_mc[1]
mc_reco_Eini = divided_recoEini_mc[1][pass_selection_mc]
mc_reco_Eend = divided_recoEend_mc[1][pass_selection_mc]
mc_reco_flag = divided_recoflag_mc[1][pass_selection_mc]
mc_reco_isCt = divided_recoisct_mc[1][pass_selection_mc]
mc_reco_weight = divided_weights_mc[1][pass_selection_mc]
#print(len(mc_reco_Eini), mc_reco_Eini, mc_reco_Eend, mc_reco_flag, mc_reco_weight, mc_pass_selection, sep='\n')

mc_meas_SIDini, mc_meas_SIDend, mc_meas_SIDint_ex = slicing.get_sliceID_histograms(mc_reco_Eini, mc_reco_Eend, mc_reco_flag, mc_reco_isCt, meas_bins)
mc_meas_SID3D, mc_meas_N3D, mc_meas_N3D_Vcov = slicing.get_3D_histogram(mc_meas_SIDini, mc_meas_SIDend, mc_meas_SIDint_ex, Nmeasbins, mc_reco_weight)

true_3D1D_map, mc_true_N1D, mc_true_N1D_err, Ntruebins_1D = multiD.map_index_to_combined_variable(mc_true_N3D, np.sqrt(np.diag(mc_true_N3D_Vcov)), Ntruebins)
meas_3D1D_map, mc_meas_N1D, mc_meas_N1D_err, Nmeasbins_1D = multiD.map_index_to_combined_variable(mc_meas_N3D, np.sqrt(np.diag(mc_meas_N3D_Vcov)), Nmeasbins)
#print(true_3D1D_map, mc_true_N1D, mc_true_N1D_err, Ntruebins_1D, sep='\n')
#print(meas_3D1D_map, mc_meas_N1D, mc_meas_N1D_err, Nmeasbins_1D, sep='\n')

if inc_sys_MCstat:
    _, mc_true_N3D_nominal, mc_true_N3D_Vcov_nominal = slicing.get_3D_histogram(mc_true_SIDini, mc_true_SIDend, mc_true_SIDint_ex, Ntruebins, np.ones_like(mc_true_weight))
    _, mc_true_N1D_nominal, _, _ = multiD.map_index_to_combined_variable(mc_true_N3D_nominal, np.sqrt(np.diag(mc_true_N3D_Vcov_nominal)), Ntruebins)

    # To prepare the fluctuated eff1D and response_matrix. They should not be fluctuated separately, because response_matrix includes events passing selection, contributing to the numorator of eff1D. Thus, after fluctuating response_matrix as the histogram of *selected events* (numorator), we fluctuate the histogram of *missed events* (denominator - numorator), and then sum the two to get the histogram of *all events* (denominator).
    mc_true_SID3D_sel = mc_true_SID3D[pass_selection_mc]
    response_matrix_nominal, _ = multiD.get_response_matrix(Nmeasbins_1D, Ntruebins_1D, meas_3D1D_map[mc_meas_SID3D], true_3D1D_map[mc_true_SID3D_sel], np.ones_like(mc_reco_weight))
    response_matrix_fluc = np.random.poisson(response_matrix_nominal)
    miss_nominal = mc_true_N1D_nominal - np.sum(response_matrix_nominal, axis=1)
    miss_fluc = np.random.poisson(miss_nominal)
    mc_true_N1D_fluc = np.sum(response_matrix_fluc, axis=1) + miss_fluc

    eff1D, mc_true_SID3D_sel = multiD.get_efficiency(mc_true_N3D, mc_true_N1D * utils.safe_divide(mc_true_N1D_fluc, mc_true_N1D_nominal) * utils.safe_divide(np.sum(response_matrix_nominal, axis=1), np.sum(response_matrix_fluc, axis=1)), mc_true_SID3D, Ntruebins_3D, pass_selection_mc, mc_reco_weight)
    mc_reco_weight_fluc = np.ones_like(mc_reco_weight)
    for ievt in range(len(mc_reco_weight)):
        mc_reco_weight_fluc[ievt] = mc_reco_weight[ievt] * utils.safe_divide(response_matrix_fluc[true_3D1D_map[mc_true_SID3D_sel[ievt]]-1,meas_3D1D_map[mc_meas_SID3D[ievt]]-1], response_matrix_nominal[true_3D1D_map[mc_true_SID3D_sel[ievt]]-1,meas_3D1D_map[mc_meas_SID3D[ievt]]-1])
    response_matrix, response = multiD.get_response_matrix(Nmeasbins_1D, Ntruebins_1D, meas_3D1D_map[mc_meas_SID3D], true_3D1D_map[mc_true_SID3D_sel], mc_reco_weight_fluc)
    print(eff1D, response_matrix,sep='\n')
else:
    eff1D, mc_true_SID3D_sel = multiD.get_efficiency(mc_true_N3D, mc_true_N1D, mc_true_SID3D, Ntruebins_3D, pass_selection_mc, mc_reco_weight)
    response_matrix, response = multiD.get_response_matrix(Nmeasbins_1D, Ntruebins_1D, meas_3D1D_map[mc_meas_SID3D], true_3D1D_map[mc_true_SID3D_sel], mc_reco_weight)
    print(eff1D, response_matrix,sep='\n')

print("### unfolding")
sig_meas_N1D, sig_meas_N1D_err = multiD.map_data_to_MC_bins(sig_meas_N3D, sig_meas_N3D_err, meas_3D1D_map)
sig_meas_V1D = np.diag(sig_meas_N1D_err*sig_meas_N1D_err)
#print(sig_meas_N1D, sig_meas_N1D_err, sep='\n')

sig_MC_scale = sum(sig_meas_N1D)/sum(mc_meas_N3D)
sig_unfold, sig_unfold_cov = multiD.unfolding(sig_meas_N1D, sig_meas_V1D, response, niter=niter)
if inc_sys_MCstat:
    unfd_N3D, unfd_N3D_Vcov = multiD.efficiency_correct_1Dvar(sig_unfold, sig_unfold_cov, eff1D, true_3D1D_map, Ntruebins_3D, mc_true_N3D, mc_true_N3D_Vcov, sig_MC_scale, utils.safe_divide(mc_true_N1D_fluc, mc_true_N1D_nominal))
else:
    unfd_N3D, unfd_N3D_Vcov = multiD.efficiency_correct_1Dvar(sig_unfold, sig_unfold_cov, eff1D, true_3D1D_map, Ntruebins_3D, mc_true_N3D, mc_true_N3D_Vcov, sig_MC_scale)
unfd_Nini, unfd_Nend, unfd_Nint_ex, unfd_Ninc = multiD.get_unfold_histograms(unfd_N3D, Ntruebins)
#print(unfd_Nini, unfd_Nend, unfd_Nint_ex, unfd_Ninc, sep='\n')


### calculate cross section
unfd_3SID_Vcov = slicing.get_Cov_3SID_from_N3D(unfd_N3D_Vcov, Ntruebins)
unfd_3N_Vcov = slicing.get_Cov_3N_from_3SID(unfd_3SID_Vcov, Ntruebins)
unfd_XS, unfd_XS_Vcov = slicing.calculate_XS_Cov_from_3N(unfd_Ninc, unfd_Nend, unfd_Nint_ex, unfd_3N_Vcov, true_bins, BetheBloch(beamPDG))
systxtfile = f'syscovtxt/sys_{sysmode}_unfd_{beamPDG}.txt'
if use_external_cov:
    unfd_list = np.transpose(np.loadtxt(systxtfile))
    unfd_XS_Vcov = np.cov(unfd_list)
    np.savetxt(f'syscovtxt/sys_{sysmode}_Cov_{beamPDG}.txt', unfd_XS_Vcov)
    print(f"Using external covariance matrix from {systxtfile}.")
elif inc_sys_bkg:
    np.savetxt(f'syscovtxt/sys_incbkg_Cov_{beamPDG}.txt', unfd_XS_Vcov)
elif use_total_cov:
    unfd_XS_Vcov_tot = np.loadtxt(f'syscovtxt/sys_incbkg_Cov_{beamPDG}.txt') # stat + sys:bkg
    sysmode_list = ['MCstat', 'MCXS', 'reweiP', 'Eloss'] # sys:MCstat, MCXS, reweiP, Eloss
    for sysmd in sysmode_list:
        unfd_XS_Vcov_tot = unfd_XS_Vcov_tot + np.loadtxt(f'syscovtxt/sys_{sysmd}_Cov_{beamPDG}.txt')
    unfd_XS_Vcov_tot = unfd_XS_Vcov_tot + np.diag(np.power([0, 19.899637548224746, 9.831537758922877, 19.289352459702855, 16.218207745826703, 36.30653231021586, 3.8789110001945346, 38.822604035627364, 11.29462574327863, 0], 2)[::-1]) # sys:SCE (list copied from draw_sys_uncert.py)
    print("Use total covariance matrix")
print(f"Energy bin edges \t{true_bins[::-1]}\nMeasured cross section \t{unfd_XS[::-1].tolist()}\nUncertainty \t\t{np.sqrt(np.diag(unfd_XS_Vcov))[::-1].tolist()}")

if save_xs_for_sys:
    with open(systxtfile, "ab") as f:
        np.savetxt(f, [unfd_XS])
else:
    ### plot
    if beamPDG == 211:
        simcurvefile_name = "input_files/exclusive_xsec.root"
        simcurve_name = "total_inel_KE"
    elif beamPDG == 2212:
        simcurvefile_name = "input_files/proton_cross_section.root"
        simcurve_name = "inel_KE"
    simcurvefile = uproot.open(simcurvefile_name)
    simcurvegraph = simcurvefile[simcurve_name]
    simcurve = simcurvegraph.values()

    plt.figure(figsize=[8,4.8])
    XS_x = true_cKE[1:-1] # the underflow and overflow bin are not used
    XS_y = unfd_XS[1:-1]
    XS_xerr = true_wKE[1:-1]
    XS_yerr = np.sqrt(np.diagonal(unfd_XS_Vcov))[1:-1] # get the uncertainty from the covariance matrix
    plt.plot(*simcurve, 'r', label="Cross-section model used in simulation")
    XS_diff = XS_y - np.interp(XS_x, simcurve[0], simcurve[1])
    inv_XS_Vcov = np.linalg.pinv(unfd_XS_Vcov[1:-1, 1:-1])
    chi2 = np.einsum("i,ij,j->", XS_diff, inv_XS_Vcov, XS_diff)
    print(f"Chi2/Ndf = {chi2}/{len(XS_diff)}")
    plt.xlabel("Kinetic energy (MeV)")
    plt.ylabel("Cross section (mb)") # 1 mb = 10^{-27} cm^2
    plt.xlim(([true_bins[-1], true_bins[0]]))
    if use_total_cov and not use_external_cov: # show total uncertainties
        plt.errorbar(XS_x, XS_y, np.sqrt(np.diagonal(unfd_XS_Vcov_tot))[1:-1], XS_xerr, fmt="b.", capsize=2, label="Measured cross section (statistical+systematics)")
        plt.errorbar(XS_x, XS_y, XS_yerr, XS_xerr, fmt="k.", capsize=2, label="Measured cross section (statistical-only)")
        plt.ylim(bottom=0)
        cov_for_plot = unfd_XS_Vcov_tot[1:-1, 1:-1]
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [0,2,1]
        plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
        plt.savefig(f"plots/cross_section_{beamPDG}.pdf")
    elif inc_sys_AltSCE: # list copied from results using AltSCE
        plt.errorbar(XS_x, np.array([695.3743452575444, 605.2876906308809, 682.4889087910665, 607.9142803951634, 683.5686614053205, 567.6110260445066, 733.2202923432238, 560.353475275872])[::-1], np.array([50.83661577788303, 37.2799923711077, 32.31978472975942, 27.363311518973852, 28.287203736824203, 31.010980609972513, 48.736916840567304, 62.17338725560848])[::-1], XS_xerr, fmt="b.", label="Measured cross section (alternative SCE map)") # pionp
        #plt.errorbar(XS_x, np.array([701.2151284670073, 950.5339363423396, 653.4133084850553, 526.7722768717589, 501.3438129830147, 533.4376321697075, 503.6115947145777, 541.3808652623917, 753.7580326408548, 419.64607837824633])[::-1], np.array([85.62732548607192, 106.41692645096067, 36.39309483426517, 33.4415705888185, 24.18600414616621, 22.986900015686818, 25.497775866967064, 35.31580867006177, 108.43390486921116, 250.62438179586113])[::-1], XS_xerr, fmt="b.", label="Measured cross section (alternative SCE map)") # proton
        plt.legend()
        plt.savefig(f"plots/XS_sysSCE_{beamPDG}.pdf")
    else:
        plt.errorbar(XS_x, XS_y, XS_yerr, XS_xerr, fmt="k.", label="Measured cross section")
        plt.ylim(bottom=0)
        cov_for_plot = unfd_XS_Vcov[1:-1, 1:-1]
        plt.legend()
    plt.show()

    plt.pcolormesh(true_bins[1:-1], true_bins[1:-1], utils.transform_cov_to_corr_matrix(cov_for_plot), cmap="RdBu_r", vmin=-1, vmax=1) # true_bins is defined as reversed but on this plot it is increasing order, which is correct
    plt.title(r"Correlation matrix for cross section")
    plt.xticks(true_bins[1:-1])
    plt.yticks(true_bins[1:-1])
    plt.xlabel(r"Kinetic energy (MeV)")
    plt.ylabel(r"Kinetic energy (MeV)")
    plt.colorbar()
    if use_total_cov and not use_external_cov:
        plt.savefig(f"plots/XS_correlation_{beamPDG}.pdf")
    plt.show()