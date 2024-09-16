from hadana.packages import *
import ROOT
import hadana.slicing_method as slicing
import hadana.multiD_mapping as multiD
from hadana.BetheBloch import BetheBloch
import hadana.parameters as parameters


beamPDG = 211
datafilename = "processed_files/procVars_pidata.pkl"
MCfilename = "processed_files/procVars_piMC.pkl"
resfilename = "processed_files/response_pi.pkl"
# types of systematic uncertainties to include
bkg_scale = [1, 1, 1, 1, 1, 1, 1] # should be imported from sideband fit  pionp [0.93, 1, 1.72, 1.43, 0.93, 1, 1]  proton [1, 1, 1, 1, 1, 1, 1]
bkg_scale_err = [0, 0, 0, 0, 0, 0, 0] # pionp [0.12, 0, 0.13, 0.11, 0.12, 0, 0]  proton [0, 0, 0, 0, 0, 0, 0]
inc_sys_bkg = False
plot_energy_hists = False
niter = 20 # 47 for pion data, 16 for proton data, 18 for pion fd, 14 for proton fd

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
Nmeasbins, Nmeasbins_3D, meas_cKE, meas_wKE = utils.set_bins(meas_bins)
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
print("### unfolding")
with open(resfilename, 'rb') as respfile: # unfolding vars modeled by MC
    responseVars = pickle.load(respfile)
response_matrix = responseVars["response_matrix"]
response_truth = responseVars["response_truth"]
response_measured = responseVars["response_measured"]
response = ROOT.RooUnfoldResponse(response_measured, response_truth, response_matrix) # RooUnfoldResponse is a complicated type and hard for serialization. If we directly saved it by pickle or uproot, information may be lost. Thus, we load the essential THists (Hmeasured, Htruth, Hresponse) to re-construct response.
eff1D = responseVars["eff1D"]
meas_3D1D_map = responseVars["meas_3D1D_map"]
true_3D1D_map = responseVars["true_3D1D_map"]
true_N3D = responseVars["true_N3D"]
true_N3D_Vcov = responseVars["true_N3D_Vcov"]
meas_N3D = responseVars["meas_N3D"]
meas_N3D_Vcov = responseVars["meas_N3D_Vcov"]

sig_meas_N1D, sig_meas_N1D_err = multiD.map_data_to_MC_bins(sig_meas_N3D, sig_meas_N3D_err, meas_3D1D_map)
sig_meas_V1D = np.diag(sig_meas_N1D_err*sig_meas_N1D_err)
#print(sig_meas_N1D, sig_meas_N1D_err, sep='\n')

sig_MC_scale = sum(sig_meas_N1D)/sum(meas_N3D)
sig_unfold, sig_unfold_cov = multiD.unfolding(sig_meas_N1D, sig_meas_V1D, response, niter=niter)
unfd_N3D, unfd_N3D_Vcov = multiD.efficiency_correct_1Dvar(sig_unfold, sig_unfold_cov, eff1D, true_3D1D_map, Ntruebins_3D, true_N3D, true_N3D_Vcov, sig_MC_scale)
unfd_Nini, unfd_Nend, unfd_Nint_ex, unfd_Ninc = multiD.get_unfold_histograms(unfd_N3D, Ntruebins)
#print(unfd_Nini, unfd_Nend, unfd_Nint_ex, unfd_Ninc, sep='\n')

### calculate cross section
unfd_3SID_Vcov = slicing.get_Cov_3SID_from_N3D(unfd_N3D_Vcov, Ntruebins)
unfd_3N_Vcov = slicing.get_Cov_3N_from_3SID(unfd_3SID_Vcov, Ntruebins)
bb = BetheBloch(beamPDG)
unfd_XS, unfd_XS_Vcov = slicing.calculate_XS_Cov_from_3N(unfd_Ninc, unfd_Nend, unfd_Nint_ex, unfd_3N_Vcov, true_bins, bb)
print(f"Measured cross section \t{unfd_XS}\nUncertainty \t\t{np.sqrt(np.diag(unfd_XS_Vcov))}")

if plot_energy_hists:
    from matplotlib.patches import Patch

    unfd_Nini_err = np.sqrt(np.diagonal(unfd_3SID_Vcov)[1:Ntruebins])
    unfd_Ninc_err = np.sqrt(np.diagonal(unfd_3N_Vcov)[:Ntruebins-1])
    unfd_Nend_err = np.sqrt(np.diagonal(unfd_3N_Vcov)[Ntruebins-1:2*(Ntruebins-1)])
    unfd_Nint_ex_err = np.sqrt(np.diagonal(unfd_3N_Vcov)[2*(Ntruebins-1):])
    
    meas_Nini, meas_Nend, meas_Nint_ex, meas_Ninc = multiD.get_unfold_histograms(sig_meas_N3D, Nmeasbins)
    meas_3SID_Vcov = slicing.get_Cov_3SID_from_N3D(np.diag(sig_meas_N3D_err*sig_meas_N3D_err), Nmeasbins)
    meas_3N_Vcov = slicing.get_Cov_3N_from_3SID(meas_3SID_Vcov, Nmeasbins)
    meas_XS, meas_XS_Vcov = slicing.calculate_XS_Cov_from_3N(meas_Ninc, meas_Nend, meas_Nint_ex, meas_3N_Vcov, meas_bins, bb)
    meas_Nini_err = np.sqrt(np.diagonal(meas_3SID_Vcov)[1:Nmeasbins])
    meas_Ninc_err = np.sqrt(np.diagonal(meas_3N_Vcov)[:Nmeasbins-1])
    meas_Nend_err = np.sqrt(np.diagonal(meas_3N_Vcov)[Nmeasbins-1:2*(Nmeasbins-1)])
    meas_Nint_ex_err = np.sqrt(np.diagonal(meas_3N_Vcov)[2*(Nmeasbins-1):])

    mctrue_Nini, mctrue_Nend, mctrue_Nint_ex, mctrue_Ninc = multiD.get_unfold_histograms(true_N3D, Ntruebins)
    mctrue_3SID_Vcov = slicing.get_Cov_3SID_from_N3D(true_N3D_Vcov, Ntruebins)
    mctrue_3N_Vcov = slicing.get_Cov_3N_from_3SID(mctrue_3SID_Vcov, Ntruebins)
    mctrue_XS, mctrue_XS_Vcov = slicing.calculate_XS_Cov_from_3N(mctrue_Ninc, mctrue_Nend, mctrue_Nint_ex, mctrue_3N_Vcov, true_bins, bb)
    mctrue_Nini_err = np.sqrt(np.diagonal(mctrue_3SID_Vcov)[1:Ntruebins])
    mctrue_Ninc_err = np.sqrt(np.diagonal(mctrue_3N_Vcov)[:Ntruebins-1])
    mctrue_Nend_err = np.sqrt(np.diagonal(mctrue_3N_Vcov)[Ntruebins-1:2*(Ntruebins-1)])
    mctrue_Nint_ex_err = np.sqrt(np.diagonal(mctrue_3N_Vcov)[2*(Ntruebins-1):])

    mcmeas_Nini, mcmeas_Nend, mcmeas_Nint_ex, mcmeas_Ninc = multiD.get_unfold_histograms(meas_N3D, Nmeasbins)
    mcmeas_3SID_Vcov = slicing.get_Cov_3SID_from_N3D(meas_N3D_Vcov, Nmeasbins)
    mcmeas_3N_Vcov = slicing.get_Cov_3N_from_3SID(mcmeas_3SID_Vcov, Nmeasbins)
    mcmeas_XS, mcmeas_XS_Vcov = slicing.calculate_XS_Cov_from_3N(mcmeas_Ninc, mcmeas_Nend, mcmeas_Nint_ex, mcmeas_3N_Vcov, meas_bins, bb)
    mcmeas_Nini_err = np.sqrt(np.diagonal(mcmeas_3SID_Vcov)[1:Nmeasbins])
    mcmeas_Ninc_err = np.sqrt(np.diagonal(mcmeas_3N_Vcov)[:Nmeasbins-1])
    mcmeas_Nend_err = np.sqrt(np.diagonal(mcmeas_3N_Vcov)[Nmeasbins-1:2*(Nmeasbins-1)])
    mcmeas_Nint_ex_err = np.sqrt(np.diagonal(mcmeas_3N_Vcov)[2*(Nmeasbins-1):])

    fig, axs = plt.subplots(2, 2, figsize=[10, 8])
    plt.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.06, hspace=0.2, wspace=0.2)
    # plot Nini
    axs[0, 0].errorbar(meas_cKE, meas_Nini, meas_Nini_err, meas_wKE, color="r", fmt=".", label="Data measured")
    axs[0, 0].errorbar(true_cKE, unfd_Nini, unfd_Nini_err, true_wKE, color="b", fmt=".", label="Data unfolded")
    histy_center = mctrue_Nini*sig_MC_scale
    line_t, = axs[0, 0].step(np.concatenate([true_bins, [true_bins[-1]]]), np.concatenate([[0], histy_center, [0]]), "b-", alpha=0.8, lw=1) # MC true
    axs[0, 0].fill_between(np.repeat(true_bins, 2)[1:-1], np.repeat(histy_center-mctrue_Nini_err, 2), np.repeat(histy_center+mctrue_Nini_err, 2), color='blue', alpha=0.3, edgecolor='none')
    histy_center = mcmeas_Nini*sig_MC_scale
    line_m, = axs[0, 0].step(np.concatenate([meas_bins, [meas_bins[-1]]]), np.concatenate([[0], histy_center, [0]]), "r-", alpha=0.8, lw=1) # MC reco
    axs[0, 0].fill_between(np.repeat(meas_bins, 2)[1:-1], np.repeat(histy_center-mcmeas_Nini_err, 2), np.repeat(histy_center+mcmeas_Nini_err, 2), color='red', alpha=0.3, edgecolor='none')
    axs[0, 0].set_title(r"Initial histogram $N_{\rm ini}$")
    axs[0, 0].set_xlabel("Kinetic energy (MeV)")
    axs[0, 0].set_ylabel("Counts")
    axs[0, 0].set_xlim([true_bins[-1], true_bins[0]])
    axs[0, 0].set_ylim(bottom=0)
    # customize legend element
    le_t = [line_t, Patch(facecolor='blue', edgecolor='none', alpha=0.3)]
    le_m = [line_m, Patch(facecolor='red', edgecolor='none', alpha=0.3)]
    handles, labels = axs[0, 0].get_legend_handles_labels()
    handles.append((le_t[0], le_t[1]))
    labels.append("MC true")
    handles.append((le_m[0], le_m[1]))
    labels.append("MC measured")
    axs[0, 0].legend(handles, labels)

    # plot Nend
    axs[0, 1].errorbar(meas_cKE, meas_Nend, meas_Nend_err, meas_wKE, color="r", fmt=".", label="Data measured")
    axs[0, 1].errorbar(true_cKE, unfd_Nend, unfd_Nend_err, true_wKE, color="b", fmt=".", label="Data unfolded")
    histy_center = mctrue_Nend*sig_MC_scale
    line_t, = axs[0, 1].step(np.concatenate([true_bins, [true_bins[-1]]]), np.concatenate([[0], histy_center, [0]]), "b-", alpha=0.8, lw=1) # MC true
    axs[0, 1].fill_between(np.repeat(true_bins, 2)[1:-1], np.repeat(histy_center-mctrue_Nend_err, 2), np.repeat(histy_center+mctrue_Nend_err, 2), color='blue', alpha=0.3, edgecolor='none')
    histy_center = mcmeas_Nend*sig_MC_scale
    line_m, = axs[0, 1].step(np.concatenate([meas_bins, [meas_bins[-1]]]), np.concatenate([[0], histy_center, [0]]), "r-", alpha=0.8, lw=1) # MC reco
    axs[0, 1].fill_between(np.repeat(meas_bins, 2)[1:-1], np.repeat(histy_center-mcmeas_Nend_err, 2), np.repeat(histy_center+mcmeas_Nend_err, 2), color='red', alpha=0.3, edgecolor='none')
    axs[0, 1].set_title(r"End histogram $N_{\rm end}$")
    axs[0, 1].set_xlabel("Kinetic energy (MeV)")
    axs[0, 1].set_ylabel("Counts")
    axs[0, 1].set_xlim([true_bins[-1], true_bins[0]])
    axs[0, 1].set_ylim(bottom=0)
    # customize legend element
    le_t = [line_t, Patch(facecolor='blue', edgecolor='none', alpha=0.3)]
    le_m = [line_m, Patch(facecolor='red', edgecolor='none', alpha=0.3)]
    handles, labels = axs[0, 1].get_legend_handles_labels()
    handles.append((le_t[0], le_t[1]))
    labels.append("MC true")
    handles.append((le_m[0], le_m[1]))
    labels.append("MC measured")
    axs[0, 1].legend(handles, labels)

    # plot Nint_ex
    axs[1, 0].errorbar(meas_cKE, meas_Nint_ex, meas_Nint_ex_err, meas_wKE, color="r", fmt=".", label="Data measured")
    axs[1, 0].errorbar(true_cKE, unfd_Nint_ex, unfd_Nint_ex_err, true_wKE, color="b", fmt=".", label="Data unfolded")
    histy_center = mctrue_Nint_ex*sig_MC_scale
    line_t, = axs[1, 0].step(np.concatenate([true_bins, [true_bins[-1]]]), np.concatenate([[0], histy_center, [0]]), "b-", alpha=0.8, lw=1) # MC true
    axs[1, 0].fill_between(np.repeat(true_bins, 2)[1:-1], np.repeat(histy_center-mctrue_Nint_ex_err, 2), np.repeat(histy_center+mctrue_Nint_ex_err, 2), color='blue', alpha=0.3, edgecolor='none')
    histy_center = mcmeas_Nint_ex*sig_MC_scale
    line_m, = axs[1, 0].step(np.concatenate([meas_bins, [meas_bins[-1]]]), np.concatenate([[0], histy_center, [0]]), "r-", alpha=0.8, lw=1) # MC reco
    axs[1, 0].fill_between(np.repeat(meas_bins, 2)[1:-1], np.repeat(histy_center-mcmeas_Nint_ex_err, 2), np.repeat(histy_center+mcmeas_Nint_ex_err, 2), color='red', alpha=0.3, edgecolor='none')
    axs[1, 0].set_title(r"Interaction histogram $N_{\rm int_{ex}}$")
    axs[1, 0].set_xlabel("Kinetic energy (MeV)")
    axs[1, 0].set_ylabel("Counts")
    axs[1, 0].set_xlim([true_bins[-1], true_bins[0]])
    axs[1, 0].set_ylim(bottom=0)
    # customize legend element
    le_t = [line_t, Patch(facecolor='blue', edgecolor='none', alpha=0.3)]
    le_m = [line_m, Patch(facecolor='red', edgecolor='none', alpha=0.3)]
    handles, labels = axs[1, 0].get_legend_handles_labels()
    handles.append((le_t[0], le_t[1]))
    labels.append("MC true")
    handles.append((le_m[0], le_m[1]))
    labels.append("MC measured")
    axs[1, 0].legend(handles, labels)

    # plot Ninc
    axs[1, 1].errorbar(meas_cKE, meas_Ninc, meas_Ninc_err, meas_wKE, color="r", fmt=".", label="Data measured")
    axs[1, 1].errorbar(true_cKE, unfd_Ninc, unfd_Ninc_err, true_wKE, color="b", fmt=".", label="Data unfolded")
    histy_center = mctrue_Ninc*sig_MC_scale
    line_t, = axs[1, 1].step(np.concatenate([true_bins, [true_bins[-1]]]), np.concatenate([[0], histy_center, [0]]), "b-", alpha=0.8, lw=1) # MC true
    axs[1, 1].fill_between(np.repeat(true_bins, 2)[1:-1], np.repeat(histy_center-mctrue_Ninc_err, 2), np.repeat(histy_center+mctrue_Ninc_err, 2), color='blue', alpha=0.3, edgecolor='none')
    histy_center = mcmeas_Ninc*sig_MC_scale
    line_m, = axs[1, 1].step(np.concatenate([meas_bins, [meas_bins[-1]]]), np.concatenate([[0], histy_center, [0]]), "r-", alpha=0.8, lw=1) # MC reco
    axs[1, 1].fill_between(np.repeat(meas_bins, 2)[1:-1], np.repeat(histy_center-mcmeas_Ninc_err, 2), np.repeat(histy_center+mcmeas_Ninc_err, 2), color='red', alpha=0.3, edgecolor='none')
    axs[1, 1].set_title(r"Incident histogram $N_{\rm inc}$")
    axs[1, 1].set_xlabel("Kinetic energy (MeV)")
    axs[1, 1].set_ylabel("Counts")
    axs[1, 1].set_xlim([true_bins[-1], true_bins[0]])
    axs[1, 1].set_ylim(bottom=0)
    # customize legend element
    le_t = [line_t, Patch(facecolor='blue', edgecolor='none', alpha=0.3)]
    le_m = [line_m, Patch(facecolor='red', edgecolor='none', alpha=0.3)]
    handles, labels = axs[1, 1].get_legend_handles_labels()
    handles.append((le_t[0], le_t[1]))
    labels.append("MC true")
    handles.append((le_m[0], le_m[1]))
    labels.append("MC measured")
    axs[1, 1].legend(handles, labels)

    plt.savefig(f'plots/energy_hists_{beamPDG}.pdf')
    plt.show()

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
plt.errorbar(XS_x, XS_y, XS_yerr, XS_xerr, fmt=".", label="Signal XS using unfolded result")
#xx = np.linspace(0, 1100, 100)
#plt.plot(xx,XS_gen_ex(xx), label="Signal cross section used in simulation")
plt.plot(*simcurve, label="Signal cross section used in simulation")
XS_diff = XS_y - np.interp(XS_x, simcurve[0], simcurve[1])
inv_XS_Vcov = np.linalg.pinv(unfd_XS_Vcov[1:-1, 1:-1])
chi2 = np.einsum("i,ij,j->", XS_diff, inv_XS_Vcov, XS_diff)
print(f"Chi2/Ndf = {chi2}/{len(XS_diff)}")
plt.xlabel("Kinetic energy (MeV)")
plt.ylabel("Cross section (mb)") # 1 mb = 10^{-27} cm^2
plt.xlim(([true_bins[-1], true_bins[0]]))
plt.ylim(bottom=0)
plt.legend()
plt.savefig(f"plots/XSmeas_{beamPDG}.pdf")
plt.show()

plt.pcolormesh(true_bins[1:-1], true_bins[1:-1], utils.transform_cov_to_corr_matrix(unfd_XS_Vcov[1:-1, 1:-1]), cmap="RdBu_r", vmin=-1, vmax=1)
plt.title(r"Correlation matrix for cross section")
plt.xticks(true_bins[1:-1])
plt.yticks(true_bins[1:-1])
plt.xlabel(r"Kinetic energy (MeV)")
plt.ylabel(r"Kinetic energy (MeV)")
plt.colorbar()
plt.savefig(f"plots/XSmeascorr_{beamPDG}.pdf")
plt.show()