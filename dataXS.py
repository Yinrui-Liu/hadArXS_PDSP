from packages import *
import get_hists
import calcXS
import multiD_unfolding
from BetheBloch import BetheBloch
from parameters import true_bins_pionp as true_bins, meas_bins_pionp as meas_bins

with open('processedVars.pkl', 'rb') as procfile: # data file (either fake or real)
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
divided_recoEini, divided_weights = get_hists.divide_vars_by_partype(reco_initial_energy, particle_type, mask=combined_true_mask, weight=reweight)
divided_recoEend, divided_weights = get_hists.divide_vars_by_partype(reco_end_energy, particle_type, mask=combined_true_mask, weight=reweight)
divided_recoflag, divided_weights = get_hists.divide_vars_by_partype(reco_sigflag, particle_type, mask=combined_true_mask, weight=reweight)
divided_FullSelection, divided_weights = get_hists.divide_vars_by_partype(mask_FullSelection, particle_type, mask=combined_true_mask, weight=reweight)

# measure data
Ntruebins, Ntruebins_3D, true_cKE, true_wKE = utils.set_bins(true_bins)
Nmeasbins = len(meas_bins)

data_pass_selection = divided_FullSelection[0]
data_reco_Eini = divided_recoEini[0][data_pass_selection]
data_reco_Eend = divided_recoEend[0][data_pass_selection]
data_reco_flag = divided_recoflag[0][data_pass_selection]
data_reco_weight = divided_weights[0][data_pass_selection]
data_meas_SIDini, data_meas_SIDend, data_meas_SIDint_ex = calcXS.get_sliceID_histograms(data_reco_Eini, data_reco_Eend, data_reco_flag, meas_bins)
data_meas_SID3D, data_meas_N3D, data_meas_N3D_Vcov = calcXS.get_3D_histogram(data_meas_SIDini, data_meas_SIDend, data_meas_SIDint_ex, Nmeasbins, data_reco_weight)
with open('response.pkl', 'rb') as respfile: # unfolding vars modeled by MC
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

data_meas_N1D, data_meas_N1D_err = multiD_unfolding.map_data_to_MC_bins(data_meas_N3D, np.sqrt(np.diag(data_meas_N3D_Vcov)), meas_3D1D_map)
data_meas_V1D = np.diag(data_meas_N1D_err*data_meas_N1D_err)
#print(data_meas_N1D, data_meas_N1D_err, sep='\n')

data_MC_scale = sum(data_meas_N1D)/sum(meas_N3D)
data_unfold, data_unfold_cov = multiD_unfolding.unfolding(data_meas_N1D, data_meas_V1D, response, niter=10)
unfd_N3D, unfd_N3D_Vcov = multiD_unfolding.efficiency_correct_1Dvar(data_unfold, data_unfold_cov, eff1D, true_3D1D_map, Ntruebins_3D, true_N3D, true_N3D_Vcov, data_MC_scale)
unfd_Nini, unfd_Nend, unfd_Nint_ex, unfd_Ninc = multiD_unfolding.get_unfold_histograms(unfd_N3D, Ntruebins)
#print(unfd_Nini, unfd_Nend, unfd_Nint_ex, unfd_Ninc, sep='\n')

unfd_3SID_Vcov = calcXS.get_Cov_3SID_from_N3D(unfd_N3D_Vcov, Ntruebins)
unfd_3N_Vcov = calcXS.get_Cov_3N_from_3SID(unfd_3SID_Vcov, Ntruebins)
unfd_XS, unfd_XS_Vcov = calcXS.calculate_XS_Cov_from_3N(unfd_Ninc, unfd_Nend, unfd_Nint_ex, unfd_3N_Vcov, true_bins, BetheBloch(211))

plt.figure(figsize=[8,4.8])
XS_x = true_cKE[1:-1] # the underflow and overflow bin are not used
XS_y = unfd_XS[1:-1]
XS_xerr = true_wKE[1:-1]
XS_yerr = np.sqrt(np.diagonal(unfd_XS_Vcov))[1:-1] # get the uncertainty from the covariance matrix
plt.errorbar(XS_x, XS_y, XS_yerr, XS_xerr, fmt=".", label="Signal XS using unfolded result")
#xx = np.linspace(0, 1100, 100)
#plt.plot(xx,XS_gen_ex(xx), label="Signal cross section used in simulation")
plt.xlabel("Kinetic energy (MeV)")
plt.ylabel("Cross section (mb)") # 1 mb = 10^{-27} cm^2
plt.xlim([0,1000])
plt.ylim(bottom=0)
plt.show()

plt.pcolormesh(true_bins[1:-1], true_bins[1:-1], utils.transform_cov_to_corr_matrix(unfd_XS_Vcov[1:-1, 1:-1]), cmap="RdBu_r", vmin=-1, vmax=1)
plt.title(r"Correlation matrix for cross section")
plt.xticks(true_bins[1:-1])
plt.yticks(true_bins[1:-1])
plt.xlabel(r"Kinetic energy (MeV)")
plt.ylabel(r"Kinetic energy (MeV)")
plt.colorbar()
plt.show()
