from packages import *
import get_hists
import calcXS
import multiD_unfolding
from BetheBloch import BetheBloch
from parameters import true_bins_pionp as true_bins, meas_bins_pionp as meas_bins


### load data histograms
with open('processedVars.pkl', 'rb') as datafile: # either fake data or real data
    processedVars = pickle.load(datafile)

mask_SelectedPart = processedVars["mask_SelectedPart"]
mask_FullSelection = processedVars["mask_FullSelection"]
combined_mask = mask_SelectedPart & mask_FullSelection
reco_initial_energy = processedVars["reco_initial_energy"]
reco_end_energy = processedVars["reco_end_energy"]
reco_sigflag = processedVars["reco_sigflag"]
particle_type = processedVars["particle_type"]
reweight = processedVars["reweight"]


### selection
print("### selection")
divided_recoEini, divided_weights = get_hists.divide_vars_by_partype(reco_initial_energy, particle_type, mask=combined_mask, weight=reweight)
divided_recoEend, divided_weights = get_hists.divide_vars_by_partype(reco_end_energy, particle_type, mask=combined_mask, weight=reweight)
divided_recoflag, divided_weights = get_hists.divide_vars_by_partype(reco_sigflag, particle_type, mask=combined_mask, weight=reweight)
data_reco_Eini = divided_recoEini[0]
data_reco_Eend = divided_recoEend[0]
data_reco_flag = divided_recoflag[0]
data_reco_weight = divided_weights[0]

Ndata = len(data_reco_Eini)
Ntruebins, Ntruebins_3D, true_cKE, true_wKE = utils.set_bins(true_bins)
Nmeasbins = len(meas_bins)
data_meas_SIDini, data_meas_SIDend, data_meas_SIDint_ex = calcXS.get_sliceID_histograms(data_reco_Eini, data_reco_Eend, data_reco_flag, meas_bins)
data_meas_SID3D, data_meas_N3D, data_meas_N3D_Vcov = calcXS.get_3D_histogram(data_meas_SIDini, data_meas_SIDend, data_meas_SIDint_ex, Nmeasbins, data_reco_weight)
data_meas_N3D_err = np.sqrt(np.diag(data_meas_N3D_Vcov))


### background subtraction
print("### background subtraction")
with open('processedVars.pkl', 'rb') as mcfile: # truth MC
    processedVars_mc = pickle.load(mcfile)

mask_SelectedPart_mc = processedVars_mc["mask_SelectedPart"]
mask_FullSelection_mc = processedVars_mc["mask_FullSelection"]
combined_mask_mc = mask_SelectedPart_mc & mask_FullSelection_mc
reco_initial_energy_mc = processedVars_mc["reco_initial_energy"]
reco_end_energy_mc = processedVars_mc["reco_end_energy"]
reco_sigflag_mc = processedVars_mc["reco_sigflag"]
particle_type_mc = processedVars_mc["particle_type"]
reweight_mc = processedVars_mc["reweight"]
divided_recoEini_mc, divided_weights_mc = get_hists.divide_vars_by_partype(reco_initial_energy_mc, particle_type_mc, mask=combined_mask_mc, weight=reweight_mc)
divided_recoEend_mc, divided_weights_mc = get_hists.divide_vars_by_partype(reco_end_energy_mc, particle_type_mc, mask=combined_mask_mc, weight=reweight_mc)
divided_recoflag_mc, divided_weights_mc = get_hists.divide_vars_by_partype(reco_sigflag_mc, particle_type_mc, mask=combined_mask_mc, weight=reweight_mc)
Ntruemc = len(reco_initial_energy_mc[combined_mask_mc]) - len(divided_recoEini_mc[0])
bkg_meas_N3D_list = []
bkg_meas_N3D_err_list = []
for ibkg in range(3, len(divided_recoEini_mc)):
    bkg_reco_Eini = divided_recoEini_mc[ibkg]
    bkg_reco_Eend = divided_recoEend_mc[ibkg]
    bkg_reco_flag = divided_recoflag_mc[ibkg]
    bkg_reco_weight = divided_weights_mc[ibkg]
    bkg_meas_SIDini, bkg_meas_SIDend, bkg_meas_SIDint_ex = calcXS.get_sliceID_histograms(bkg_reco_Eini, bkg_reco_Eend, bkg_reco_flag, meas_bins)
    bkg_meas_SID3D, bkg_meas_N3D, bkg_meas_N3D_Vcov = calcXS.get_3D_histogram(bkg_meas_SIDini, bkg_meas_SIDend, bkg_meas_SIDint_ex, Nmeasbins, bkg_reco_weight)
    bkg_meas_N3D_list.append(bkg_meas_N3D)
    bkg_meas_N3D_err_list.append(np.sqrt(np.diag(bkg_meas_N3D_Vcov)))

bkg_scale = [1,1,1,1,1,1,1] # should be imported from sideband fit
bkg_scale_err = [0,0,0,0,0,0,0]
sig_meas_N3D, sig_meas_N3D_err = get_hists.bkg_subtraction(data_meas_N3D, data_meas_N3D_err, bkg_meas_N3D_list, bkg_meas_N3D_err_list, mc2data_scale=Ndata/Ntruemc, bkg_scale=bkg_scale, bkg_scale_err=bkg_scale_err)


### unfolding
print("### unfolding")
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

sig_meas_N1D, sig_meas_N1D_err = multiD_unfolding.map_data_to_MC_bins(sig_meas_N3D, sig_meas_N3D_err, meas_3D1D_map)
sig_meas_V1D = np.diag(sig_meas_N1D_err*sig_meas_N1D_err)
#print(sig_meas_N1D, sig_meas_N1D_err, sep='\n')

sig_MC_scale = sum(sig_meas_N1D)/sum(meas_N3D)
sig_unfold, sig_unfold_cov = multiD_unfolding.unfolding(sig_meas_N1D, sig_meas_V1D, response, niter=10)
unfd_N3D, unfd_N3D_Vcov = multiD_unfolding.efficiency_correct_1Dvar(sig_unfold, sig_unfold_cov, eff1D, true_3D1D_map, Ntruebins_3D, true_N3D, true_N3D_Vcov, sig_MC_scale)
unfd_Nini, unfd_Nend, unfd_Nint_ex, unfd_Ninc = multiD_unfolding.get_unfold_histograms(unfd_N3D, Ntruebins)
#print(unfd_Nini, unfd_Nend, unfd_Nint_ex, unfd_Ninc, sep='\n')


### calculate cross section
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