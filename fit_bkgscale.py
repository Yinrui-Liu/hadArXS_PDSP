from hadana.packages import *
from hadana.processor import Processor
import hadana.selection as selection


# TB added fake data fits, and re-consider the normalization in fitting
# pduneana_MC_20g4rw, PDSPProd4_data_1GeV_reco2_ntuple_v09_41_00_04
PDSP_ntuple_name_MC = "pduneana_MC_20g4rw"
PDSP_ntuple_name_data = "PDSPProd4_data_1GeV_reco2_ntuple_v09_41_00_04"
beamPDG = 211
procfilename_MC = "processed_files/procVars_bkgfit_MC.pkl"
procfilename_data = "processed_files/procVars_bkgfit_data.pkl" # procfilename_data = procfilename_MC for fake data study
Nevents = None
variables_to_load = [
    "event",
    "reco_beam_calo_wire",
    "reco_beam_type",
    "reco_beam_vertex_nHits",
    "reco_beam_vertex_michel_score_weight_by_charge",
    "reco_beam_Chi2_proton",
    "reco_beam_Chi2_ndof",
    "MC",
    "beam_inst_X",
    "beam_inst_Y",
    "reco_beam_calo_startX",
    "reco_beam_calo_startY",
    "reco_beam_calo_startZ",
    "reco_beam_calo_endX",
    "reco_beam_calo_endY",
    "reco_beam_calo_endZ",
    "reco_beam_calo_X",
    "reco_beam_calo_Y",
    "reco_beam_calo_Z",
    "true_beam_traj_X",
    "true_beam_traj_Y",
    "true_beam_traj_Z",
    "reco_reconstructable_beam_event",
    "true_beam_PDG",
    "beam_inst_trigger",
    "beam_inst_nMomenta",
    "beam_inst_nTracks",
    "beam_inst_PDG_candidates",
    "beam_inst_P",
    "reco_beam_calibrated_dEdX_SCE",
    "reco_beam_resRange_SCE",
    "true_beam_traj_KE",
    "reco_beam_true_byE_matched",
    "reco_beam_true_byE_origin",
    "reco_beam_true_byE_PDG",
    "true_beam_endProcess",
    "g4rw_full_grid_piplus_coeffs",
    "g4rw_full_grid_proton_coeffs",
    "true_beam_startP",
]
if beamPDG == 211:
    particle = selection.Particle(beamPDG, 139.57)
    particle.SetCandidatePDGlist([-13, 13, 211])
elif beamPDG == 2212:
    particle = selection.Particle(beamPDG, 938.272)
    particle.SetCandidatePDGlist(2212)
selection_bkgfit_mu = [True,True,True,True,False,True] # full selection except for Michel score cut
selection_bkgfit_p = [True,True,True,True,True,False] # full selection except for Proton cut
incBQcut_bkgfit_spi = [True,False,True] # no angle cut in the beam quality cut

if os.path.exists(procfilename_MC):
    with open(procfilename_MC, 'rb') as procfile_MC:
        processedVars_MC = pickle.load(procfile_MC)
    print(f"Using existing file {procfilename_MC}")
    if 'mask_Selection_bkgfit_mu' not in processedVars_MC:
        print(f"Processing to add bkgfit-related variables in {procfilename_MC}")
        PDSP_ntuple_MC = uproot.open(f"input_files/{PDSP_ntuple_name_MC}.root")
        pduneana_MC = PDSP_ntuple_MC["pduneana/beamana"]

        eventset_MC = Processor(pduneana_MC, particle, isMC=True, selection=selection_bkgfit_mu, fake_data=False)
        eventset_MC.LoadVariables(variables_to_load)
        eventset_MC.ProcessEvent(Nevents=Nevents)
        mask_Selection_bkgfit_mu = eventset_MC.mask_FullSelection
        if len(mask_Selection_bkgfit_mu) != len(processedVars_MC["mask_FullSelection"]):
            raise Exception("The number of events are not the same. Run load_ntuple.py again!")
        del eventset_MC

        eventset_MC = Processor(pduneana_MC, particle, isMC=True, selection=selection_bkgfit_p, fake_data=False)
        eventset_MC.LoadVariables(variables_to_load)
        eventset_MC.ProcessEvent(Nevents=Nevents)
        mask_Selection_bkgfit_p = eventset_MC.mask_FullSelection
        del eventset_MC
        
        eventset_MC = Processor(pduneana_MC, particle, isMC=True, incBQcut=incBQcut_bkgfit_spi, fake_data=False)
        eventset_MC.LoadVariables(variables_to_load)
        eventset_MC.ProcessEvent(Nevents=Nevents)
        mask_Selection_bkgfit_spi = eventset_MC.mask_FullSelection
        
        processedVars_MC["Michel_score_bkgfit_mu"] = eventset_MC.Michel_score_bkgfit_mu
        processedVars_MC["mask_Selection_bkgfit_mu"] = mask_Selection_bkgfit_mu
        processedVars_MC["proton_chi2_bkgfit_p"] = eventset_MC.proton_chi2_bkgfit_p
        processedVars_MC["mask_Selection_bkgfit_p"] = mask_Selection_bkgfit_p
        processedVars_MC["costheta_bkgfit_spi"] = eventset_MC.costheta_bkgfit_spi
        processedVars_MC["mask_Selection_bkgfit_spi"] = mask_Selection_bkgfit_spi
        with open(procfilename_MC, 'wb') as procfile_MC:
            pickle.dump(processedVars_MC, procfile_MC)
else:
    print(f"{procfilename_MC} not exists. Run load_ntuple.py first to get processed file.")

if os.path.exists(procfilename_data):
    with open(procfilename_data, 'rb') as procfile_data:
        processedVars_data = pickle.load(procfile_data)
    print(f"Using existing file {procfilename_data}")
    if 'mask_Selection_bkgfit_mu' not in processedVars_data:
        print(f"Processing to add bkgfit-related variables in {procfilename_data}")
        PDSP_ntuple_data = uproot.open(f"input_files/{PDSP_ntuple_name_data}.root")
        pduneana_data = PDSP_ntuple_data["pduneana/beamana"]

        eventset_data = Processor(pduneana_data, particle, isMC=False, selection=selection_bkgfit_mu)
        eventset_data.LoadVariables(variables_to_load)
        eventset_data.ProcessEvent(Nevents=Nevents)
        mask_Selection_bkgfit_mu = eventset_data.mask_FullSelection
        if len(mask_Selection_bkgfit_mu) != len(processedVars_data["mask_FullSelection"]):
            raise Exception("The number of events are not the same. Run load_ntuple.py again!")
        del eventset_data

        eventset_data = Processor(pduneana_data, particle, isMC=False, selection=selection_bkgfit_p)
        eventset_data.LoadVariables(variables_to_load)
        eventset_data.ProcessEvent(Nevents=Nevents)
        mask_Selection_bkgfit_p = eventset_data.mask_FullSelection
        del eventset_data
        
        eventset_data = Processor(pduneana_data, particle, isMC=False, incBQcut=incBQcut_bkgfit_spi)
        eventset_data.LoadVariables(variables_to_load)
        eventset_data.ProcessEvent(Nevents=Nevents)
        mask_Selection_bkgfit_spi = eventset_data.mask_FullSelection
        
        processedVars_data["Michel_score_bkgfit_mu"] = eventset_data.Michel_score_bkgfit_mu
        processedVars_data["mask_Selection_bkgfit_mu"] = mask_Selection_bkgfit_mu
        processedVars_data["proton_chi2_bkgfit_p"] = eventset_data.proton_chi2_bkgfit_p
        processedVars_data["mask_Selection_bkgfit_p"] = mask_Selection_bkgfit_p
        processedVars_data["costheta_bkgfit_spi"] = eventset_data.costheta_bkgfit_spi
        processedVars_data["mask_Selection_bkgfit_spi"] = mask_Selection_bkgfit_spi
        with open(procfilename_data, 'wb') as procfile_data:
            pickle.dump(processedVars_data, procfile_data)
else:
    print(f"{procfilename_data} not exists. Run load_ntuple.py first to get processed file.")


mask_SelectedPart_MC = processedVars_MC["mask_SelectedPart"]
weights_MC = processedVars_MC["reweight"]
mask_SelectedPart_data = processedVars_data["mask_SelectedPart"]
weights_data = processedVars_data["reweight"]

pardict = {
    0: "Data", 
    1: "Pion inelastic", 
    2: "Pion decay", 
    3: "Muon", 
    4: "misID:cosmic", 
    5: "misID:proton", 
    6: "misID:pion", 
    7: "misID:muon", 
    8: "misID:e/γ", 
    9: "misID:other", 
}
parcolordict = {
    "Pion inelastic": "firebrick",
    "Pion decay": "orange",
    "Muon": "springgreen",
    "misID:cosmic": "deepskyblue",
    "misID:proton": "darkviolet",
    "misID:pion": "hotpink",
    "misID:muon": "green",
    "misID:e/γ": "yellow",
    "misID:other": "peru",
}
# uncomment for fake data study, and don't forget to add "&(isFake==1)" for mask_MC and "&(isFake==0)" for mask_data
'''PDSP_ntuple_MC = uproot.open(f"input_files/{PDSP_ntuple_name_MC}.root")
pduneana_MC = PDSP_ntuple_MC["pduneana/beamana"]
isFake = np.array(pduneana_MC["event"])%2'''
### use Michel score distribution for muon bkg fit
mask_Selection_bkgfit_mu = processedVars_MC["mask_Selection_bkgfit_mu"]
mask_MC = (mask_SelectedPart_MC & mask_Selection_bkgfit_mu)[:Nevents]
weights_MC_mu = weights_MC[:Nevents][mask_MC]
x_MC = processedVars_MC["Michel_score_bkgfit_mu"][:Nevents][mask_MC]
par_type_MC = processedVars_MC["particle_type"][:Nevents][mask_MC]

mask_Selection_bkgfit_mu = processedVars_data["mask_Selection_bkgfit_mu"]
mask_data = (mask_SelectedPart_data & mask_Selection_bkgfit_mu)[:Nevents]
weights_data_mu = weights_data[:Nevents][mask_data]
x_data = processedVars_data["Michel_score_bkgfit_mu"][:Nevents][mask_data]

bins = np.linspace(0, 1, 51)
fit_range = [0.6, 0.9]
def sideband_fit_mu(scale_factor):
    weight_bkg = np.where( (par_type_MC==3)|(par_type_MC==7), scale_factor, 1)
    weights_MC_rew = weights_MC_mu*weight_bkg
    chi2, _ = utils.cal_chi2_2hists(x_data, x_MC, weights_data_mu, weights_MC_rew, bins, fit_range, scale21=np.sum(weights_data_mu)/np.sum(weights_MC_mu))
    return chi2
m = iminuit.Minuit(sideband_fit_mu, scale_factor=1)
m.migrad()
sf_mu = m.values["scale_factor"]
sferr_mu = m.errors["scale_factor"]
print(f"Best fit scale factor: {sf_mu}")
print(f"Uncertainty on scale factor: {sferr_mu}")

x_hist_data, data_errors, _ = utils.get_vars_hists([x_data], [weights_data_mu], bins)
x_hist_data = x_hist_data[0]
data_errors = data_errors[0]
x_hist_MC, _ = np.histogram(x_MC, bins, weights=weights_MC_mu)

weight_mu = np.where( (par_type_MC==3)|(par_type_MC==7), sf_mu, 1)
weights_MC_rew = weights_MC_mu*weight_mu
x_hist_MC_rew, _ = np.histogram(x_MC, bins, weights=weights_MC_rew)

bin_centers = (np.array(bins[:-1]) + np.array(bins[1:])) / 2
bin_widths = np.diff(bins)
plt.figure(figsize=(10, 6))
plt.errorbar(bin_centers, x_hist_data, yerr=data_errors, fmt='.', label='Data', color='k')
plt.step(np.concatenate([[bins[0]], bins]), np.concatenate([[0], x_hist_MC*sum(weights_data_mu)/sum(weights_MC_mu), [0]]), where='post', label='Original MC (total)', color='gold')
divided_vars_mc, divided_weights_mc = utils.divide_vars_by_partype(x_MC, par_type_MC, mask=np.ones_like(x_MC, dtype=bool), weight=weights_MC_mu)
divided_weights_mc = [np.array(i)*sum(weights_data_mu)/sum(weights_MC_mu) for i in divided_weights_mc]
plt.hist(divided_vars_mc[1:], bins, weights=divided_weights_mc[1:], label=[f'{pardict[i+1]}' for i in range(len(divided_vars_mc[1:]))], color=[f'{parcolordict[pardict[i+1]]}' for i in range(len(divided_vars_mc[1:]))], stacked=True, alpha=0.3)
plt.step(np.concatenate([[bins[0]], bins]), np.concatenate([[0], x_hist_MC_rew*sum(weights_data_mu)/sum(weights_MC_rew), [0]]), where='post', label=f'Reweighted MC (muon weight = {sf_mu:.3f}±{sferr_mu:.3f})', color='r', linestyle='--')
plt.xlabel('Daughter Michel score')
plt.ylabel('Counts (all normalized to data)')
plt.legend()
plt.ylim([0.1, None])
plt.yscale('log')
plt.savefig("plots/bkgfit_mu.pdf")
plt.show()

### use proton chi2 distribution for proton bkg fit
mask_Selection_bkgfit_p = processedVars_MC["mask_Selection_bkgfit_p"]
mask_MC = (mask_SelectedPart_MC & mask_Selection_bkgfit_p)[:Nevents]
weights_MC_p = weights_MC[:Nevents][mask_MC]
x_MC = processedVars_MC["proton_chi2_bkgfit_p"][:Nevents][mask_MC]
par_type_MC = processedVars_MC["particle_type"][:Nevents][mask_MC]

mask_Selection_bkgfit_p = processedVars_data["mask_Selection_bkgfit_p"]
mask_data = (mask_SelectedPart_data & mask_Selection_bkgfit_p)[:Nevents]
weights_data_p = weights_data[:Nevents][mask_data]
x_data = processedVars_data["proton_chi2_bkgfit_p"][:Nevents][mask_data]

bins = np.linspace(0, 100, 51)
fit_range = [20, 70]
def sideband_fit_p(scale_factor):
    weight_bkg = np.where(par_type_MC==5, scale_factor, 1)
    weights_MC_rew = weights_MC_p*weight_bkg
    chi2, _ = utils.cal_chi2_2hists(x_data, x_MC, weights_data_p, weights_MC_rew, bins, fit_range, scale21=np.sum(weights_data_p)/np.sum(weights_MC_p))
    return chi2
m = iminuit.Minuit(sideband_fit_p, scale_factor=1)
m.migrad()
sf_p = m.values["scale_factor"]
sferr_p = m.errors["scale_factor"]
print(f"Best fit scale factor: {sf_p}")
print(f"Uncertainty on scale factor: {sferr_p}")

x_hist_data, data_errors, _ = utils.get_vars_hists([x_data], [weights_data_p], bins)
x_hist_data = x_hist_data[0]
data_errors = data_errors[0]
x_hist_MC, _ = np.histogram(x_MC, bins, weights=weights_MC_p)

weight_p = np.where(par_type_MC==5, sf_p, 1)
weights_MC_rew = weights_MC_p*weight_p
x_hist_MC_rew, _ = np.histogram(x_MC, bins, weights=weights_MC_rew)

bin_centers = (np.array(bins[:-1]) + np.array(bins[1:])) / 2
bin_widths = np.diff(bins)
plt.figure(figsize=(10, 6))
plt.errorbar(bin_centers, x_hist_data, yerr=data_errors, fmt='.', label='Data', color='k')
plt.step(np.concatenate([[bins[0]], bins]), np.concatenate([[0], x_hist_MC*sum(weights_data_p)/sum(weights_MC_p), [0]]), where='post', label='Original MC (total)', color='gold')
divided_vars_mc, divided_weights_mc = utils.divide_vars_by_partype(x_MC, par_type_MC, mask=np.ones_like(x_MC, dtype=bool), weight=weights_MC_p)
divided_weights_mc = [np.array(i)*sum(weights_data_p)/sum(weights_MC_p) for i in divided_weights_mc]
plt.hist(divided_vars_mc[1:], bins, weights=divided_weights_mc[1:], label=[f'{pardict[i+1]}' for i in range(len(divided_vars_mc[1:]))], color=[f'{parcolordict[pardict[i+1]]}' for i in range(len(divided_vars_mc[1:]))], stacked=True, alpha=0.3)
plt.step(np.concatenate([[bins[0]], bins]), np.concatenate([[0], x_hist_MC_rew*sum(weights_data_p)/sum(weights_MC_rew), [0]]), where='post', label=f'Reweighted MC (proton weight = {sf_p:.3f}±{sferr_p:.3f})', color='r', linestyle='--')
plt.xlabel(r'Proton dE/dx $\chi^2$')
plt.ylabel('Counts (all normalized to data)')
plt.legend()
plt.ylim([0.1, None])
plt.yscale('log')
plt.savefig("plots/bkgfit_p.pdf")
plt.show()

### use beam angle costheta for secondary pion bkg fit
mask_Selection_bkgfit_spi = processedVars_MC["mask_Selection_bkgfit_spi"]
mask_MC = (mask_SelectedPart_MC & mask_Selection_bkgfit_spi)[:Nevents]
weights_MC_spi = weights_MC[:Nevents][mask_MC]
x_MC = processedVars_MC["costheta_bkgfit_spi"][:Nevents][mask_MC]
par_type_MC = processedVars_MC["particle_type"][:Nevents][mask_MC]

mask_Selection_bkgfit_spi = processedVars_data["mask_Selection_bkgfit_spi"]
mask_data = (mask_SelectedPart_data & mask_Selection_bkgfit_spi)[:Nevents]
weights_data_spi = weights_data[:Nevents][mask_data]
x_data = processedVars_data["costheta_bkgfit_spi"][:Nevents][mask_data]

bins = np.linspace(0.85, 1, 51)
fit_range = [0.9, 0.95]
weights_MC_spi = weights_MC_spi*np.where(par_type_MC==5, sf_p, 1)
def sideband_fit_spi(scale_factor):
    weight_bkg = np.where(par_type_MC==6, scale_factor, 1)
    weights_MC_rew = weights_MC_spi*weight_bkg
    chi2, _ = utils.cal_chi2_2hists(x_data, x_MC, weights_data_spi, weights_MC_rew, bins, fit_range, scale21=np.sum(weights_data_spi)/np.sum(weights_MC_spi))
    return chi2
m = iminuit.Minuit(sideband_fit_spi, scale_factor=1)
m.migrad()
sf_spi = m.values["scale_factor"]
sferr_spi = m.errors["scale_factor"]
print(f"Best fit scale factor: {sf_spi}")
print(f"Uncertainty on scale factor: {sferr_spi}")

x_hist_data, data_errors, _ = utils.get_vars_hists([x_data], [weights_data_spi], bins)
x_hist_data = x_hist_data[0]
data_errors = data_errors[0]
x_hist_MC, _ = np.histogram(x_MC, bins, weights=weights_MC_spi)

weight_spi = np.where(par_type_MC==6, sf_spi, 1)
weights_MC_rew = weights_MC_spi*weight_spi
x_hist_MC_rew, _ = np.histogram(x_MC, bins, weights=weights_MC_rew)

bin_centers = (np.array(bins[:-1]) + np.array(bins[1:])) / 2
bin_widths = np.diff(bins)
plt.figure(figsize=(10, 6))
plt.errorbar(bin_centers, x_hist_data, yerr=data_errors, fmt='.', label='Data', color='k')
plt.step(np.concatenate([[bins[0]], bins]), np.concatenate([[0], x_hist_MC*sum(weights_data_spi)/sum(weights_MC_spi), [0]]), where='post', label='Original MC (total)', color='gold')
divided_vars_mc, divided_weights_mc = utils.divide_vars_by_partype(x_MC, par_type_MC, mask=np.ones_like(x_MC, dtype=bool), weight=weights_MC_spi)
divided_weights_mc = [np.array(i)*sum(weights_data_spi)/sum(weights_MC_spi) for i in divided_weights_mc]
plt.hist(divided_vars_mc[1:], bins, weights=divided_weights_mc[1:], label=[f'{pardict[i+1]}' for i in range(len(divided_vars_mc[1:]))], color=[f'{parcolordict[pardict[i+1]]}' for i in range(len(divided_vars_mc[1:]))], stacked=True, alpha=0.3)
plt.step(np.concatenate([[bins[0]], bins]), np.concatenate([[0], x_hist_MC_rew*sum(weights_data_spi)/sum(weights_MC_rew), [0]]), where='post', label=f'Reweighted MC (secondary pion weight = {sf_spi:.3f}±{sferr_spi:.3f})', color='r', linestyle='--')
plt.xlabel(r'Beam angle $\cos\theta$')
plt.ylabel('Counts (all normalized to data)')
plt.legend()
plt.ylim([0.1, None])
plt.yscale('log')
plt.savefig("plots/bkgfit_spi.pdf")
plt.show()