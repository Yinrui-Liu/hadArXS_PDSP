from hadana.packages import *
from hadana.processor import Processor
import hadana.selection as selection
import hadana.MC_reweight as reweight


# pduneana_MC_20g4rw, PDSPProd4_data_1GeV_reco2_ntuple_v09_41_00_04
PDSP_ntuple_name_MC = "pduneana_MC_20g4rw"
PDSP_ntuple_name_data = "PDSPProd4_data_1GeV_reco2_ntuple_v09_41_00_04"
beamPDG = 211
outfilename_MC = "processed_files/procVars_muonrew_MC.pkl"
outfilename_data = "processed_files/procVars_muonrew_data.pkl"
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
selection_muonrew = [True,True,True,False,False,True] # full selection except for fiducial volume cut and Michel score cut

if os.path.exists(outfilename_MC):
    with open(outfilename_MC, 'rb') as procfile_MC:
        processedVars_MC = pickle.load(procfile_MC)
    print(f"Using existing file {outfilename_MC}")
else:
    PDSP_ntuple_MC = uproot.open(f"input_files/{PDSP_ntuple_name_MC}.root")
    pduneana_MC = PDSP_ntuple_MC["pduneana/beamana"]

    eventset_MC = Processor(pduneana_MC, particle, isMC=True, selection=selection_muonrew, fake_data=False)
    eventset_MC.LoadVariables(variables_to_load)
    eventset_MC.ProcessEvent(Nevents=Nevents)
    processedVars_MC = eventset_MC.GetOutVarsDict()

    weights_MC = reweight.cal_bkg_reweight(processedVars_MC) * reweight.cal_momentum_reweight(processedVars_MC)
    processedVars_MC["reweight"] = weights_MC

    with open(outfilename_MC, 'wb') as procfile_MC:
        pickle.dump(processedVars_MC, procfile_MC)

if os.path.exists(outfilename_data):
    with open(outfilename_data, 'rb') as procfile_data:
        processedVars_data = pickle.load(procfile_data)
    print(f"Using existing file {outfilename_data}")
else:
    PDSP_ntuple_data = uproot.open(f"input_files/{PDSP_ntuple_name_data}.root")
    pduneana_data = PDSP_ntuple_data["pduneana/beamana"]

    eventset_data = Processor(pduneana_data, particle, isMC=False, selection=selection_muonrew)
    eventset_data.LoadVariables(variables_to_load)
    eventset_data.ProcessEvent(Nevents=Nevents)
    processedVars_data = eventset_data.GetOutVarsDict()

    weights_data = reweight.cal_bkg_reweight(processedVars_data) * reweight.cal_momentum_reweight(processedVars_data)
    processedVars_data["reweight"] = weights_data

    with open(outfilename_data, 'wb') as procfile_data:
        pickle.dump(processedVars_data, procfile_data)


### reconstructed track length
mask_SelectedPart = processedVars_MC["mask_SelectedPart"]
mask_FullSelection = processedVars_MC["mask_FullSelection"]
mask_MC = (mask_SelectedPart & mask_FullSelection)[:Nevents]
weights_MC = processedVars_MC["reweight"]
weights_MC = weights_MC[:Nevents][mask_MC]
weights_MC = np.ones_like(weights_MC) # do not include the existing weights (which may include muon reweight and momentum reweight)
x_MC = processedVars_MC["reco_track_length"][:Nevents][mask_MC]
par_type_MC = processedVars_MC["particle_type"][:Nevents][mask_MC]

mask_SelectedPart = processedVars_data["mask_SelectedPart"]
mask_FullSelection = processedVars_data["mask_FullSelection"]
mask_data = (mask_SelectedPart & mask_FullSelection)[:Nevents]
weights_data = processedVars_data["reweight"]
weights_data = weights_data[:Nevents][mask_data]
x_data = processedVars_data["reco_track_length"][:Nevents][mask_data]

bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 500]

### get muon reweight
wmu_list = np.linspace(1.55, 1.7, 100)
chi2_list = []
for wmu in wmu_list:
    weight_mu = np.where(par_type_MC==3, wmu, 1)
    weights_MC_rew = weights_MC*weight_mu
    chi2, _ = utils.cal_chi2_2hists(x_data, x_MC, weights_data, weights_MC_rew, bins, fit_range=[150, 500])
    chi2_list.append(chi2)
min_chi2 = min(chi2_list)
min_chi2_index = chi2_list.index(min_chi2)
wmu_min = wmu_list[min_chi2_index]
plt.plot(wmu_list, chi2_list, label="FOM")
plt.plot([wmu_list[0], wmu_list[-1]], [min_chi2, min_chi2], "r--")
plt.plot([wmu_list[0], wmu_list[-1]], [min_chi2+1, min_chi2+1], "r:")
plt.legend()
plt.xlabel("Muon weight")
plt.ylabel(r"$\chi^2$")
plt.savefig("plots/muonrew_curve.pdf")
plt.show()

### compare the histograms
x_hist_data, data_errors, _ = utils.get_vars_hists([x_data], [weights_data], bins)
x_hist_data = x_hist_data[0]
data_errors = data_errors[0]
#x_hist_data, _ = np.histogram(x_data, bins, weights=weights_data)
x_hist_MC, _ = np.histogram(x_MC, bins, weights=weights_MC)

weight_mu = np.where(par_type_MC==3, wmu_min, 1)
weights_MC_rew = weights_MC*weight_mu
x_hist_MC_rew, _ = np.histogram(x_MC, bins, weights=weights_MC_rew)

# Plot the histograms
bin_centers = (np.array(bins[:-1]) + np.array(bins[1:])) / 2
bin_widths = np.diff(bins)
plt.figure(figsize=(10, 6))
plt.errorbar(bin_centers, x_hist_data, yerr=data_errors, fmt='.', label='Data', color='k')
#plt.bar(bin_centers, x_hist_data, width=bin_widths, alpha=0.6, label='Data', color='k')
plt.step(np.concatenate([[bins[0]], bins]), np.concatenate([[0], x_hist_MC*sum(weights_data)/sum(weights_MC), [0]]), where='post', label='Original MC', color='gold')
#plt.bar(bin_centers, x_hist_MC*sum(weights_data)/sum(weights_MC), width=bin_widths, label='Original MC', edgecolor='r', linestyle='--', fill=False)
plt.step(np.concatenate([[bins[0]], bins]), np.concatenate([[0], x_hist_MC_rew*sum(weights_data)/sum(weights_MC_rew), [0]]), where='post', label=f'Reweighted MC (muon weight = {wmu_min:.3f})', color='r', linestyle='--')
plt.xlabel('Reconstructed track length [cm]')
plt.ylabel('Counts (all normalized to data)')
plt.title('Histograms of Data and MC')
plt.legend()
plt.ylim([0, None])
plt.savefig("plots/muonrew_hist.pdf")
plt.show()
