from hadana.packages import *
from hadana.processor import Processor
import hadana.selection as selection
import hadana.MC_reweight as reweight


PDSP_ntuple_name = "pduneana_MC_20g4rw"
beamPDG = 211
Nevents = None


PDSP_ntuple = uproot.open(f"input_files/{PDSP_ntuple_name}.root")
isMC = True
pduneana = PDSP_ntuple["pduneana/beamana"]

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

eventset = Processor(pduneana, particle, isMC, selection=[False,False,True,False,False,False], fake_data=False, incBQcut=[False,False,True]) # only apply beam scraper cut
eventset.fidvol_low = 0 # fiducial volume is not used
eventset.LoadVariables(variables_to_load)
eventset.ProcessEvent(Nevents=Nevents)
processedVars = eventset.GetOutVarsDict()

weights = reweight.cal_bkg_reweight(processedVars) * reweight.cal_momentum_reweight(processedVars)
processedVars["reweight"] = weights

mask_SelectedPart = processedVars["mask_SelectedPart"]
mask_FullSelection = processedVars["mask_FullSelection"]
mask_TruePDGMatched = processedVars["mask_TrueSignal"] & np.array(processedVars["reco_beam_true_byE_matched"], dtype=bool)
mask = (mask_SelectedPart & mask_FullSelection & mask_TruePDGMatched)[:Nevents]
weights = weights[:Nevents][mask]

true_Eff = processedVars["true_initial_energy"][:Nevents][mask]
beam_inst_KE = processedVars["beam_inst_KE"][:Nevents][mask]
upEloss = beam_inst_KE - true_Eff

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(beam_inst_KE, upEloss, s=weights*10, marker='.', alpha=0.3, color='b', label='MC true beam events')
plt.xlabel('beam_inst_KE [MeV]')
plt.ylabel('beam_inst_KE - true_Eff [MeV]')
if beamPDG == 211:
    xrange = [600, 1100]
    x_fit = np.linspace(700, 1050, 100)
    fit_filter = (beam_inst_KE > 700) & (beam_inst_KE < 1050) & (upEloss > -100) & (upEloss < 150)
elif beamPDG == 2212:
    xrange = [250, 650]
    x_fit = np.linspace(300, 600, 100)
    fit_filter = (beam_inst_KE > 300) & (beam_inst_KE < 600) & (upEloss > -50) & (upEloss < 100)
# Fit a quadratic curve
coeffs, cov_matrix = np.polyfit(beam_inst_KE[fit_filter], upEloss[fit_filter], 2, w=weights[fit_filter], cov=True)
errors = np.sqrt(np.diag(cov_matrix))
print(f"Fit Parameters: a = {coeffs[0]:.4g} ± {errors[0]:.4g}, b = {coeffs[1]:.4g} ± {errors[1]:.4g}, c = {coeffs[2]:.4g} ± {errors[2]:.4g}")
poly = np.poly1d(coeffs)
y_fit = poly(x_fit)
plt.plot(x_fit, y_fit, color='red', linestyle='--', label=f'Quadratic Fit')
plt.legend()
plt.xlim(xrange)
plt.ylim([-150, 200])
plt.grid(True)
plt.show()