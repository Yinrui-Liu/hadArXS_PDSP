from hadana.packages import *
import hadana.parameters as parameters
from hadana.processor import GetUpstreamEnergyLoss

beampdg = 211
procMCname = f"processed_files/procVars_piMC.pkl"

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
PDSP_ntuple = uproot.open(f"input_files/pduneana_MC_20g4rw.root")
pduneana_mc = PDSP_ntuple["pduneana/beamana"]

with open(procMCname, 'rb') as mcfile:
    processedVars_mc = pickle.load(mcfile)
mask_SelectedPart_mc = processedVars_mc["mask_SelectedPart"]
mask_FullSelection_mc = processedVars_mc["mask_FullSelection"]
mask_BeamMatched_mc = np.array(processedVars_mc["reco_beam_true_byE_matched"], dtype=bool)
mask_TrueSignal = processedVars_mc["mask_TrueSignal"]
combined_mask_mc = mask_SelectedPart_mc & mask_FullSelection_mc & mask_BeamMatched_mc & mask_TrueSignal
particle_type_mc = processedVars_mc["particle_type"]
reweight_mc = processedVars_mc["reweight"]
#reweight_mc = np.ones_like(reweight_mc) # no reweighting
weights = reweight_mc[combined_mask_mc]
print(f"Number of events: {len(weights)}; total weight {sum(weights)}.")

### Ebeam
Ebeam_reco = processedVars_mc["beam_inst_KE"]
Ebeam_true = utils.P_to_KE(np.array(pduneana_mc["true_beam_startP"])*1000, utils.get_mass_from_pdg(beampdg))
binedges = np.linspace(-200, 200, 100)
fit_range = [-50, 50]

x_data = (Ebeam_reco - Ebeam_true)[combined_mask_mc]
m = utils.fit_gaus_hist(x_data, weights, x_range=fit_range, initial_guesses=[0, 30])
print(f"Fitted parameters: mu={m.values['mu']:.2f}±{m.errors['mu']:.2f}, sigma={m.values['sigma']:.2f}±{m.errors['sigma']:.2f}")
# Plot the data and the fitted function
plt.hist(x_data, binedges, density=True, weights=weights, alpha=0.6, color='g', label=f'Data (mean={sum(x_data * weights)/sum(weights):.2f})')
x_fit = np.linspace(*fit_range, 1000)
y_fit = utils.gaussian(x_fit, m.values['mu'], m.values['sigma'])
plt.plot(x_fit, y_fit, color='red', label=f"Fitted Gaussian: μ={m.values['mu']:.2f}±{m.errors['mu']:.2f}, σ={m.values['sigma']:.2f}±{m.errors['sigma']:.2f}")
plt.legend()
plt.xlabel('Ebeam_reco - Ebeam_true [MeV]')
plt.savefig(f"plots/Ebeam_diff_{beampdg}.pdf")
plt.show()

plt.scatter(Ebeam_true[combined_mask_mc], x_data, s=weights*10, marker='.', alpha=0.3, color='b', label='True beam matched MC after full selection')
plt.xlabel('Ebeam_true [MeV]')
plt.ylabel('Ebeam_reco - Ebeam_true [MeV]')
plt.legend()
plt.xlim([0, 1200])
plt.ylim([-200, 200])
plt.grid(True)
plt.savefig(f"plots/Ebeam_2D_{beampdg}.png")
plt.show()

### Eloss
Eloss_reco = GetUpstreamEnergyLoss(Ebeam_reco, beampdg)
Eloss_true = Ebeam_true - processedVars_mc["true_frontface_energy"]
binedges = np.linspace(-200, 200, 100)
fit_range = [-30, 30]

x_data = (Eloss_reco - Eloss_true)[combined_mask_mc]
m = utils.fit_gaus_hist(x_data, weights, x_range=fit_range, initial_guesses=[0, 30])
print(f"Fitted parameters: mu={m.values['mu']:.2f}±{m.errors['mu']:.2f}, sigma={m.values['sigma']:.2f}±{m.errors['sigma']:.2f}")
# Plot the data and the fitted function
plt.hist(x_data, binedges, density=True, weights=weights, alpha=0.6, color='g', label=f'Data (mean={sum(x_data * weights)/sum(weights):.2f})')
x_fit = np.linspace(*fit_range, 1000)
y_fit = utils.gaussian(x_fit, m.values['mu'], m.values['sigma'])
plt.plot(x_fit, y_fit, color='red', label=f"Fitted Gaussian: μ={m.values['mu']:.2f}±{m.errors['mu']:.2f}, σ={m.values['sigma']:.2f}±{m.errors['sigma']:.2f}")
plt.legend()
plt.xlabel('Eloss_reco - Eloss_true [MeV]')
plt.savefig(f"plots/Eloss_diff_{beampdg}.pdf")
plt.show()

plt.scatter(Eloss_true[combined_mask_mc], x_data, s=weights*10, marker='.', alpha=0.3, color='b', label='True beam matched MC after full selection')
plt.xlabel('Eloss_true [MeV]')
plt.ylabel('Eloss_reco - Eloss_true [MeV]')
plt.legend()
plt.xlim([0, 1200])
plt.ylim([-200, 200])
plt.grid(True)
plt.savefig(f"plots/Eloss_2D_{beampdg}.png")
plt.show()

### Edepo
Edepo_reco = processedVars_mc["reco_frontface_energy"] - processedVars_mc["reco_end_energy"]
Edepo_true = processedVars_mc["true_frontface_energy"] - processedVars_mc["true_end_energy"]
binedges = np.linspace(-200, 200, 100)
fit_range = [-50, 50]

x_data = (Edepo_reco - Edepo_true)[combined_mask_mc]
m = utils.fit_gaus_hist(x_data, weights, x_range=fit_range, initial_guesses=[0, 30])
print(f"Fitted parameters: mu={m.values['mu']:.2f}±{m.errors['mu']:.2f}, sigma={m.values['sigma']:.2f}±{m.errors['sigma']:.2f}")
#m = utils.fit_doublegaus_hist(x_data, weights, x_range=fit_range, initial_guesses=[0, 30, 10, 100, 0.8]) # double gaussian fit
#print(f"Fitted parameters: mu1={m.values['mu1']:.2f}±{m.errors['mu1']:.2f}, sigma1={m.values['sigma1']:.2f}±{m.errors['sigma1']:.2f}, mu2={m.values['mu2']:.2f}±{m.errors['mu2']:.2f}, sigma2={m.values['sigma2']:.2f}±{m.errors['sigma2']:.2f}, alpha={m.values['alpha']:.2f}±{m.errors['alpha']:.2f}")
# Plot the data and the fitted function
plt.hist(x_data, binedges, density=True, weights=weights, alpha=0.6, color='g', label=f'Data (mean={sum(x_data * weights)/sum(weights):.2f})')
x_fit = np.linspace(*fit_range, 1000)
y_fit = utils.gaussian(x_fit, m.values['mu'], m.values['sigma'])
plt.plot(x_fit, y_fit, color='red', label=f"Fitted Gaussian: μ={m.values['mu']:.2f}±{m.errors['mu']:.2f}, σ={m.values['sigma']:.2f}±{m.errors['sigma']:.2f}")
plt.legend()
plt.xlabel('Edepo_reco - Edepo_true [MeV]')
plt.savefig(f"plots/Edepo_diff_{beampdg}.pdf")
plt.show()

plt.scatter(Edepo_true[combined_mask_mc], x_data, s=weights*10, marker='.', alpha=0.3, color='b', label='True beam matched MC after full selection')
plt.xlabel('Edepo_true [MeV]')
plt.ylabel('Edepo_reco - Edepo_true [MeV]')
plt.legend()
plt.xlim([0, 1200])
plt.ylim([-200, 200])
plt.grid(True)
plt.savefig(f"plots/Edepo_2D_{beampdg}.png")
plt.show()

### KEini
Eini_reco = processedVars_mc["reco_initial_energy"]
Eini_true = processedVars_mc["true_initial_energy"] # use processed_files/procVars_upEloss_{beampdg}.pkl
binedges = np.linspace(-200, 200, 100)
fit_range = [-100, 100]

x_data = (Eini_reco - Eini_true)[combined_mask_mc]
m = utils.fit_gaus_hist(x_data, weights, x_range=fit_range, initial_guesses=[0, 30])
print(f"Fitted parameters: mu={m.values['mu']:.2f}±{m.errors['mu']:.2f}, sigma={m.values['sigma']:.2f}±{m.errors['sigma']:.2f}")
# Plot the data and the fitted function
plt.hist(x_data, binedges, density=True, weights=weights, alpha=0.6, color='g', label=f'Data (mean={sum(x_data * weights)/sum(weights):.2f})')
x_fit = np.linspace(*fit_range, 1000)
y_fit = utils.gaussian(x_fit, m.values['mu'], m.values['sigma'])
plt.plot(x_fit, y_fit, color='red', label=f"Fitted Gaussian: μ={m.values['mu']:.2f}±{m.errors['mu']:.2f}, σ={m.values['sigma']:.2f}±{m.errors['sigma']:.2f}")
plt.legend()
plt.xlabel('KEini_reco - KEini_true [MeV]')
plt.savefig(f"plots/KEini_diff_{beampdg}.pdf")
plt.show()

plt.scatter(Eini_true[combined_mask_mc], x_data, s=weights*10, marker='.', alpha=0.3, color='b', label='True beam matched MC after full selection')
plt.xlabel('KEini_true [MeV]')
plt.ylabel('KEini_reco - KEini_true [MeV]')
plt.legend()
plt.xlim([0, 1200])
plt.ylim([-200, 200])
plt.grid(True)
plt.savefig(f"plots/KEini_2D_{beampdg}.png")
plt.show()

### KEend
Eend_reco = processedVars_mc["reco_end_energy"]
Eend_true = processedVars_mc["true_end_energy"] # use processed_files/procVars_upEloss_{beampdg}.pkl
binedges = np.linspace(-200, 200, 100)
fit_range = [-100, 100]

x_data = (Eend_reco - Eend_true)[combined_mask_mc]
m = utils.fit_gaus_hist(x_data, weights, x_range=fit_range, initial_guesses=[0, 30])
print(f"Fitted parameters: mu={m.values['mu']:.2f}±{m.errors['mu']:.2f}, sigma={m.values['sigma']:.2f}±{m.errors['sigma']:.2f}")
#m = utils.fit_doublegaus_hist(x_data, weights, x_range=fit_range, initial_guesses=[0, 30, 10, 100, 0.8]) # double gaussian fit
#print(f"Fitted parameters: mu1={m.values['mu1']:.2f}±{m.errors['mu1']:.2f}, sigma1={m.values['sigma1']:.2f}±{m.errors['sigma1']:.2f}, mu2={m.values['mu2']:.2f}±{m.errors['mu2']:.2f}, sigma2={m.values['sigma2']:.2f}±{m.errors['sigma2']:.2f}, alpha={m.values['alpha']:.2f}±{m.errors['alpha']:.2f}")
# Plot the data and the fitted function
plt.hist(x_data, binedges, density=True, weights=weights, alpha=0.6, color='g', label=f'Data (mean={sum(x_data * weights)/sum(weights):.2f})')
x_fit = np.linspace(*fit_range, 1000)
y_fit = utils.gaussian(x_fit, m.values['mu'], m.values['sigma'])
plt.plot(x_fit, y_fit, color='red', label=f"Fitted Gaussian: μ={m.values['mu']:.2f}±{m.errors['mu']:.2f}, σ={m.values['sigma']:.2f}±{m.errors['sigma']:.2f}")
plt.legend()
plt.xlabel('KEend_reco - KEend_true [MeV]')
plt.savefig(f"plots/KEend_diff_{beampdg}.pdf")
plt.show()

plt.scatter(Eend_true[combined_mask_mc], x_data, s=weights*10, marker='.', alpha=0.3, color='b', label='True beam matched MC after full selection')
plt.xlabel('KEend_true [MeV]')
plt.ylabel('KEend_reco - KEend_true [MeV]')
plt.legend()
plt.xlim([0, 1200])
plt.ylim([-200, 200])
plt.grid(True)
plt.savefig(f"plots/KEend_2D_{beampdg}.png")
plt.show()