from hadana.packages import *
from hadana.processor import Processor
import hadana.selection as selection
import hadana.MC_reweight as reweight


# pduneana_MC_20g4rw, PDSPProd4_data_1GeV_reco2_ntuple_v09_41_00_04
PDSP_ntuple_name = "pduneana_MC_20g4rw"
beamPDG = 211
Nevents = None


PDSP_ntuple = uproot.open(f"input_files/{PDSP_ntuple_name}.root")
if "MC" in PDSP_ntuple_name:
    isMC = True
else:
    isMC = False
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

eventset = Processor(pduneana, particle, isMC, selection=[True,True,False,False,False,False]) # selection up to beam quality cuts
eventset.LoadVariables(variables_to_load)
eventset.ProcessEvent(Nevents=Nevents)
processedVars = eventset.GetOutVarsDict()

weights = reweight.cal_bkg_reweight(processedVars) * reweight.cal_momentum_reweight(processedVars)
processedVars["reweight"] = weights

mask_SelectedPart = processedVars["mask_SelectedPart"]
mask_FullSelection = processedVars["mask_FullSelection"]
mask = (mask_SelectedPart & mask_FullSelection)[:Nevents]
weights = weights[:Nevents][mask]


### start x
x_data = np.array(pduneana["reco_beam_calo_startX"])[:Nevents][mask] # edit here the var name
m = utils.fit_gaus_hist(x_data, weights, x_range=[-40, -20], initial_guesses=[-30, 5]) # edit here the fit range and initial guess
print(f"Fitted parameters: mu={m.values['mu']:.2f}±{m.errors['mu']:.2f}, sigma={m.values['sigma']:.2f}±{m.errors['sigma']:.2f}")
# Plot the data and the fitted function
xrange_draw = [-80, 20] # edit here the plot range
plt.hist(x_data, bins=100, range=xrange_draw, density=True, weights=weights, alpha=0.6, color='g', label='Data')
x_fit = np.linspace(*xrange_draw, 1000)
y_fit = utils.gaussian(x_fit, m.values['mu'], m.values['sigma'])
plt.plot(x_fit, y_fit, color='red', label='Fitted Gaussian')
plt.legend()
plt.xlabel("reco_beam_calo_startX [cm]") # edit here the xlabel
plt.show()

### start y
x_data = np.array(pduneana["reco_beam_calo_startY"])[:Nevents][mask]
m = utils.fit_gaus_hist(x_data, weights, x_range=[412, 432], initial_guesses=[422, 5])
print(f"Fitted parameters: mu={m.values['mu']:.2f}±{m.errors['mu']:.2f}, sigma={m.values['sigma']:.2f}±{m.errors['sigma']:.2f}")
# Plot the data and the fitted function
xrange_draw = [370, 470]
plt.hist(x_data, bins=100, range=xrange_draw, density=True, weights=weights, alpha=0.6, color='g', label='Data')
x_fit = np.linspace(*xrange_draw, 1000)
y_fit = utils.gaussian(x_fit, m.values['mu'], m.values['sigma'])
plt.plot(x_fit, y_fit, color='red', label='Fitted Gaussian')
plt.legend()
plt.xlabel("reco_beam_calo_startY [cm]")
plt.show()

### start z
x_data = np.array(pduneana["reco_beam_calo_startZ"])[:Nevents][mask]
if isMC:
    m = utils.fit_gaus_hist(x_data, weights, x_range=[-0.4, 0.6], initial_guesses=[0.1, 0.2])
else:
    m = utils.fit_gaus_hist(x_data, weights, x_range=[1, 5], initial_guesses=[3, 1])
print(f"Fitted parameters: mu={m.values['mu']:.3f}±{m.errors['mu']:.3f}, sigma={m.values['sigma']:.3f}±{m.errors['sigma']:.3f}")
# Plot the data and the fitted function
xrange_draw = [-3, 9]
plt.hist(x_data, bins=100, range=xrange_draw, density=True, weights=weights, alpha=0.6, color='g', label='Data')
x_fit = np.linspace(*xrange_draw, 1000)
y_fit = utils.gaussian(x_fit, m.values['mu'], m.values['sigma'])
plt.plot(x_fit, y_fit, color='red', label='Fitted Gaussian')
plt.legend()
plt.xlabel("reco_beam_calo_startZ [cm]")
plt.show()

### inst x
x_data = np.array(pduneana["beam_inst_X"])[:Nevents][mask]
m = utils.fit_gaus_hist(x_data, weights, x_range=[-40, -20], initial_guesses=[-30, 5])
print(f"Fitted parameters: mu={m.values['mu']:.2f}±{m.errors['mu']:.2f}, sigma={m.values['sigma']:.2f}±{m.errors['sigma']:.2f}")
# Plot the data and the fitted function
xrange_draw = [-80, 20]
plt.hist(x_data, bins=100, range=xrange_draw, density=True, weights=weights, alpha=0.6, color='g', label='Data')
x_fit = np.linspace(*xrange_draw, 1000)
y_fit = utils.gaussian(x_fit, m.values['mu'], m.values['sigma'])
plt.plot(x_fit, y_fit, color='red', label='Fitted Gaussian')
plt.legend()
plt.xlabel("beam_inst_X [cm]")
plt.show()

### inst y
x_data = np.array(pduneana["beam_inst_Y"])[:Nevents][mask]
m = utils.fit_gaus_hist(x_data, weights, x_range=[412, 432], initial_guesses=[422, 5])
print(f"Fitted parameters: mu={m.values['mu']:.2f}±{m.errors['mu']:.2f}, sigma={m.values['sigma']:.2f}±{m.errors['sigma']:.2f}")
# Plot the data and the fitted function
xrange_draw = [370, 470]
plt.hist(x_data, bins=100, range=xrange_draw, density=True, weights=weights, alpha=0.6, color='g', label='Data')
x_fit = np.linspace(*xrange_draw, 1000)
y_fit = utils.gaussian(x_fit, m.values['mu'], m.values['sigma'])
plt.plot(x_fit, y_fit, color='red', label='Fitted Gaussian')
plt.legend()
plt.xlabel("beam_inst_Y [cm]")
plt.show()

### angle variables
reco_beam_calo_startX = np.array(pduneana["reco_beam_calo_startX"])[:Nevents]
reco_beam_calo_startY = np.array(pduneana["reco_beam_calo_startY"])[:Nevents]
reco_beam_calo_startZ = np.array(pduneana["reco_beam_calo_startZ"])[:Nevents]
reco_beam_calo_endX = np.array(pduneana["reco_beam_calo_endX"])[:Nevents]
reco_beam_calo_endY = np.array(pduneana["reco_beam_calo_endY"])[:Nevents]
reco_beam_calo_endZ = np.array(pduneana["reco_beam_calo_endZ"])[:Nevents]
pt0 = np.array([reco_beam_calo_startX, reco_beam_calo_startY, reco_beam_calo_startZ])
pt1 = np.array([reco_beam_calo_endX, reco_beam_calo_endY, reco_beam_calo_endZ])
dir = pt1 - pt0
norms = np.linalg.norm(dir, axis=0)[mask]
dir = np.transpose(dir)[mask][norms!=0]
norms = norms[norms!=0]
dir = np.transpose(dir)/norms


### angle x
x_data = np.arccos(dir[0]) * 180/np.pi
m = utils.fit_gaus_hist(x_data, weights, x_range=[97, 105], initial_guesses=[101, 3]) # may update to double gaussian fit in the future
print(f"Fitted parameters: mu={m.values['mu']:.2f}±{m.errors['mu']:.2f}, sigma={m.values['sigma']:.2f}±{m.errors['sigma']:.2f}")
# Plot the data and the fitted function
xrange_draw = [70, 130]
plt.hist(x_data, bins=100, range=xrange_draw, density=True, weights=weights, alpha=0.6, color='g', label='Data')
x_fit = np.linspace(*xrange_draw, 1000)
y_fit = utils.gaussian(x_fit, m.values['mu'], m.values['sigma'])
plt.plot(x_fit, y_fit, color='red', label='Fitted Gaussian')
plt.legend()
plt.xlabel("theta_x (deg)")
plt.show()

### angle y
x_data = np.arccos(dir[1]) * 180/np.pi
m = utils.fit_gaus_hist(x_data, weights, x_range=[96, 108], initial_guesses=[102, 4])
print(f"Fitted parameters: mu={m.values['mu']:.2f}±{m.errors['mu']:.2f}, sigma={m.values['sigma']:.2f}±{m.errors['sigma']:.2f}")
# Plot the data and the fitted function
xrange_draw = [70, 130]
plt.hist(x_data, bins=100, range=xrange_draw, density=True, weights=weights, alpha=0.6, color='g', label='Data')
x_fit = np.linspace(*xrange_draw, 1000)
y_fit = utils.gaussian(x_fit, m.values['mu'], m.values['sigma'])
plt.plot(x_fit, y_fit, color='red', label='Fitted Gaussian')
plt.legend()
plt.xlabel("theta_y (deg)")
plt.show()

### angle z
x_data = np.arccos(dir[2]) * 180/np.pi
m = utils.fit_gaus_hist(x_data, weights, x_range=[12, 22], initial_guesses=[17, 3.5])
print(f"Fitted parameters: mu={m.values['mu']:.2f}±{m.errors['mu']:.2f}, sigma={m.values['sigma']:.2f}±{m.errors['sigma']:.2f}")
# Plot the data and the fitted function
xrange_draw = [-5, 40]
plt.hist(x_data, bins=100, range=xrange_draw, density=True, weights=weights, alpha=0.6, color='g', label='Data')
x_fit = np.linspace(*xrange_draw, 1000)
y_fit = utils.gaussian(x_fit, m.values['mu'], m.values['sigma'])
plt.plot(x_fit, y_fit, color='red', label='Fitted Gaussian')
plt.legend()
plt.xlabel("theta_z (deg)")
plt.show()