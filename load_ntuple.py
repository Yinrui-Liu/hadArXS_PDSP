from hadana.packages import *
from hadana.processor import Processor
import hadana.selection as selection
import hadana.MC_reweight as reweight


# pduneana_MC_20g4rw, PDSPProd4_data_1GeV_reco2_ntuple_v09_41_00_04, PDSPProd4_data_1GeV_reco2_ntuple_AltSCEData
PDSP_ntuple_name = "PDSPProd4_data_1GeV_reco2_ntuple_v09_41_00_04"
beamPDG = 211
outfilename = "processed_files/procVars_piPDSP.pkl"
Nevents = None # change Nevents for smaller sample size


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
    "true_beam_daughter_PDG",
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

eventset = Processor(pduneana, particle, isMC, selection=[True,True,True,True,True,True], fake_data=False) # fake_data is False for all true MC, True for all fake data, None for half-half
eventset.LoadVariables(variables_to_load)
eventset.ProcessEvent(Nevents=Nevents)
processedVars = eventset.GetOutVarsDict()

reweight = reweight.cal_bkg_reweight(processedVars) * reweight.cal_momentum_reweight(processedVars)
processedVars["reweight"] = reweight

with open(outfilename, 'wb') as procfile:
    pickle.dump(processedVars, procfile)


'''
print(reweight.cal_momentum_reweight(processedVars))
print(reweight.cal_bkg_reweight(processedVars))
print(reweight.cal_g4rw(processedVars, 0.9))

combined_mask = mask_SelectedPart & mask_FullSelection
print(len(eventset.true_initial_energy[combined_mask & (eventset.particle_type==1)]))

plt.hist(eventset.true_initial_energy[combined_mask], bins=np.arange(0,1000,30), alpha=0.3, label="KEi")
plt.hist(eventset.true_end_energy[combined_mask], bins=np.arange(0,1000,30), alpha=0.3, label="KEf")
plt.legend()
plt.savefig("plots/test_t.png")
plt.clf()

plt.hist(eventset.reco_initial_energy[combined_mask], bins=np.arange(0,1000,30), alpha=0.3, label="KEi")
plt.hist(eventset.reco_end_energy[combined_mask], bins=np.arange(0,1000,30), alpha=0.3, label="KEf")
plt.legend()
plt.savefig("plots/test_r.png")
plt.clf()
'''