from packages import *
from processor import Processor
import selection

# pduneana_MC_20g4rw, PDSPProd4_data_1GeV_reco2_ntuple_v09_41_00_04
PDSP_ntuple_name = "pduneana_MC_20g4rw"
PDSP_ntuple = uproot.open(f"/Users/lyret/{PDSP_ntuple_name}.root")
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
]

pionp = selection.Particle(211, 139.57)
pionp.SetCandidatePDGlist([-13, 13, 211])

proton = selection.Particle(2212, 938.272)
proton.SetCandidatePDG(2212)

eventset = Processor(pduneana, proton, isMC)
eventset.LoadVariables(variables_to_load)
eventset.ProcessEvent(Nevents=None)

mask_SelectedPart = np.array(eventset.mask_SelectedPart, dtype=bool)
mask_FullSelection = np.array(eventset.mask_FullSelection, dtype=bool)
combined_mask = mask_SelectedPart & mask_FullSelection


'''
print(len(eventset.true_initial_energy[combined_mask & (eventset.particle_type==1)]))

plt.hist(eventset.true_initial_energy[combined_mask], bins=np.arange(0,1000,30), alpha=0.3, label="KEi")
plt.hist(eventset.true_end_energy[combined_mask], bins=np.arange(0,1000,30), alpha=0.3, label="KEf")
plt.legend()
plt.savefig("test_t.png")
plt.clf()

plt.hist(eventset.reco_initial_energy[combined_mask], bins=np.arange(0,1000,30), alpha=0.3, label="KEi")
plt.hist(eventset.reco_end_energy[combined_mask], bins=np.arange(0,1000,30), alpha=0.3, label="KEf")
plt.legend()
plt.savefig("test_r.png")
plt.clf()
'''