import uproot
import selection

# pduneana_MC_20g4rw.root, PDSPProd4_data_1GeV_reco2_ntuple_v09_41_00_04.root
PDSP_ntuple = uproot.open("/Users/lyret/PDSPProd4_data_1GeV_reco2_ntuple_v09_41_00_04.root")
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
]
Nevents = 10000
eventset = pduneana.iterate(expressions=variables_to_load, library="np") # entry_stop=Nevents, step_size=20
Nevt_tot = 0
Nevt_isPar = 0
Nevt_selected = 0
for evt in eventset:
    Nevt_tot += len(evt["event"])

    pionp = selection.Particle(211, 139.57)
    pionp.SetCandidatePDGlist([-13, 13, 211])
    
    proton = selection.Particle(2212, 938.272)
    proton.SetCandidatePDG(2212)

    anapar = proton

    mask_SelectedPart =  anapar.IsSelectedPart(evt)
    evt_processed = anapar.PassSelection(evt)

    Nevt_isPar += len(evt_processed[mask_SelectedPart])
    Nevt_selected += len(evt_processed[mask_SelectedPart][evt_processed[mask_SelectedPart]])

    print(f"{Nevt_tot} events processed.")

print(Nevt_tot, Nevt_isPar, Nevt_selected)