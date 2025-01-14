import uproot

# Path to the ROOT file
root_file_path = "/media/sf_Saikat_sharedfolder/ProtoDUNE_root_files/PDSPProd4a_MC_2GeV_reco1_sce_datadriven_v1_ntuple_v09_41_00_03.root_split_1.root"

# Name of the TTree
tree_name = "beamana"

# List of variables to extract
variables_to_load = [
    "event",  # Event number (optional, if available)
    "true_beam_PDG",  # True particle PDG
    "reco_beam_true_byE_matched",  # Whether the reco track matches the true particle
    "reco_beam_true_byE_PDG",  # Reconstructed particle PDG
    "true_beam_endProcess",  # True particle interaction process
]

# Open the ROOT file and load the TTree
with uproot.open(root_file_path) as file:
    tree = file[tree_name]

    # Iterate over the events in the TTree
    for data in tree.iterate(variables_to_load, library="np"):
        # Loop over the batch of events
        for i in range(len(data["true_beam_PDG"])):
            event_number = data["event"][i]
            true_pdg = data["true_beam_PDG"][i]
            reco_matched = data["reco_beam_true_byE_matched"][i]
            reco_pdg = data["reco_beam_true_byE_PDG"][i]
            end_process = data["true_beam_endProcess"][i]

            # Print event-by-event information
            print(f"Event: {event_number}")
            print(f"  True Particle PDG: {true_pdg}")
            print(f"  Reco Matched: {reco_matched}")
            print(f"  Reco PDG: {reco_pdg}")
            print(f"  End Process: {end_process}")
            print("-" * 40)
