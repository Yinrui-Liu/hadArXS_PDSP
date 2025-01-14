import pickle
import numpy as np

# Load the preprocessed .pkl file
pkl_file = "processed_files/procVars_piMC_100MeVBin_6Proton.pkl"
with open(pkl_file, "rb") as file:
    processedVars = pickle.load(file)

# Extract necessary variables
particle_type = processedVars["particle_type"]
mask_SelectedPart = processedVars["mask_SelectedPart"]
mask_FullSelection = processedVars["mask_FullSelection"]

# Combine masks with AND condition
combined_mask = mask_SelectedPart & mask_FullSelection

# Interaction types for categorization
interaction_types = {
    0: "Fake Data",
    1: "Pion Inelastic",
    2: "Pion Decay",
    3: "Muon",
    4: "MisID: Cosmic",
    5: "MisID: Proton",
    6: "MisID: Pion",
    7: "MisID: Muon",
    8: "MisID: Electron/Gamma",
    9: "MisID: Other",
}

# Filter events that pass the combined selection cut
events_after_combined_cut = particle_type[combined_mask]

# Count events by type after the combined cut
event_counts = {itype: np.sum(events_after_combined_cut == itype) for itype in interaction_types}

# Print results
print("Event counts after combined (AND) cut:")
for itype, count in event_counts.items():
    print(f"  {interaction_types[itype]}: {count}")
