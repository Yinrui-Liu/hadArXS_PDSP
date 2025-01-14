import pickle
import numpy as np

# Load the processed variables from the pickle file
pkl_file = "processed_files/procVars_piMC_100MeVBin_6Proton.pkl"
with open(pkl_file, "rb") as file:
    processedVars = pickle.load(file)

# Extract necessary variables
particle_type = processedVars["particle_type"]
mask_SelectedPart = processedVars["mask_SelectedPart"]
mask_FullSelection = processedVars["mask_FullSelection"]
true_initial_energy = processedVars["true_initial_energy"]

# Define bin edges for KE
binedges = np.linspace(0, 280, 50)
bincenters = (binedges[:-1] + binedges[1:]) / 2

# Print variables
print("Particle Type (first 10 values):", particle_type[:10])
print("Mask Selected Part (first 10 values):", mask_SelectedPart[:10])
print("Mask Full Selection (first 10 values):", mask_FullSelection[:10])
print("True Initial Energy (first 10 values):", true_initial_energy[:10])
print("Bin Edges:", binedges)
print("Bin Centers:", bincenters)

# Additional diagnostic information
total_events_per_bin, _ = np.histogram(true_initial_energy[particle_type == 1], bins=binedges)
selected_events_per_bin, _ = np.histogram(
    true_initial_energy[(particle_type == 1) & mask_FullSelection], bins=binedges
)
efficiency = selected_events_per_bin / total_events_per_bin
efficiency = np.nan_to_num(efficiency)  # Replace NaN with 0 for empty bins

print("Total Events Per Bin:", total_events_per_bin)
print("Selected Events Per Bin:", selected_events_per_bin)
print("Efficiency:", efficiency)
