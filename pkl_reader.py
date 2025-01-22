import pickle

# Path to your .pkl file
pkl_file_path = "/home/saikat/HadANA_Versions/hadArXS_PDSP/processed_files/procVars_piMC_100MeVBin_6Proton.pkl"

# Load the .pkl file
with open(pkl_file_path, "rb") as file:
    processedVars = pickle.load(file)

# Check the keys in the dictionary
print("Keys in the processedVars dictionary:")
print(processedVars.keys())

# Define the number of entries to print
num_entries = 10

# Print the first few entries for specific variables
variables_to_check = ["true_beam_PDG", "true_beam_endProcess", "mask_SelectedPart", "mask_FullSelection", "mask_TrueSignal"]

print(f"\nDisplaying the first {num_entries} entries for selected variables:")
for var in variables_to_check:
    if var in processedVars:
        print(f"\nVariable: {var}")
        print(processedVars[var][:num_entries])  # Print the first few entries
    else:
        print(f"\nVariable {var} not found in the .pkl file")
