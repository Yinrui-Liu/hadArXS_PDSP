from hadana.packages import *
import hadana.get_histograms as get_hists


use_real_data = True
beampdg = 2212
binedges = np.linspace(0, 600, 100)
xlabel = "reco_end_energy [MeV]"

partypedict_pionp = {
    0: "Data", 
    1: "PiInel", 
    2: "PiDecay", 
    3: "Muon", 
    4: "misID:cosmic", 
    5: "misID:p", 
    6: "misID:pi", 
    7: "misID:mu", 
    8: "misID:e/γ", 
    9: "misID:other", 
}
partypedict_proton = {
    0: "Data", 
    1: "PInel", 
    2: "PElas", 
    3: "misID:cosmic", 
    4: "misID:p", 
    5: "misID:pi", 
    6: "misID:mu", 
    7: "misID:e/γ", 
    8: "misID:other", 
}
parcolordict = {
    "PiInel": "firebrick",
    "PInel": "firebrick",
    "PiDecay": "orange",
    "PElas": "orange",
    "Muon": "springgreen",
    "misID:cosmic": "deepskyblue",
    "misID:p": "darkviolet",
    "misID:pi": "hotpink",
    "misID:mu": "green",
    "misID:e/γ": "yellow",
    "misID:other": "peru",
}

if beampdg == 211:
    infile = "pi"
    pardict = partypedict_pionp
elif beampdg == 2212:
    infile = "p"
    pardict = partypedict_proton

if use_real_data:
    with open(f'processed_files/procVars_{infile}data.pkl', 'rb') as datafile:
        processedVars_data = pickle.load(datafile)
    reco_initial_energy_data = processedVars_data["reco_initial_energy"]
    reco_end_energy_data = processedVars_data["reco_end_energy"]
    mask_SelectedPart_data = processedVars_data["mask_SelectedPart"]
    mask_FullSelection_data = processedVars_data["mask_FullSelection"]
    combined_mask_data = mask_SelectedPart_data & mask_FullSelection_data
    particle_type_data = processedVars_data["particle_type"]
    reweight_data = processedVars_data["reweight"]
    reco_trklen_data = processedVars_data["reco_track_length"]
    reco_sigflag_data = processedVars_data["reco_sigflag"]
    reco_containing_data = processedVars_data["reco_containing"]

with open(f'processed_files/procVars_{infile}MC.pkl', 'rb') as mcfile:
    processedVars_mc = pickle.load(mcfile)
reco_initial_energy_mc = processedVars_mc["reco_initial_energy"]
reco_end_energy_mc = processedVars_mc["reco_end_energy"]
mask_SelectedPart_mc = processedVars_mc["mask_SelectedPart"]
mask_FullSelection_mc = processedVars_mc["mask_FullSelection"]
combined_mask_mc = mask_SelectedPart_mc & mask_FullSelection_mc
particle_type_mc = processedVars_mc["particle_type"]
reweight_mc = processedVars_mc["reweight"]
reco_trklen_mc = processedVars_mc["reco_track_length"]
reco_sigflag_mc = processedVars_mc["reco_sigflag"]
reco_containing_mc = processedVars_mc["reco_containing"]

divided_vars_mc, divided_weights_mc = get_hists.divide_vars_by_partype(reco_end_energy_mc, particle_type_mc, mask=combined_mask_mc, weight=reweight_mc)
Nmc_sep = [sum(i) for i in divided_weights_mc[1:]]
Nmc = sum(Nmc_sep)
if use_real_data:
    divided_vars_data, divided_weights_data = get_hists.divide_vars_by_partype(reco_end_energy_data, particle_type_data, mask=combined_mask_data, weight=reweight_data)
    hists_data, hists_err_data, _ = get_hists.get_vars_hists(divided_vars_data, divided_weights_data, binedges)
    Ndata = sum(divided_weights_data[0])
else:
    hists_data, hists_err_data, _ = get_hists.get_vars_hists(divided_vars_mc, divided_weights_mc, binedges)
    Ndata = sum(divided_weights_mc[0])

print(f"Ndata = {Ndata:.1f}, Nmc = {Nmc:.1f}")
plt.errorbar((binedges[:-1]+binedges[1:])/2, hists_data[0], yerr=hists_err_data[0], fmt='o', color='k', markersize=1, label=f"Data {Ndata:.0f}")
MC_data_scale = Ndata / Nmc
plt.hist(divided_vars_mc[1:], binedges, weights=[i*MC_data_scale for i in divided_weights_mc[1:]], label=[f'{pardict[i+1]} {MC_data_scale*Nmc_sep[i]:.0f}' for i in range(len(divided_vars_mc[1:]))], color=[f'{parcolordict[pardict[i+1]]}' for i in range(len(divided_vars_mc[1:]))], stacked=True)

plt.xlim([binedges[0], binedges[-1]])
plt.xlabel(xlabel)
plt.ylabel("Weighted counts")
plt.legend()
#plt.savefig("plots/test_plot.pdf")
plt.show()