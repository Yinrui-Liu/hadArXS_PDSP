from hadana.packages import *
import hadana.parameters as parameters

use_real_data = True
beampdg = 211
binedges = np.linspace(0, 280, 50)
xlabel = r"Reconstructed track length [cm]" ### also edit below the variable to plot
procdataname = f"processed_files/procVars_pidata.pkl" # processed_files/procVars_pidata.pkl
procMCname = f"processed_files/procVars_piMC.pkl" # processed_files/procVars_piMC.pkl

partypedict_pionp = {
    0: "Data", 
    1: "Pion inelastic", 
    2: "Pion decay", 
    3: "Muon", 
    4: "misID:cosmic", 
    5: "misID:proton", 
    6: "misID:pion", 
    7: "misID:muon", 
    8: "misID:e/γ", 
    9: "misID:other", 
}
partypedict_proton = {
    0: "Data", 
    1: "Proton inelatic", 
    2: "Stopping proton", 
    3: "misID:cosmic", 
    4: "misID:proton", 
    5: "misID:pion", 
    6: "misID:muon", 
    7: "misID:e/γ", 
    8: "misID:other", 
}
parcolordict = {
    "Pion inelastic": "r",
    "Proton inelatic": "r",
    "Pion decay": "orange",
    "Stopping proton": "orange",
    "Muon": "springgreen",
    "misID:cosmic": "deepskyblue",
    "misID:proton": "darkviolet",
    "misID:pion": "hotpink",
    "misID:muon": "green",
    "misID:e/γ": "yellow",
    "misID:other": "peru",
}
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
PDSP_ntuple = uproot.open(f"input_files/PDSPProd4_data_1GeV_reco2_ntuple_v09_41_00_04.root")
pduneana_data = PDSP_ntuple["pduneana/beamana"]

if beampdg == 211:
    pardict = partypedict_pionp
elif beampdg == 2212:
    pardict = partypedict_proton

if use_real_data:
    with open(procdataname, 'rb') as datafile:
        processedVars_data = pickle.load(datafile)
    mask_SelectedPart_data = processedVars_data["mask_SelectedPart"]
    mask_FullSelection_data = processedVars_data["mask_FullSelection"]
    combined_mask_data = mask_SelectedPart_data & mask_FullSelection_data
    particle_type_data = processedVars_data["particle_type"]
    reweight_data = processedVars_data["reweight"]

with open(procMCname, 'rb') as mcfile:
    processedVars_mc = pickle.load(mcfile)
mask_SelectedPart_mc = processedVars_mc["mask_SelectedPart"]
mask_FullSelection_mc = processedVars_mc["mask_FullSelection"]
combined_mask_mc = mask_SelectedPart_mc & mask_FullSelection_mc
particle_type_mc = processedVars_mc["particle_type"]
reweight_mc = processedVars_mc["reweight"]
#reweight_mc = np.ones_like(reweight_mc) # no reweighting

### edit here the variable to plot
if False: # to draw the angle variable
    parBQ = parameters.pionBQ if beampdg==211 else parameters.protonBQ if beampdg==2212 else None

    pt0 = np.array([pduneana_mc["reco_beam_calo_startX"], pduneana_mc["reco_beam_calo_startY"], pduneana_mc["reco_beam_calo_startZ"]])
    pt1 = np.array([pduneana_mc["reco_beam_calo_endX"], pduneana_mc["reco_beam_calo_endY"], pduneana_mc["reco_beam_calo_endZ"]])
    dir = pt1 - pt0
    norms = np.linalg.norm(dir, axis=0)
    norms = np.where(norms == 0, 1, norms) # to prevent 0 due to reco_beam_calo_start==-999 and reco_beam_calo_end==-999
    dir /= norms
    dir = np.transpose(dir)
    beamdir_mc = np.array([np.cos(np.deg2rad(parBQ["beam_angleX_mc"])), np.cos(np.deg2rad(parBQ["beam_angleY_mc"])), np.cos(np.deg2rad(parBQ["beam_angleZ_mc"]))])
    beamdir_mc /= np.linalg.norm(beamdir_mc)
    beamdir = [beamdir_mc.tolist()]*len(dir)
    beam_costh_mc = np.einsum('ij,ij->i', dir, beamdir)

    pt0 = np.array([pduneana_data["reco_beam_calo_startX"], pduneana_data["reco_beam_calo_startY"], pduneana_data["reco_beam_calo_startZ"]])
    pt1 = np.array([pduneana_data["reco_beam_calo_endX"], pduneana_data["reco_beam_calo_endY"], pduneana_data["reco_beam_calo_endZ"]])
    dir = pt1 - pt0
    norms = np.linalg.norm(dir, axis=0)
    norms = np.where(norms == 0, 1, norms) # to prevent 0 due to reco_beam_calo_start==-999 and reco_beam_calo_end==-999
    dir /= norms
    dir = np.transpose(dir)
    beamdir_data = np.array([np.cos(np.deg2rad(parBQ["beam_angleX_data"])), np.cos(np.deg2rad(parBQ["beam_angleY_data"])), np.cos(np.deg2rad(parBQ["beam_angleZ_data"]))])
    beamdir_data /= np.linalg.norm(beamdir_data)
    beamdir = [beamdir_data.tolist()]*len(dir)
    beam_costh_data = np.einsum('ij,ij->i', dir, beamdir)

varhist_mc = processedVars_mc["reco_track_length"] # examples: processedVars_mc["reco_track_length"], np.array(pduneana_mc["beam_inst_P"]), (np.array(pduneana_mc["reco_beam_calo_startX"])-parameters.pionBQ["beam_startX_mc"])/parameters.pionBQ["beam_startX_rms_mc"], np.where(np.array(pduneana_mc["reco_beam_vertex_nHits"]) != 0, np.array(pduneana_mc["reco_beam_vertex_michel_score_weight_by_charge"]), -999), np.where(np.array([len(calo_wire) != 0 for calo_wire in np.array(pduneana_mc["reco_beam_calo_wire"])]), np.array(pduneana_mc["reco_beam_Chi2_proton"]) / np.array(pduneana_mc["reco_beam_Chi2_ndof"]), -1)
varhist_data = processedVars_data["reco_track_length"] # examples: processedVars_data["reco_track_length"], np.array(pduneana_data["beam_inst_P"]), (np.array(pduneana_data["reco_beam_calo_startX"])-parameters.pionBQ["beam_startX_data"])/parameters.pionBQ["beam_startX_rms_data"], np.where(np.array(pduneana_data["reco_beam_vertex_nHits"]) != 0, np.array(pduneana_data["reco_beam_vertex_michel_score_weight_by_charge"]), -999), np.where(np.array([len(calo_wire) != 0 for calo_wire in np.array(pduneana_data["reco_beam_calo_wire"])]), np.array(pduneana_data["reco_beam_Chi2_proton"]) / np.array(pduneana_data["reco_beam_Chi2_ndof"]), -1)

# draw the data points and stacked MC histograms by event type in the comparison plot
divided_vars_mc, divided_weights_mc = utils.divide_vars_by_partype(varhist_mc, particle_type_mc, mask=combined_mask_mc, weight=reweight_mc)
Nmc_sep = [sum(i) for i in divided_weights_mc[1:]]
Nmc = sum(Nmc_sep)
if use_real_data:
    divided_vars_data, divided_weights_data = utils.divide_vars_by_partype(varhist_data, particle_type_data, mask=combined_mask_data, weight=reweight_data)
    hists_data, hists_err_data, _ = utils.get_vars_hists(divided_vars_data, divided_weights_data, binedges)
    Ndata = sum(divided_weights_data[0])
else:
    hists_data, hists_err_data, _ = utils.get_vars_hists(divided_vars_mc, divided_weights_mc, binedges)
    Ndata = sum(divided_weights_mc[0])
hists_data = hists_data[0]
hists_err_data = hists_err_data[0]
hists_mc, hists_err_mc_sep, _ = utils.get_vars_hists(divided_vars_mc[1:], divided_weights_mc[1:], binedges)
hists_err_mc = np.zeros_like(hists_err_mc_sep[0])
for err in hists_err_mc_sep:
    hists_err_mc += (err*err)
hists_err_mc = np.sqrt(hists_err_mc)

print(f"Ndata = {Ndata:.1f}, Nmc = {Nmc:.1f}")
ax1 = plt.axes([0.11, 0.24, 0.86, 0.74])
ax2 = plt.axes([0.11, 0.09, 0.86, 0.12])
bincenters = (binedges[:-1]+binedges[1:])/2
ax1.errorbar(bincenters, hists_data, yerr=hists_err_data, fmt='o', color='k', markersize=1, label=f"Data {Ndata:.0f}")
MC_data_scale = Ndata / Nmc
binmc, _, _ = ax1.hist(divided_vars_mc[1:], binedges, weights=[i*MC_data_scale for i in divided_weights_mc[1:]], label=[f'{pardict[i+1]} {MC_data_scale*Nmc_sep[i]:.0f}' for i in range(len(divided_vars_mc[1:]))], color=[f'{parcolordict[pardict[i+1]]}' for i in range(len(divided_vars_mc[1:]))], stacked=True) # binmc is cumulative hists_mc

ratio_err = hists_data/binmc[-1] * np.sqrt(np.power(hists_err_mc/binmc[-1], 2) + np.power(hists_err_data/hists_data, 2)) # error of the ratio
ax2.errorbar(bincenters, hists_data/binmc[-1], yerr=ratio_err, fmt='o', color='k', markersize=1)
ax2.plot(binedges, np.ones_like(binedges), 'r:')

ax1.set_xticks(np.arange(binedges[0], binedges[-1], 10), minor=True)
ax1.set_xticklabels([])
ax1.set_xlim([binedges[0], binedges[-1]])
ax1.set_ylabel("Weighted counts")
#ax1.set_yscale('log')
ax1.legend()
ax2.set_xticks(np.arange(binedges[0], binedges[-1], 10), minor=True)
ax2.set_xlim([binedges[0], binedges[-1]])
ax2.set_xlabel(xlabel)
ax2.set_yticks([0,1,2])
ax2.set_ylim([0, 2])
ax2.set_ylabel("Data/MC")
#plt.savefig(f"plots/vardist_{beampdg}.pdf")
plt.show()