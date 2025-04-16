from hadana.packages import *
import ROOT
import hadana.slicing_method as slicing
import hadana.multiD_mapping as multiD
import hadana.parameters as parameters


beamPDG = 211
procfilename = "processed_files/procVars_piMC_test.pkl"
respfilename = "processed_files/response_pi_test.pkl"


with open(procfilename, 'rb') as procfile:
    processedVars = pickle.load(procfile)

if beamPDG == 211:
    true_bins = parameters.true_bins_pionp
    meas_bins = parameters.meas_bins_pionp
elif beamPDG == 2212:
    true_bins = parameters.true_bins_proton
    meas_bins = parameters.meas_bins_proton
Ntrueinttype = 6 # number of true interaction types ["noint", "inel", "cex", "dcex", "abs", "prod"]
Nrecointtype = 6 # number of reco interaction types ["noint", "inel", "cex", "dcex", "abs", "prod"]
signal_int_type = 4 # same as the selected_type

mask_TrueSignal = processedVars["mask_TrueSignal"]
mask_SelectedPart = processedVars["mask_SelectedPart"]
combined_true_mask = mask_SelectedPart & mask_TrueSignal
mask_FullSelection = processedVars["mask_FullSelection"]
beam_matched = processedVars["reco_beam_true_byE_matched"]
combined_reco_mask = mask_FullSelection & np.array(beam_matched, dtype=bool)
reco_initial_energy = processedVars["reco_initial_energy"]
reco_end_energy = processedVars["reco_end_energy"]
reco_channel = processedVars["reco_int_type"]
reco_containing = processedVars["reco_containing"]
particle_type = processedVars["particle_type"]
reweight = processedVars["reweight"]

true_initial_energy = processedVars["true_initial_energy"]
true_end_energy = processedVars["true_end_energy"]
true_channel = processedVars["true_int_type"]
true_containing = processedVars["true_containing"]

particle_type_bool = np.where(particle_type==20, 0, 1) # 0 for fake data, 1 for truth MC
divided_trueEini, divided_weights = utils.divide_vars_by_partype(true_initial_energy, particle_type_bool, mask=combined_true_mask, weight=reweight, alltypes=[0,1])
divided_trueEend, divided_weights = utils.divide_vars_by_partype(true_end_energy, particle_type_bool, mask=combined_true_mask, weight=reweight, alltypes=[0,1])
divided_truechnl, divided_weights = utils.divide_vars_by_partype(true_channel, particle_type_bool, mask=combined_true_mask, weight=reweight, alltypes=[0,1])
divided_trueisct, divided_weights = utils.divide_vars_by_partype(true_containing, particle_type_bool, mask=combined_true_mask, weight=reweight, alltypes=[0,1])
true_Eini = divided_trueEini[1]
true_Eend = divided_trueEend[1]
true_chnl = divided_truechnl[1]
true_isCt = divided_trueisct[1]
true_weight = divided_weights[1]
Ntruebins, Ntruebins_3D, true_cKE, true_wKE = utils.set_bins(true_bins)
Nmeasbins = len(meas_bins)
true_SIDini, true_SIDend, true_SIDint_ex = slicing.get_sliceID_histograms(true_Eini, true_Eend, true_chnl==signal_int_type, true_isCt, true_bins)
true_Nini, true_Nend, true_Nint_ex, true_Ninc = slicing.derive_energy_histograms(true_SIDini, true_SIDend, true_SIDint_ex, Ntruebins, true_weight)
_, Ntruebins_2D = multiD.get_SID2Dmap(Ntruebins)
Ntruebins_3D = Ntruebins_2D * Ntrueinttype # number of bins for the combined 3D variable (IDini, IDend, int_type)
true_SID3D, true_N3D, true_N3D_Vcov = slicing.get_3D_histogram(true_SIDini, true_SIDend, true_chnl, Ntruebins_2D, Ntruebins_3D, true_weight)

divided_recoEini, divided_weights = utils.divide_vars_by_partype(reco_initial_energy, particle_type_bool, mask=combined_true_mask, weight=reweight, alltypes=[0,1])
divided_recoEend, divided_weights = utils.divide_vars_by_partype(reco_end_energy, particle_type_bool, mask=combined_true_mask, weight=reweight, alltypes=[0,1])
divided_recochnl, divided_weights = utils.divide_vars_by_partype(reco_channel, particle_type_bool, mask=combined_true_mask, weight=reweight, alltypes=[0,1])
divided_recoisct, divided_weights = utils.divide_vars_by_partype(reco_containing, particle_type_bool, mask=combined_true_mask, weight=reweight, alltypes=[0,1])
divided_FullSelection, divided_weights = utils.divide_vars_by_partype(mask_FullSelection, particle_type_bool, mask=combined_true_mask, weight=reweight, alltypes=[0,1])
pass_selection = divided_FullSelection[1]
reco_Eini = divided_recoEini[1][pass_selection]
reco_Eend = divided_recoEend[1][pass_selection]
reco_chnl = divided_recochnl[1][pass_selection]
reco_isCt = divided_recoisct[1][pass_selection]
reco_weight = divided_weights[1][pass_selection]
#print(len(reco_Eini), reco_Eini, reco_Eend, reco_chnl, reco_weight, pass_selection, sep='\n')

meas_SIDini, meas_SIDend, meas_SIDint_ex = slicing.get_sliceID_histograms(reco_Eini, reco_Eend, reco_chnl==signal_int_type, reco_isCt, meas_bins)
_, Nmeasbins_2D = multiD.get_SID2Dmap(Nmeasbins)
Nmeasbins_3D = Nmeasbins_2D * Nrecointtype # number of bins for the combined 3D variable (IDini, IDend, int_type)
meas_SID3D, meas_N3D, meas_N3D_Vcov = slicing.get_3D_histogram(meas_SIDini, meas_SIDend, reco_chnl, Nmeasbins_2D, Nmeasbins_3D, reco_weight)

true_3D1D_map, true_N1D, true_N1D_err, Ntruebins_1D = multiD.map_index_to_combined_variable(true_N3D, np.sqrt(np.diag(true_N3D_Vcov)), Ntruebins_3D, enable=False)
meas_3D1D_map, meas_N1D, meas_N1D_err, Nmeasbins_1D = multiD.map_index_to_combined_variable(meas_N3D, np.sqrt(np.diag(meas_N3D_Vcov)), Nmeasbins_3D, enable=False)
#print(true_3D1D_map, true_N1D, true_N1D_err, Ntruebins_1D, sep='\n')
#print(meas_3D1D_map, meas_N1D, meas_N1D_err, Nmeasbins_1D, sep='\n')

eff1D, true_SID3D_sel = multiD.get_efficiency(true_N1D, true_SID3D, true_3D1D_map, Ntruebins_3D, pass_selection, reco_weight)
response_matrix, response = multiD.get_response_matrix(Nmeasbins_1D, Ntruebins_1D, meas_3D1D_map[meas_SID3D], true_3D1D_map[true_SID3D_sel], reco_weight)
#print(eff1D, response_matrix, sep='\n')

'''true_N3D_sel_noweight, _ = np.histogram(true_SID3D_sel, bins=np.arange(Ntruebins_3D+1))
eff1D_noweight = utils.safe_divide(true_N3D_sel_noweight[true_3D1D_map>0], true_N1D)
eff1D_upperr = []
eff1D_lowerr = []
for ii in range(len(eff1D)): # Note the error calculation by ClopperPearson ignores weights, so this is a rough estimates. In the end , we rely on toys to esimate the uncertainty on efficiency
    eff1D_upperr.append(ROOT.TEfficiency.ClopperPearson(true_N1D[ii], true_N3D_sel_noweight[true_3D1D_map>0][ii], 0.6826894921, True) - eff1D_noweight[ii])
    eff1D_lowerr.append(eff1D_noweight[ii] - ROOT.TEfficiency.ClopperPearson(true_N1D[ii], true_N3D_sel_noweight[true_3D1D_map>0][ii], 0.6826894921, False))
plt.figure(figsize=[8.6,4.8])
plt.errorbar(np.arange(0, Ntruebins_1D), eff1D, yerr=[eff1D_lowerr, eff1D_upperr], fmt="r.", ecolor="g", elinewidth=1, label="Efficiency")'''
plt.errorbar(np.arange(0, Ntruebins_1D), eff1D, yerr=0, fmt="r.", ecolor="g", elinewidth=1, label="Efficiency")
plt.xlabel(r"${\rm ID_{rem}}$")
plt.ylabel("Efficiency")
plt.title(r"Efficiency for true ${\rm ID_{rem}}$")
plt.xlim([-1, Ntruebins_1D])
plt.ylim([0, 1])
plt.legend()
# plt.savefig(f"plots/efficiency_plot_{beamPDG}.pdf")
plt.show()

plt.imshow(np.ma.masked_where(response_matrix == 0, response_matrix), origin="lower")
plt.title(r"Response matrix for ${\rm ID_{rem}}$")
plt.xlabel(r"Measured ${\rm ID_{rem}}$")
plt.ylabel(r"True ${\rm ID_{rem}}$")
plt.colorbar(label="Counts")
# plt.savefig(f"plots/response_matrix_{beamPDG}.pdf")
plt.show()

with open(respfilename, 'wb') as respfile: # save the response modeled by MC
    responseVars = {}
    responseVars["response_matrix"] = response.Hresponse()
    responseVars["response_truth"] = response.Htruth()
    responseVars["response_measured"] = response.Hmeasured()
    responseVars["eff1D"] = eff1D
    responseVars["meas_3D1D_map"] = meas_3D1D_map
    responseVars["true_3D1D_map"] = true_3D1D_map
    responseVars["true_N3D"] = true_N3D
    responseVars["true_N3D_Vcov"] = true_N3D_Vcov
    responseVars["meas_N3D"] = meas_N3D
    responseVars["meas_N3D_Vcov"] = meas_N3D_Vcov
    pickle.dump(responseVars, respfile)
