from hadana.packages import *
import hadana.slicing_method as slicing
import hadana.multiD_mapping as multiD
from hadana.BetheBloch import BetheBloch
import hadana.parameters as parameters
import hadana.MC_reweight as reweight


beamPDG = 211
datafilename = "/home/saikat/HadANA_Versions/hadArXS_PDSP/processed_files/procVars_piMC.pkl"
MCfilename = "/home/saikat/HadANA_Versions/hadArXS_PDSP/processed_files/procVars_piMC.pkl"
# types of systematic uncertainties to include
bkg_scale = [1, 1, 1, 1, 1, 1, 1] # should be imported from sideband fit  pionp [0.93, 1, 1.72, 1.43, 0.93, 1, 1]  proton [1, 1, 1, 1, 1, 1, 1]
bkg_scale_err = [0, 0, 0, 0, 0, 0, 0] # pionp [0.12, 0, 0.13, 0.11, 0.12, 0, 0]  proton [0, 0, 0, 0, 0, 0, 0]
inc_sys_bkg = False
inc_sys_MCstat = False
inc_sys_MCXS = False
inc_sys_reweiP = False
inc_sys_Eloss = False
inc_sys_AltSCE = False
niter = 47 # 47 for pion data, 16 for proton data
save_xs_for_sys = False
sysmode = 'Eloss'
use_external_cov = False
use_total_cov = False


if beamPDG == 211:
    true_bins = parameters.true_bins_pionp
    meas_bins = parameters.meas_bins_pionp
elif beamPDG == 2212:
    true_bins = parameters.true_bins_proton
    meas_bins = parameters.meas_bins_proton

### load data histograms
with open(datafilename, 'rb') as datafile: # either fake data or real data
    processedVars = pickle.load(datafile)

mask_SelectedPart = processedVars["mask_SelectedPart"]
mask_FullSelection = processedVars["mask_FullSelection"]
combined_mask = mask_SelectedPart & mask_FullSelection
reco_initial_energy = processedVars["reco_initial_energy"]
reco_end_energy = processedVars["reco_end_energy"]
reco_sigflag = processedVars["reco_sigflag"]
reco_containing = processedVars["reco_containing"]
particle_type = processedVars["particle_type"]
weight_dt = processedVars["reweight"]


### selection
print("### selection")
divided_recoEini, divided_weights = utils.divide_vars_by_partype(reco_initial_energy, particle_type, mask=combined_mask, weight=weight_dt)
divided_recoEend, divided_weights = utils.divide_vars_by_partype(reco_end_energy, particle_type, mask=combined_mask, weight=weight_dt)
divided_recoflag, divided_weights = utils.divide_vars_by_partype(reco_sigflag, particle_type, mask=combined_mask, weight=weight_dt)
divided_recoisct, divided_weights = utils.divide_vars_by_partype(reco_containing, particle_type, mask=combined_mask, weight=weight_dt)
data_reco_Eini = divided_recoEini[0]
data_reco_Eend = divided_recoEend[0]
data_reco_flag = divided_recoflag[0]
data_reco_isCt = divided_recoisct[0]
data_reco_weight = divided_weights[0]

Ndata = len(data_reco_Eini)
Ntruebins, Ntruebins_3D, true_cKE, true_wKE = utils.set_bins(true_bins)
Nmeasbins = len(meas_bins)
data_meas_SIDini, data_meas_SIDend, data_meas_SIDint_ex = slicing.get_sliceID_histograms(data_reco_Eini, data_reco_Eend, data_reco_flag, data_reco_isCt, meas_bins)
data_meas_SID3D, data_meas_N3D, data_meas_N3D_Vcov = slicing.get_3D_histogram(data_meas_SIDini, data_meas_SIDend, data_meas_SIDint_ex, Nmeasbins, data_reco_weight)
data_meas_N3D_err = np.sqrt(np.diag(data_meas_N3D_Vcov))


### background subtraction
print("### background subtraction")
with open(MCfilename, 'rb') as mcfile: # truth MC
    processedVars_mc = pickle.load(mcfile)

mask_SelectedPart_mc = processedVars_mc["mask_SelectedPart"]
mask_FullSelection_mc = processedVars_mc["mask_FullSelection"]
combined_mask_mc = mask_SelectedPart_mc & mask_FullSelection_mc
reco_initial_energy_mc = processedVars_mc["reco_initial_energy"]
reco_end_energy_mc = processedVars_mc["reco_end_energy"]
reco_sigflag_mc = processedVars_mc["reco_sigflag"]
reco_containing_mc = processedVars_mc["reco_containing"]
particle_type_mc = processedVars_mc["particle_type"]
weight_mc = processedVars_mc["reweight"]
if inc_sys_Eloss:
    if beamPDG == 211:
        rdm_Eloss_shift = np.random.normal(0, 2.0) # 2.0 MeV constant error for pion upstream E loss
    elif beamPDG == 2212:
        rdm_Eloss_shift = np.random.normal(0, 0.6) # 0.6 MeV constant error for proton upstream E loss
    reco_initial_energy_mc = np.where(reco_initial_energy_mc>0, reco_initial_energy_mc - rdm_Eloss_shift, reco_initial_energy_mc)
    reco_end_energy_mc = np.where(reco_end_energy_mc>0, reco_end_energy_mc - rdm_Eloss_shift, reco_end_energy_mc)
if inc_sys_reweiP:
    rdm_reweiP_radius = np.random.normal(0, 1)
    rdm_reweiP_angle = np.random.uniform(0, 2*np.pi)
    weight_mc = reweight.cal_bkg_reweight(processedVars_mc) * reweight.cal_momentum_reweight(processedVars_mc, rdm_reweiP_radius, rdm_reweiP_angle)
if inc_sys_MCXS:
    rdm_MCXS = np.random.normal(1, 0.15) # assign 15% uncertainty for MC XS model
    weight_mc *= reweight.cal_g4rw(processedVars_mc, rdm_MCXS)
divided_recoEini_mc, divided_weights_mc = utils.divide_vars_by_partype(reco_initial_energy_mc, particle_type_mc, mask=combined_mask_mc, weight=weight_mc)
divided_recoEend_mc, divided_weights_mc = utils.divide_vars_by_partype(reco_end_energy_mc, particle_type_mc, mask=combined_mask_mc, weight=weight_mc)
divided_recoflag_mc, divided_weights_mc = utils.divide_vars_by_partype(reco_sigflag_mc, particle_type_mc, mask=combined_mask_mc, weight=weight_mc)
divided_recoisct_mc, divided_weights_mc = utils.divide_vars_by_partype(reco_containing_mc, particle_type_mc, mask=combined_mask_mc, weight=weight_mc)
Ntruemc = len(reco_initial_energy_mc[combined_mask_mc]) - len(divided_recoEini_mc[0])
bkg_meas_N3D_list = []
bkg_meas_N3D_err_list = []
for ibkg in range(3, len(divided_recoEini_mc)):
    bkg_reco_Eini = divided_recoEini_mc[ibkg]
    bkg_reco_Eend = divided_recoEend_mc[ibkg]
    bkg_reco_flag = divided_recoflag_mc[ibkg]
    bkg_reco_isCt = divided_recoisct_mc[ibkg]
    bkg_reco_weight = divided_weights_mc[ibkg]
    bkg_meas_SIDini, bkg_meas_SIDend, bkg_meas_SIDint_ex = slicing.get_sliceID_histograms(bkg_reco_Eini, bkg_reco_Eend, bkg_reco_flag, bkg_reco_isCt, meas_bins)
    bkg_meas_SID3D, bkg_meas_N3D, bkg_meas_N3D_Vcov = slicing.get_3D_histogram(bkg_meas_SIDini, bkg_meas_SIDend, bkg_meas_SIDint_ex, Nmeasbins, bkg_reco_weight)
    bkg_meas_N3D_list.append(bkg_meas_N3D)
    bkg_meas_N3D_err_list.append(np.sqrt(np.diag(bkg_meas_N3D_Vcov)))

print(f"Bkg scale \t{bkg_scale}\nError \t\t{bkg_scale_err}")
sig_meas_N3D, sig_meas_N3D_err = utils.bkg_subtraction(data_meas_N3D, data_meas_N3D_err, bkg_meas_N3D_list, bkg_meas_N3D_err_list, mc2data_scale=Ndata/Ntruemc, bkg_scale=bkg_scale, bkg_scale_err=bkg_scale_err, include_bkg_err=inc_sys_bkg)



### unfolding
print("### model response")
mask_TrueSignal_mc = processedVars_mc["mask_TrueSignal"]
combined_true_mask_mc = mask_SelectedPart_mc & mask_TrueSignal_mc
beam_matched_mc = processedVars_mc["reco_beam_true_byE_matched"]
combined_reco_mask_mc = mask_FullSelection_mc & np.array(beam_matched_mc, dtype=bool)
true_initial_energy_mc = processedVars_mc["true_initial_energy"]
true_end_energy_mc = processedVars_mc["true_end_energy"]
true_sigflag_mc = processedVars_mc["true_sigflag"]
true_containing_mc = processedVars_mc["true_containing"]

particle_type_bool_mc = np.where(particle_type_mc==0, 0, 1) # 0 for fake data, 1 for truth MC
divided_trueEini_mc, divided_weights_mc = utils.divide_vars_by_partype(true_initial_energy_mc, particle_type_bool_mc, mask=combined_true_mask_mc, weight=weight_mc)
divided_trueEend_mc, divided_weights_mc = utils.divide_vars_by_partype(true_end_energy_mc, particle_type_bool_mc, mask=combined_true_mask_mc, weight=weight_mc)
divided_trueflag_mc, divided_weights_mc = utils.divide_vars_by_partype(true_sigflag_mc, particle_type_bool_mc, mask=combined_true_mask_mc, weight=weight_mc)
divided_trueisct_mc, divided_weights_mc = utils.divide_vars_by_partype(true_containing_mc, particle_type_bool_mc, mask=combined_true_mask_mc, weight=weight_mc)
mc_true_Eini = divided_trueEini_mc[1]
mc_true_Eend = divided_trueEend_mc[1]
mc_true_flag = divided_trueflag_mc[1]
mc_true_isCt = divided_trueisct_mc[1]
mc_true_weight = divided_weights_mc[1]

mc_true_SIDini, mc_true_SIDend, mc_true_SIDint_ex = slicing.get_sliceID_histograms(mc_true_Eini, mc_true_Eend, mc_true_flag, mc_true_isCt, true_bins)
mc_true_Nini, mc_true_Nend, mc_true_Nint_ex, mc_true_Ninc = slicing.derive_energy_histograms(mc_true_SIDini, mc_true_SIDend, mc_true_SIDint_ex, Ntruebins, mc_true_weight)
mc_true_SID3D, mc_true_N3D, mc_true_N3D_Vcov = slicing.get_3D_histogram(mc_true_SIDini, mc_true_SIDend, mc_true_SIDint_ex, Ntruebins, mc_true_weight)

divided_recoEini_mc, divided_weights_mc = utils.divide_vars_by_partype(reco_initial_energy_mc, particle_type_bool_mc, mask=combined_true_mask_mc, weight=weight_mc)
divided_recoEend_mc, divided_weights_mc = utils.divide_vars_by_partype(reco_end_energy_mc, particle_type_bool_mc, mask=combined_true_mask_mc, weight=weight_mc)
divided_recoflag_mc, divided_weights_mc = utils.divide_vars_by_partype(reco_sigflag_mc, particle_type_bool_mc, mask=combined_true_mask_mc, weight=weight_mc)
divided_recoisct_mc, divided_weights_mc = utils.divide_vars_by_partype(reco_containing_mc, particle_type_bool_mc, mask=combined_true_mask_mc, weight=weight_mc)
divided_FullSelection_mc, divided_weights_mc = utils.divide_vars_by_partype(mask_FullSelection_mc, particle_type_bool_mc, mask=combined_true_mask_mc, weight=weight_mc)
pass_selection_mc = divided_FullSelection_mc[1]
mc_reco_Eini = divided_recoEini_mc[1][pass_selection_mc]
mc_reco_Eend = divided_recoEend_mc[1][pass_selection_mc]
mc_reco_flag = divided_recoflag_mc[1][pass_selection_mc]
mc_reco_isCt = divided_recoisct_mc[1][pass_selection_mc]
mc_reco_weight = divided_weights_mc[1][pass_selection_mc]
#print(len(mc_reco_Eini), mc_reco_Eini, mc_reco_Eend, mc_reco_flag, mc_reco_weight, mc_pass_selection, sep='\n')

mc_meas_SIDini, mc_meas_SIDend, mc_meas_SIDint_ex = slicing.get_sliceID_histograms(mc_reco_Eini, mc_reco_Eend, mc_reco_flag, mc_reco_isCt, meas_bins)
mc_meas_SID3D, mc_meas_N3D, mc_meas_N3D_Vcov = slicing.get_3D_histogram(mc_meas_SIDini, mc_meas_SIDend, mc_meas_SIDint_ex, Nmeasbins, mc_reco_weight)

true_3D1D_map, mc_true_N1D, mc_true_N1D_err, Ntruebins_1D = multiD.map_index_to_combined_variable(mc_true_N3D, np.sqrt(np.diag(mc_true_N3D_Vcov)), Ntruebins)
meas_3D1D_map, mc_meas_N1D, mc_meas_N1D_err, Nmeasbins_1D = multiD.map_index_to_combined_variable(mc_meas_N3D, np.sqrt(np.diag(mc_meas_N3D_Vcov)), Nmeasbins)
#print(true_3D1D_map, mc_true_N1D, mc_true_N1D_err, Ntruebins_1D, sep='\n')
#print(meas_3D1D_map, mc_meas_N1D, mc_meas_N1D_err, Nmeasbins_1D, sep='\n')

if inc_sys_MCstat:
    _, mc_true_N3D_nominal, mc_true_N3D_Vcov_nominal = slicing.get_3D_histogram(
        mc_true_SIDini, mc_true_SIDend, mc_true_SIDint_ex, Ntruebins, np.ones_like(mc_true_weight))
    _, mc_true_N1D_nominal, _, _ = multiD.map_index_to_combined_variable(
        mc_true_N3D_nominal, np.sqrt(np.diag(mc_true_N3D_Vcov_nominal)), Ntruebins)

    print ("Here")
    
    mc_true_SID3D_sel = mc_true_SID3D[pass_selection_mc]
    response_matrix_nominal, _ = multiD.get_response_matrix(
        Nmeasbins_1D, Ntruebins_1D, meas_3D1D_map[mc_meas_SID3D],
        true_3D1D_map[mc_true_SID3D_sel], np.ones_like(mc_reco_weight))
    response_matrix_fluc = np.random.poisson(response_matrix_nominal)

    # Adding Tikhonov regularization and L-Curve generation
    lambda_values = np.logspace(-4, 2, 50)  # Regularization parameter range
    residual_norms = []
    regularization_norms = []

    for lambda_reg in lambda_values:
        # Tikhonov regularization
        L = np.identity(Ntruebins_1D - 1)  # Regularization matrix
        tikhonov_response_matrix = response_matrix_fluc + lambda_reg * L
        
        # Solve the regularized system
        try:
            solution = np.linalg.solve(tikhonov_response_matrix, mc_true_N1D_fluc[:Ntruebins_1D - 1])
        except np.linalg.LinAlgError:
            solution = np.zeros(Ntruebins_1D - 1)  # Handle singular matrices gracefully
        
        # Compute residual and regularization norms
        residual = np.dot(tikhonov_response_matrix, solution) - mc_true_N1D_fluc[:Ntruebins_1D - 1]
        residual_norm = np.linalg.norm(residual)
        regularization_norm = np.linalg.norm(np.dot(L, solution))
        
        # Store norms for the L-curve
        residual_norms.append(residual_norm)
        regularization_norms.append(regularization_norm)

    # Plot the L-curve
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.loglog(residual_norms, regularization_norms, marker='o', label="L-curve")
    plt.xlabel("Residual Norm ||Ax - b||")
    plt.ylabel("Regularization Norm ||Lx||")
    plt.title("L-Curve for Tikhonov Regularization")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()
