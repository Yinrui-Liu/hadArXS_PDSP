from hadana.packages import *
import hadana.slicing_method as slicing
from hadana.BetheBloch import BetheBloch


beamPDG = 211
with open('processed_files/procVars_piMC_test.pkl', 'rb') as procfile:
    processedVars = pickle.load(procfile)
if beamPDG == 211:
    true_bins = np.array([1000,950,900,850,800,750,700,650,600,550,500,450,400,350,300,250,200,150,100,50,0])
elif beamPDG == 2212:
    true_bins = np.array([500,475,450,425,400,375,350,325,300,275,250,225,200,175,150,125,100,75,50,25,0])

"""processedVars["int_type"] = []

for daughters in processedVars["true_beam_daughter_PDG"]:
    n_pi_plus = 0
    n_pi_zero = 0
    n_pi_minus = 0
    for particle in daughters:
        if particle == -211:
            n_pi_minus += 1
        elif particle == 211:
            n_pi_plus += 1
        elif particle == 111:
            n_pi_zero += 1
    int_type = None
    if n_pi_plus == 1 and n_pi_zero == 0 and n_pi_minus == 0:
        int_type = "inel"
    elif n_pi_plus == 0 and n_pi_zero == 1 and n_pi_minus == 0:
        int_type = "cex"
    elif n_pi_plus == 0 and n_pi_zero == 0 and n_pi_minus == 1:
        int_type = "dcex"
    elif n_pi_plus == 0 and n_pi_zero == 0 and n_pi_minus == 0:
        int_type = "abs"
    elif (n_pi_plus + n_pi_zero +n_pi_minus > 1):
        int_type = "prod"
    processedVars["int_type"].append(int_type)
    

processedVars["selected_ex"] = [False] * len(processedVars["true_initial_energy"])
for i, ev_type in enumerate(processedVars["int_type"]):
    if ev_type == "inel": # Need to change this each time to select for different event types
        processedVars["selected_ex"][i] = True

selected_ex = processedVars["selected_ex"] # purposely redundant for now, but can just be incorporated into the above loop and skip the dict assign
"""
mask_TrueSignal = processedVars["mask_TrueSignal"] # & mask_exlusvie, exclusive cut goes here^, in addition to the mask we're using to select signal

"""
mask_combined = []
for i, val in enumerate(mask_TrueSignal):# Here is the combined exclusive and signal mask
    mask_combined.append(val and selected_ex[i])
"""
selected_type = "abs"
true_initial_energy = processedVars["true_initial_energy"]
true_end_energy = processedVars["true_end_energy"]
# selected_ex = [channel == selected_type for channel in processedVars["int_type"]]
# true_sigflag = [a and b for a, b in zip(processedVars["true_sigflag"], selected_ex)]
true_sigflag = processedVars["true_sigflag"]
# print(true_sigflag)
true_containing = processedVars["true_containing"]
#particle_type = processedVars["particle_type"]
particle_type = np.zeros_like(true_sigflag) # use all MC (not just truth MC)
reweight = processedVars["reweight"]
# print(processedVars["int_type"])
divided_trueEini, divided_weights = utils.divide_vars_by_partype(true_initial_energy, particle_type, mask=mask_TrueSignal, weight=reweight)
divided_trueEend, divided_weights = utils.divide_vars_by_partype(true_end_energy, particle_type, mask=mask_TrueSignal, weight=reweight)
divided_trueflag, divided_weights = utils.divide_vars_by_partype(true_sigflag, particle_type, mask=mask_TrueSignal, weight=reweight)
divided_trueisct, divided_weights = utils.divide_vars_by_partype(true_containing, particle_type, mask=mask_TrueSignal, weight=reweight)
true_Eini = divided_trueEini[0]
true_Eend = divided_trueEend[0]
true_flag = divided_trueflag[0]
true_isCt = divided_trueisct[0]
true_weight = divided_weights[0]
# print(len(true_Eini), true_Eini, true_Eend, true_flag, true_isCt, true_weight, sep='\n')

Ntruebins, Ntruebins_3D, true_cKE, true_wKE = utils.set_bins(true_bins)
true_SIDini, true_SIDend, true_SIDint_ex = slicing.get_sliceID_histograms(true_Eini, true_Eend, true_flag, true_isCt, true_bins)
#here goes the exclusive selections
true_Nini, true_Nend, true_Nint_ex, true_Ninc = slicing.derive_energy_histograms(true_SIDini, true_SIDend, true_SIDint_ex, Ntruebins, true_weight)
true_SID3D, true_N3D, true_N3D_Vcov = slicing.get_3D_histogram(true_SIDini, true_SIDend, true_SIDint_ex, Ntruebins, true_weight)
true_3SID_Vcov = slicing.get_Cov_3SID_from_N3D(true_N3D_Vcov, Ntruebins)
true_3N_Vcov = slicing.get_Cov_3N_from_3SID(true_3SID_Vcov, Ntruebins)
true_XS, true_XS_Vcov = slicing.calculate_XS_Cov_from_3N(true_Ninc, true_Nend, true_Nint_ex, true_3N_Vcov, true_bins, BetheBloch(beamPDG))

if beamPDG == 211:
    simcurvefile_name = "input_files/exclusive_xsec.root"
    simcurve_name = "total_inel_KE"
elif beamPDG == 2212:
    simcurvefile_name = "input_files/proton_cross_section.root"
    simcurve_name = "inel_KE"
simcurvefile = uproot.open(simcurvefile_name)
simcurvegraph = simcurvefile[simcurve_name]
incl_simcurve = simcurvegraph.values() # define the new splines under here
ex_file_name = "input_files/exclusive_xsec.root"
abs_simcurve = uproot.open(ex_file_name)["abs_KE"].values()
cex_simcurve = uproot.open(ex_file_name)["cex_KE"].values()
dcex_simcurve = uproot.open(ex_file_name)["dcex_KE"].values()
inel_simcurve = uproot.open(ex_file_name)["inel_KE"].values()
prod_simcurve = uproot.open(ex_file_name)["prod_KE"].values()

plt.figure(figsize=[8,4.8])
XS_x = true_cKE[1:-1] # the underflow and overflow bin are not used
XS_y = true_XS[1:-1]
XS_xerr = true_wKE[1:-1]
XS_yerr = np.sqrt(np.diagonal(true_XS_Vcov))[1:-1] # get the uncertainty from the covariance matrix
plt.errorbar(XS_x, XS_y, XS_yerr, XS_xerr, fmt=".", label="Extracted true signal cross section")
# print("XS:", XS_y.tolist())
# print("XSerr:", XS_yerr.tolist())
#xx = np.linspace(0, 1100, 100)
#plt.plot(xx,XS_gen_ex(xx), label="Signal cross section used in simulation")
plt.plot(*incl_simcurve, label="Signal cross section used in simulation")
plt.plot(*abs_simcurve, label="Absorption")
plt.plot(*cex_simcurve, label="Charge Exchange")
plt.plot(*dcex_simcurve, label="Double Charge Exchange")
plt.plot(*inel_simcurve, label="Inelastic")
plt.plot(*prod_simcurve, label="Pion Production")
sim_curve_dict = {"incl": incl_simcurve, "abs": abs_simcurve,"cex": cex_simcurve,
                  "dcex": dcex_simcurve, "inel": inel_simcurve, "prod": prod_simcurve}
XS_diff = XS_y - np.interp(XS_x, sim_curve_dict[selected_type][0], sim_curve_dict[selected_type][1]) # Need to fix this to not be hardcoded
inv_XS_Vcov = np.linalg.pinv(true_XS_Vcov[1:-1, 1:-1])
chi2 = np.einsum("i,ij,j->", XS_diff, inv_XS_Vcov, XS_diff)
print(f"Chi2/Ndf = {chi2}/{len(XS_diff)}")
title_dict = {"abs": "Pion Absorption", "cex": "Charge Exchange", "dcex": "Double Charge Exchange",
              "inel": "Pion Inelastic", "prod": "Pion Production"}
plt.title("Exclusive XSec, " + title_dict[selected_type])
plt.xlabel("Kinetic energy (MeV)")
plt.ylabel("Cross section (mb)") # 1 mb = 10^{-27} cm^2
plt.xlim([true_bins[-1], true_bins[0]])
plt.ylim(bottom=0)
plt.legend()
# plt.savefig(f"plots/XStrue_{beamPDG}.pdf") #Try not to overwrite yourself yeah?
plt.show()

plt.pcolormesh(true_bins[1:-1], true_bins[1:-1], utils.transform_cov_to_corr_matrix(true_XS_Vcov[1:-1, 1:-1]), cmap="RdBu_r", vmin=-1, vmax=1)
plt.title(r"Correlation matrix for cross section")
plt.xticks(true_bins[1:-1])
plt.yticks(true_bins[1:-1])
plt.xlabel(r"Kinetic energy (MeV)")
plt.ylabel(r"Kinetic energy (MeV)")
plt.colorbar()
# plt.savefig(f"plots/XStrueerr_{beamPDG}.pdf")
# plt.show()