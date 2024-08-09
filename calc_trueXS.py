from hadana.packages import *
import hadana.slicing_method as slicing
from hadana.BetheBloch import BetheBloch


beamPDG = 2212
#true_bins = np.array([1000,950,900,850,800,750,700,650,600,550,500,450,400,350,300,250,200,150,100,50,0])
true_bins = np.array([500,450,400,350,300,250,200,150,100,70,40,10,0])
with open('processed_files/procVars_pMC.pkl', 'rb') as procfile:
    processedVars = pickle.load(procfile)


mask_TrueSignal = processedVars["mask_TrueSignal"]
true_initial_energy = processedVars["true_initial_energy"]
true_end_energy = processedVars["true_end_energy"]
true_sigflag = processedVars["true_sigflag"]
true_containing = processedVars["true_containing"]
#particle_type = processedVars["particle_type"]
particle_type = np.zeros_like(true_sigflag) # use all MC (not just truth MC)
reweight = processedVars["reweight"]

divided_trueEini, divided_weights = utils.divide_vars_by_partype(true_initial_energy, particle_type, mask=mask_TrueSignal, weight=reweight)
divided_trueEend, divided_weights = utils.divide_vars_by_partype(true_end_energy, particle_type, mask=mask_TrueSignal, weight=reweight)
divided_trueflag, divided_weights = utils.divide_vars_by_partype(true_sigflag, particle_type, mask=mask_TrueSignal, weight=reweight)
divided_trueisct, divided_weights = utils.divide_vars_by_partype(true_containing, particle_type, mask=mask_TrueSignal, weight=reweight)
true_Eini = divided_trueEini[0]
true_Eend = divided_trueEend[0]
true_flag = divided_trueflag[0]
true_isCt = divided_trueisct[0]
true_weight = divided_weights[0]
print(len(true_Eini), true_Eini, true_Eend, true_flag, true_isCt, true_weight, sep='\n')

Ntruebins, Ntruebins_3D, true_cKE, true_wKE = utils.set_bins(true_bins)
true_SIDini, true_SIDend, true_SIDint_ex = slicing.get_sliceID_histograms(true_Eini, true_Eend, true_flag, true_isCt, true_bins)
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
simcurve = simcurvegraph.values()

plt.figure(figsize=[8,4.8])
XS_x = true_cKE[1:-1] # the underflow and overflow bin are not used
XS_y = true_XS[1:-1]
XS_xerr = true_wKE[1:-1]
XS_yerr = np.sqrt(np.diagonal(true_XS_Vcov))[1:-1] # get the uncertainty from the covariance matrix
plt.errorbar(XS_x, XS_y, XS_yerr, XS_xerr, fmt=".", label="Extracted true signal cross section")
#xx = np.linspace(0, 1100, 100)
#plt.plot(xx,XS_gen_ex(xx), label="Signal cross section used in simulation")
plt.plot(*simcurve, label="Signal cross section used in simulation")
plt.xlabel("Kinetic energy (MeV)")
plt.ylabel("Cross section (mb)") # 1 mb = 10^{-27} cm^2
plt.xlim([true_bins[-1], true_bins[0]])
plt.ylim(bottom=0)
plt.legend()
plt.show()

plt.pcolormesh(true_bins[1:-1], true_bins[1:-1], utils.transform_cov_to_corr_matrix(true_XS_Vcov[1:-1, 1:-1]), cmap="RdBu_r", vmin=-1, vmax=1)
plt.title(r"Correlation matrix for cross section")
plt.xticks(true_bins[1:-1])
plt.yticks(true_bins[1:-1])
plt.xlabel(r"Kinetic energy (MeV)")
plt.ylabel(r"Kinetic energy (MeV)")
plt.colorbar()
plt.show()