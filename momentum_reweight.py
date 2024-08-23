from hadana.packages import *
from hadana.processor import Processor
import hadana.selection as selection
import hadana.MC_reweight as reweight
from hadana.BetheBloch import BetheBloch
from scipy.stats import norm
from iminuit import cost


### load the corresponding ntuples
PDSP_ntuple_name_MC = "pduneana_MC_20g4rw"
PDSP_ntuple_name_data = "PDSPProd4_data_1GeV_reco2_ntuple_v09_41_00_04"
beamPDG = 2212
outfilename_MC = f"processed_files/procVars_momrew_MC_{beamPDG}.pkl"
outfilename_data = f"processed_files/procVars_momrew_data_{beamPDG}.pkl"
Nevents = None
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
if beamPDG == 211:
    particle = selection.Particle(beamPDG, 139.57)
    particle.SetCandidatePDGlist([-13, 13, 211])
    bb_mu = None
elif beamPDG == 2212:
    particle = selection.Particle(beamPDG, 938.272)
    particle.SetCandidatePDGlist(2212)
selection_momrew = [True,True,True,False,False,False] # full selection except for fiducial volume cut and Michel score cut

if os.path.exists(outfilename_MC):
    with open(outfilename_MC, 'rb') as procfile_MC:
        processedVars_MC = pickle.load(procfile_MC)
    print(f"Using existing file {outfilename_MC}")
else:
    PDSP_ntuple_MC = uproot.open(f"input_files/{PDSP_ntuple_name_MC}.root")
    pduneana_MC = PDSP_ntuple_MC["pduneana/beamana"]

    eventset_MC = Processor(pduneana_MC, particle, isMC=True, selection=selection_momrew, fake_data=False, runPassStoppingProtonCut=True, extra_correct_KEi=False)
    eventset_MC.fidvol_low = 0 # fiducial volume is not used
    eventset_MC.LoadVariables(variables_to_load)
    eventset_MC.ProcessEvent(Nevents=Nevents)
    processedVars_MC = eventset_MC.GetOutVarsDict()

    weights_MC = reweight.cal_bkg_reweight(processedVars_MC) * reweight.cal_momentum_reweight(processedVars_MC)
    processedVars_MC["reweight"] = weights_MC
    
    if beamPDG == 211:
        bb_mu = BetheBloch(13) # use muons for momentum reweighting
        processedVars_MC["Michel_score_bkgfit_mu"] = np.where(pduneana_MC["reco_beam_vertex_nHits"].array()[:Nevents]!=0, pduneana_MC["reco_beam_vertex_michel_score_weight_by_charge"].array()[:Nevents], -999)
        reco_KE_from_trklen_MC = bb_mu.KE_from_range_spline(processedVars_MC["reco_track_length"])
        reco_range_from_KE_MC = bb_mu.range_from_KE_spline(processedVars_MC["reco_initial_energy"])
        processedVars_MC["reco_KE_from_trklen"] = reco_KE_from_trklen_MC
        processedVars_MC["reco_range_from_KE"] = reco_range_from_KE_MC
    elif beamPDG == 2212:
        bb_p = eventset_MC.bb # use stopping protons for momentum reweighting
        processedVars_MC["chi2_stopping_proton"] = eventset_MC.chi2_stopping_proton
        processedVars_MC["trklen_csda_proton"] = eventset_MC.trklen_csda_proton
        reco_KE_from_trklen_MC = bb_p.KE_from_range_spline(processedVars_MC["reco_track_length"])
        reco_range_from_KE_MC = bb_p.range_from_KE_spline(processedVars_MC["reco_initial_energy"])
        processedVars_MC["reco_KE_from_trklen"] = reco_KE_from_trklen_MC
        processedVars_MC["reco_range_from_KE"] = reco_range_from_KE_MC

    with open(outfilename_MC, 'wb') as procfile_MC:
        pickle.dump(processedVars_MC, procfile_MC)

if os.path.exists(outfilename_data):
    with open(outfilename_data, 'rb') as procfile_data:
        processedVars_data = pickle.load(procfile_data)
    print(f"Using existing file {outfilename_MC}")
else:
    PDSP_ntuple_data = uproot.open(f"input_files/{PDSP_ntuple_name_data}.root")
    pduneana_data = PDSP_ntuple_data["pduneana/beamana"]

    eventset_data = Processor(pduneana_data, particle, isMC=False, selection=selection_momrew, runPassStoppingProtonCut=True)
    eventset_data.fidvol_low = 0 # fiducial volume is not used
    eventset_data.LoadVariables(variables_to_load)
    eventset_data.ProcessEvent(Nevents=Nevents)
    processedVars_data = eventset_data.GetOutVarsDict()

    weights_data = reweight.cal_bkg_reweight(processedVars_data) * reweight.cal_momentum_reweight(processedVars_data)
    processedVars_data["reweight"] = weights_data

    if beamPDG == 211:
        if bb_mu is None:
            bb_mu = BetheBloch(13)
        processedVars_data["Michel_score_bkgfit_mu"] = np.where(pduneana_data["reco_beam_vertex_nHits"].array()[:Nevents]!=0, pduneana_data["reco_beam_vertex_michel_score_weight_by_charge"].array()[:Nevents], -999)
        reco_KE_from_trklen_data = bb_mu.KE_from_range_spline(processedVars_data["reco_track_length"])
        reco_range_from_KE_data = bb_mu.range_from_KE_spline(processedVars_data["reco_initial_energy"])
        processedVars_data["reco_KE_from_trklen"] = reco_KE_from_trklen_data
        processedVars_data["reco_range_from_KE"] = reco_range_from_KE_data
    elif beamPDG == 2212:
        bb_p = eventset_data.bb
        processedVars_data["chi2_stopping_proton"] = eventset_data.chi2_stopping_proton
        processedVars_data["trklen_csda_proton"] = eventset_data.trklen_csda_proton
        reco_KE_from_trklen_data = bb_p.KE_from_range_spline(processedVars_data["reco_track_length"])
        reco_range_from_KE_data = bb_p.range_from_KE_spline(processedVars_data["reco_initial_energy"])
        processedVars_data["reco_KE_from_trklen"] = reco_KE_from_trklen_data
        processedVars_data["reco_range_from_KE"] = reco_range_from_KE_data

    with open(outfilename_data, 'wb') as procfile_data:
        pickle.dump(processedVars_data, procfile_data)


### selecting samples for momentum reweighting 
mask_SelectedPart_MC = processedVars_MC["mask_SelectedPart"]
mask_FullSelection_MC = processedVars_MC["mask_FullSelection"]
tratio_MC = processedVars_MC["reco_track_length"] / processedVars_MC["reco_range_from_KE"]
weights_MC = processedVars_MC["reweight"]
mask_SelectedPart_data = processedVars_data["mask_SelectedPart"]
mask_FullSelection_data = processedVars_data["mask_FullSelection"]
tratio_data = processedVars_data["reco_track_length"] / processedVars_data["reco_range_from_KE"]
weights_data = processedVars_data["reweight"]
if beamPDG == 211:
    mask_muon_MC = (processedVars_MC["Michel_score_bkgfit_mu"] > 0.6)[:Nevents]
    mask_MC = (mask_SelectedPart_MC & mask_FullSelection_MC & mask_muon_MC)[:Nevents]
    mask_muon_data = (processedVars_data["Michel_score_bkgfit_mu"] > 0.6)[:Nevents]
    mask_data = (mask_SelectedPart_data & mask_FullSelection_data & mask_muon_data)[:Nevents]
elif beamPDG == 2212:
    mask_stopproton_MC = ((processedVars_MC["chi2_stopping_proton"] < 5) & (processedVars_MC["trklen_csda_proton"] > 0.8))[:Nevents]
    mask_MC = (mask_SelectedPart_MC & mask_FullSelection_MC & mask_stopproton_MC)[:Nevents]
    mask_stopproton_data = ((processedVars_data["chi2_stopping_proton"] < 5) & (processedVars_data["trklen_csda_proton"] > 0.8))[:Nevents]
    mask_data = (mask_SelectedPart_data & mask_FullSelection_data & mask_stopproton_data)[:Nevents]

draw_tratio = True
if draw_tratio: # draw tratio distribution
    plt.hist(tratio_MC[:Nevents][mask_MC], bins=np.linspace(0, 1.2, 60), alpha=0.5, label="MC")
    plt.hist(tratio_data[:Nevents][mask_data], bins=np.linspace(0, 1.2, 60), alpha=0.5, label="Data")
    plt.xlim([0, 1.2])
    plt.ylim([0, None])
    plt.xlabel("reco_track_length / reco_range_from_KE")
    plt.legend()
    plt.show()
if beamPDG == 211:
    mask_MC = np.array(mask_MC, dtype=bool) & (tratio_MC > 0.9)[:Nevents] # For muons, we select tratio > 0.9 based on the tratio plots. For proton, it is not necessary, since there is not a second peak caused by the broken tracks at the gap of the two TPCs
    mask_data = np.array(mask_data, dtype=bool) & (tratio_data > 0.9)[:Nevents]
weights_MC = weights_MC[:Nevents][mask_MC] # in fact, weights is not used, which means we use the original MC to get the reweighting parameters
weights_data = weights_data[:Nevents][mask_data]
print(f"\nMC selected events: {len(weights_MC)}\tData selected events: {len(weights_data)}\n")
mcweight = np.ones_like(weights_MC)*len(weights_data)/len(weights_MC)
beam_inst_KE_MC = processedVars_MC["beam_inst_KE"][:Nevents][mask_MC]
beam_inst_P_MC = np.sqrt(beam_inst_KE_MC*beam_inst_KE_MC + 2*beam_inst_KE_MC*particle.mass) # MeV
true_beam_P_MC = processedVars_MC["true_beam_startP"][:Nevents][mask_MC]*1000 # MeV
reco_KE_from_trklen_MC = processedVars_MC["reco_KE_from_trklen"][:Nevents][mask_MC]
beam_inst_KE_data = processedVars_data["beam_inst_KE"][:Nevents][mask_data]
beam_inst_P_data = np.sqrt(beam_inst_KE_data*beam_inst_KE_data + 2*beam_inst_KE_data*particle.mass) # MeV
reco_KE_from_trklen_data = processedVars_data["reco_KE_from_trklen"][:Nevents][mask_data]


### Least Squares fits to get the mu and sigma to different momentum distributions
bins_p = np.linspace(700, 1300, 50)
nnmc, xemc, _ = plt.hist(beam_inst_P_MC, bins=bins_p, weights=mcweight, alpha=0.3, label="MC beam instrumented momentum")
nndt, xedt, _ = plt.hist(beam_inst_P_data, bins=bins_p, alpha=0.3, label="data beam instrumented momentum")
nnmc_t, xemc_t, _ = plt.hist(true_beam_P_MC, bins=bins_p, weights=mcweight, histtype="step", label="MC true beam momentum")

def gauss_pdf(x, mu, sigma):
    return norm.pdf(x, mu, sigma)
def gauss_cdf(xe, mu, sigma):
    return norm.cdf(xe, mu, sigma)
def gauss_extpdf(x, mu, sigma, n):
    #return n, n*norm.pdf(x, mu, sigma)
    return n*norm.pdf(x, mu, sigma)

c1 = cost.LeastSquares((xemc[1:]+xemc[:-1])/2, nnmc, np.maximum(np.sqrt(nnmc),1), gauss_extpdf)
c2 = cost.LeastSquares((xedt[1:]+xedt[:-1])/2, nndt, np.maximum(np.sqrt(nndt),1), gauss_extpdf)
c3 = cost.LeastSquares((xemc_t[1:]+xemc_t[:-1])/2, nnmc_t, np.maximum(np.sqrt(nnmc_t),1), gauss_extpdf)

m1 = iminuit.Minuit(c1, mu=1000, sigma=60, n=2000)
m1.migrad() # Gaussian fit to beam_inst_P_MC
mu0inst = m1.values["mu"]; mu0inst_err = m1.errors["mu"]
sigma0inst = m1.values["sigma"]; sigma0inst_err = m1.errors["sigma"]
print(f"beam_inst_P_MC fitted (mu, sigma, n) = ({mu0inst:.4f}±{mu0inst_err:.4f}, {sigma0inst:.4f}±{sigma0inst_err:.4f}, {m1.values['n']:.1f}±{m1.errors['n']:.1f})")

m2 = iminuit.Minuit(c2, mu=1000, sigma=60, n=2000)
m2.migrad() # Gaussian fit to beam_inst_P_data
muu = m2.values["mu"]; muu_err = m2.errors["mu"]
sigmaa = m2.values["sigma"]; sigmaa_err = m2.errors["sigma"]
print(f"beam_inst_P_data fitted (mu, sigma, n) = ({muu:.4f}±{muu_err:.4f}, {sigmaa:.4f}±{sigmaa_err:.4f}, {m2.values['n']:.1f}±{m2.errors['n']:.1f})")

m3 = iminuit.Minuit(c3, mu=1000, sigma=60, n=2000)
m3.migrad() # Gaussian fit to true_beam_P_MC
mu0 = m3.values["mu"]; mu0_err = m3.errors["mu"]
sigma0 = m3.values["sigma"]; sigma0_err = m3.errors["sigma"]
print(f"true_beam_P_MC fitted (mu, sigma, n) = ({mu0:.4f}±{mu0_err:.4f}, {sigma0:.4f}±{sigma0_err:.4f}, {m3.values['n']:.1f}±{m3.errors['n']:.1f})")
plt.xlabel("Momentum [MeV]")
plt.legend()
plt.show()


### 2D grid of the reweighting parameters (mu, sigma) using the reco_KE_from_trklen distributions
if beamPDG == 211:
    xbins = [0, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900, 920, 940, 960, 980, 1000, 1020, 1040, 1060, 1080, 1100, 1800]
elif beamPDG == 2212:
    xbins = [0, 250, 270, 290, 310, 330, 350, 370, 390, 410, 430, 450, 470, 490, 510, 530, 550, 800]
nbins = len(xbins)-1
#datahist, bin_edges, _ = plt.hist(reco_KE_from_trklen_data, bins=xbins, alpha=0.5, label="data"); plt.show()
datahist, bin_edges = np.histogram(reco_KE_from_trklen_data, bins=xbins)
def getChi2(mu, sigma, mu0=mu0, sigma0=sigma0, wlimit = 3):
    deno = np.exp(-(true_beam_P_MC - mu0)**2/2/sigma0**2)#/sigma0
    numo = np.exp(-(true_beam_P_MC - mu)**2/2/sigma**2)#/sigma
    weight = numo/deno
    weight = np.clip(weight, 1/wlimit, wlimit)
    normfact = len(numo)/sum(weight) * mcweight
    weight = weight * normfact
    MChist, _ = np.histogram(reco_KE_from_trklen_MC, bins=xbins, weights=weight)
    chi2 = (datahist - MChist)**2/np.maximum(datahist + MChist, 1)
    return np.sum(chi2[1:-1])/(nbins-3), weight

# fit (mu, sigma) to minimize chi2
def chi2_to_minimize(mu, sigma):
    chi2_value, _ = getChi2(mu, sigma)
    return chi2_value*(nbins-3) # the input function to Minuit is chi2 rather than chi2/ndof
m = iminuit.Minuit(chi2_to_minimize, mu=1000, sigma=60)
m.migrad()
mu_fit = m.values["mu"]
sigma_fit = m.values["sigma"]
mu_err = m.errors["mu"]
sigma_err = m.errors["sigma"]
print(f"\nFitted mu: {mu_fit:.4f} ± {mu_err:.4f} MeV")
print(f"Fitted sigma: {sigma_fit:.4f} ± {sigma_err:.4f} MeV")
# get the one-sigma uncertainty ellipse
cov_matrix = m.covariance
lambda1, lambda2 = np.linalg.eigvalsh(cov_matrix)
oval_a = np.sqrt(lambda1)
oval_b = np.sqrt(lambda2)
oval_phi = np.pi - 0.5 * np.arctan2(2 * cov_matrix[0,1], cov_matrix[0,0] - cov_matrix[1,1]) # rotation angle
print(f"The error ellipse: a = {oval_a}, b = {oval_b}, phi = {oval_phi}")

mu_list = np.linspace(960, 1040, 40) # (1000, 1030) for pion, (985, 1005) for proton, (960, 1040) by default
sigma_list = np.linspace(50, 90, 40) # (60, 90) for pion, (50, 70) for proton, (50, 90) by default
mm, ss = np.meshgrid(mu_list, sigma_list)
Chi2 = np.zeros_like(mm)
print(f"\nCalulating chi^2 for mu in [{mu_list[0]}, {mu_list[-1]}] and sigma in [{sigma_list[0]}, {sigma_list[-1]}]...")
for i in range(len(mm)):
    for j in range(len(mm[0])):
        Chi2[i,j], _ = getChi2(mm[i,j], ss[i,j])

plt.figure(figsize=[10,10])
plt.imshow(Chi2, extent = [min(mu_list),max(mu_list),min(sigma_list),max(sigma_list)],origin="lower",aspect=(max(mu_list)-min(mu_list))/(max(sigma_list)-min(sigma_list)))
cbar = plt.colorbar()
#cbar.set_label(r"Figure of merit (FOM): $\chi^2/N{\rm df}$")
plt.title(r"Figure of merit (FOM): $\chi^2/N{\rm df}$")
plt.xlabel(r"$\mu$ [MeV]")
plt.ylabel(r"$\sigma$ [MeV]")
plt.scatter([muu], [sigmaa], color="yellow", marker="o", label="Data beam inst P")
plt.scatter([mu0inst], [sigma0inst], color="blue", marker="o", label="MC beam inst P")
plt.scatter([mu0], [sigma0], color="lightgreen", marker="*", label="MC true beam P original")

oval_phi_list = np.linspace(0, 2*np.pi, 100)
oval_x = mu_fit + oval_a * np.cos(oval_phi_list) * np.cos(oval_phi) - oval_b * np.sin(oval_phi_list) * np.sin(oval_phi)
oval_y = sigma_fit + oval_a * np.cos(oval_phi_list) * np.sin(oval_phi) + oval_b * np.sin(oval_phi_list) * np.cos(oval_phi)
plt.plot(mu_fit, sigma_fit, 'r*', label=f"MC true beam P reweighted (FOM={getChi2(mu_fit, sigma_fit)[0]:.2f})")
plt.plot(oval_x, oval_y, 'r:', label=r'1-$\sigma$ uncertainty ellipse')
'''
# previous naive 2D grid estimate rather than 2D fit
minChi2 = np.min(Chi2)
minidx = np.where(Chi2 == minChi2)
onesigmaChi2 = minChi2 + 2.30/(nbins-3) # Table 40.2 https://pdg.lbl.gov/2020/reviews/rpp2020-rev-statistics.pdf
onesigmaidx = []
for i in range(len(mm)):
    for j in range(len(mm[0])-1):
        if Chi2[i,j]>onesigmaChi2 and Chi2[i,j+1]<onesigmaChi2:
            onesigmaidx.append([i,j])
        if Chi2[i,j]<onesigmaChi2 and Chi2[i,j+1]>onesigmaChi2:
            onesigmaidx.append([i,j+1])
for i in range(len(mm)-1):
    for j in range(len(mm[0])):
        if Chi2[i,j]>onesigmaChi2 and Chi2[i+1,j]<onesigmaChi2:
            onesigmaidx.append([i,j])
        if Chi2[i,j]<onesigmaChi2 and Chi2[i+1,j]>onesigmaChi2:
            onesigmaidx.append([i+1,j])
mur = mm[minidx][0]
sigmar = ss[minidx][0]
print(f"Obtained mu = {mur:.5f} (res {np.average(mu_list[1:]-mu_list[:-1]):.2g})")
print(f"Obtained sigma = {sigmar:.5f} (res {np.average(sigma_list[1:]-sigma_list[:-1]):.2g})")
plt.scatter(mur, sigmar, color="r", marker="*", label=f"MC true reweighted (FOM={minChi2:.2f})")
plt.scatter(mm[tuple(np.transpose(onesigmaidx))], ss[tuple(np.transpose(onesigmaidx))], color="orange", marker=".", s=5, label=f"One-sigma (FOM={onesigmaChi2:.2f})")
'''
plt.legend(fontsize=12, loc="upper left")
plt.xlim([mu_list[0], mu_list[-1]])
plt.ylim([sigma_list[0], sigma_list[-1]])
plt.show()


### plot the distributions after reweighting
print("\nShowing distributions after reweighting...")
chi2, neweight = getChi2(mu_fit, sigma_fit)
am,bm,_ = plt.hist(reco_KE_from_trklen_MC, density=True, bins=xbins, histtype="step", label="MC original", weights=mcweight)
aa,bb,_ = plt.hist(reco_KE_from_trklen_data, density=True, bins=xbins, alpha=0.2, label="data")
aw,bw,_ = plt.hist(reco_KE_from_trklen_MC, density=True, bins=xbins, histtype="step", label="MC reweighted", weights=neweight,color="r")
plt.legend()
plt.xlabel("Recontructed KE from track length [MeV]")
plt.show()

amp,bmp,_ = plt.hist(beam_inst_P_MC, bins=bins_p, histtype="step", label="MC original", weights=mcweight, density=True)
aap,bbp,_ = plt.hist(beam_inst_P_data, bins=bins_p, alpha=0.2, label="data", density=True)
awp,bwp,_ = plt.hist(beam_inst_P_MC, bins=bins_p, histtype="step", label="MC reweighted", weights=neweight, color="r", density=True)
plt.legend()
plt.title("beam_inst_P")
plt.xlabel("Beam instrumented momentum [MeV]")
plt.show()

plt.hist(true_beam_P_MC, bins=bins_p, histtype="step", label="MC true original", weights=mcweight, density=True)
plt.hist(true_beam_P_MC, bins=bins_p, histtype="step", label="MC true reweighted", weights=neweight, color="r", density=True)
plt.legend()
plt.title("true_beam_startP")
plt.xlabel("True beam momentum [MeV]")
plt.show()


### extra correction to the beam_inst_KE distribution after reweighting true_beam_P
if beamPDG == 211:
    mu_init = 900
    sigma_init = 60
    bins_ke = np.linspace(600, 1200, 50)
elif beamPDG == 2212:
    mu_init = 450
    sigma_init = 30
    bins_ke = np.linspace(300, 600, 50)
nnmc, xemc, _ = plt.hist(beam_inst_KE_MC, bins=bins_ke, weights=neweight, alpha=0.3, label="MC beam instrumented KE (after reweighting)")
nndt, xedt, _ = plt.hist(beam_inst_KE_data, bins=bins_ke, alpha=0.3, label="Data beam instrumented KE")
print(beam_inst_KE_MC, beam_inst_KE_data, sep='\n')
c1 = cost.LeastSquares((xemc[1:]+xemc[:-1])/2, nnmc, np.maximum(np.sqrt(nnmc),1), gauss_extpdf)
c2 = cost.LeastSquares((xedt[1:]+xedt[:-1])/2, nndt, np.maximum(np.sqrt(nndt),1), gauss_extpdf)

m1 = iminuit.Minuit(c1, mu=mu_init, sigma=sigma_init, n=20000)
m1.migrad() # Gaussian fit to beam_inst_KE_MC after reweighting
mu0inst = m1.values["mu"]; mu0inst_err = m1.errors["mu"]
sigma0inst = m1.values["sigma"]; sigma0inst_err = m1.errors["sigma"]
print(f"beam_inst_KE_MC after reweighting fitted (mu, sigma, n) = ({mu0inst:.4f}±{mu0inst_err:.4f}, {sigma0inst:.4f}±{sigma0inst_err:.4f}, {m1.values['n']:.1f}±{m1.errors['n']:.1f})")

m2 = iminuit.Minuit(c2, mu=mu_init, sigma=sigma_init, n=20000)
m2.migrad() # Gaussian fit to beam_inst_KE_data
muu = m2.values["mu"]; muu_err = m2.errors["mu"]
sigmaa = m2.values["sigma"]; sigmaa_err = m2.errors["sigma"]
print(f"beam_inst_KE_data fitted (mu, sigma, n) = ({muu:.4f}±{muu_err:.4f}, {sigmaa:.4f}±{sigmaa_err:.4f}, {m2.values['n']:.1f}±{m2.errors['n']:.1f})")

print(f"The extra shifting is {muu-mu0inst}, smearing is sqrt({sigmaa*sigmaa-sigma0inst*sigma0inst}). Update them in Processor.ProcessEvent().")

plt.xlabel("beam_inst_KE [MeV]")
plt.legend()
plt.show()
print("Done")