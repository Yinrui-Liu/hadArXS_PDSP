import numpy as np
import iminuit

def GetStoppingProtonChi2PID(trkdedx, trkres, dedx_range_pro):
    npt = 0
    chi2pro = 0

    for i in range(len(trkdedx)):
        # Ignore the first and the last point
        if i == 0 or i == len(trkdedx) - 1:
            continue
        if trkdedx[i] > 1000:  # Protect against large pulse height
            continue

        bin = dedx_range_pro.FindBin(trkres[i])
        
        if bin >= 1 and bin <= dedx_range_pro.GetNbinsX():
            bincpro = dedx_range_pro.GetBinContent(bin)
            if bincpro < 1e-6:  # For 0 bin content, use neighboring bins
                bincpro = (dedx_range_pro.GetBinContent(bin - 1) + dedx_range_pro.GetBinContent(bin + 1)) / 2
            
            binepro = dedx_range_pro.GetBinError(bin)
            if binepro < 1e-6:
                binepro = (dedx_range_pro.GetBinError(bin - 1) + dedx_range_pro.GetBinError(bin + 1)) / 2
            
            errdedx = 0.04231 + 0.0001783 * trkdedx[i] * trkdedx[i]  # Resolution on dE/dx
            errdedx *= trkdedx[i]
            chi2pro += ((trkdedx[i] - bincpro) / np.sqrt(binepro**2 + errdedx**2))**2
            npt += 1

    if npt > 0:
        return chi2pro / npt
    else:
        return 9999

def set_bins(KEbins):
    Nbins = len(KEbins)
    cKE = (KEbins[:-1] + KEbins[1:])/2 # energy bin centers
    wKE = (KEbins[:-1] - KEbins[1:])/2 # energy bin half-width
    Nbins_3D = Nbins**3 # number of bins for the combined variable
    return Nbins, Nbins_3D, cKE, wKE

def transform_cov_to_corr_matrix(cov): # a useful function to get correlation matrix from a covariance matrix
    corr = np.zeros_like(cov)
    nrows = len(cov)
    ncols = len(cov[0])
    if nrows != ncols:
        raise Exception("Input covariance matrix is not square!")
    for ir in range(nrows):
        for ic in range(ncols):
            if cov[ir, ic] != 0:
                corr[ir, ic] = cov[ir, ic]/np.sqrt(cov[ir, ir]*cov[ic, ic])
    return corr

def safe_divide(numerator, denominator):
    mask = (numerator == 0) & (denominator == 0)
    numerator = numerator.astype(np.float64)
    denominator = denominator.astype(np.float64)
    result = np.empty_like(denominator)
    result[~mask] = numerator[~mask] / denominator[~mask]
    result[mask] = 1
    return result

def gaussian(x, mu, sigma):
        return np.exp(-(x - mu)**2 / (2 * sigma**2)) / np.sqrt(2*np.pi) / sigma
def fit_gaus_hist(data, weights, x_range, initial_guesses):
    #if weights is None:
    #    weights = np.ones_like(data)
    mask = (data >= x_range[0]) & (data <= x_range[1])
    data = data[mask]
    weights = weights[mask]
    
    # Define the negative log-likelihood function
    def negative_log_likelihood(mu, sigma):
        # Ensure sigma is positive to avoid taking log of zero or negative values
        if sigma <= 0:
            return np.inf
        
        # Compute the Gaussian probability density function values
        pdf_values = gaussian(data, mu, sigma)
        
        # Negative log-likelihood
        nll = -np.sum(weights * np.log(pdf_values))
        return nll

    # Initialize Minuit
    m = iminuit.Minuit(negative_log_likelihood, mu=initial_guesses[0], sigma=initial_guesses[1])

    # Perform the minimization
    m.migrad()
    return m

def cal_chi2_2hists(arr_1, arr_2, weight_1, weight_2, bins, fit_range=None, scale21=None): # bins and fit_range should be increasing
    if fit_range is None:
        fit_range = [bins[0], bins[-1]]
    for ibin in range(len(bins)):
        if bins[ibin] >= fit_range[0]:
            fit_binidx_min = ibin
            break
    for ibin in range(len(bins)-1, -1, -1):
        if bins[ibin] <= fit_range[1]:
            fit_binidx_max = ibin
            break

    if scale21 is None:
        scale21 = sum(weight_1)/sum(weight_2)
    hist_1, hist_1_err, _ = get_vars_hists([arr_1], [weight_1], bins)
    hist_2, hist_2_err, _ = get_vars_hists([arr_2], [weight_2*scale21], bins)
    hist_1_err = np.clip(hist_1_err, 1, None) # avoid Poisson error to be 0
    chi2 = (hist_1[0] - hist_2[0])*(hist_1[0] - hist_2[0]) / (hist_1_err[0]*hist_1_err[0] + hist_2_err[0]*hist_2_err[0])
    nfitbins = fit_binidx_max - fit_binidx_min
    return np.sum(chi2[fit_binidx_min:fit_binidx_max]), nfitbins

### previously in get_histograms.py
def divide_vars_by_partype(vars, particle_type, mask=None, weight=None):
    ntypes = max(particle_type)+1
    divided_vars = []
    divided_weights = []
    if mask is None:
        mask = np.ones_like(vars, dtype=bool)
    for itype in range(ntypes):
        divided_vars.append( vars[mask & (particle_type==itype)] )
        divided_weights.append( weight[mask & (particle_type==itype)] )
        #hist, _ = np.histogram(vars[mask & (particle_type==itype)], binedges, weights=weight[mask & (particle_type==itype)])
        #hists.append(hist)
    return divided_vars, divided_weights

def get_vars_hists(var_list, weight_list, binedges=None, stack_hist_idx=[], xlabel=None): # prefer binedges to be an array; stack_hist_idx=np.arange(1,10)
    hists = []
    hists_err = []
    if binedges is None:
        binedges = np.linspace(min(min(var_list)), max(max(var_list)), 21)
    for itype in range(len(var_list)):
        if itype in stack_hist_idx: # save the stacked histogram plot for the variables
            hist, _, _ = plt.hist(var_list[itype], binedges, weights=weight_list[itype], label=itype, stacked=True)
        else:
            hist, _ = np.histogram(var_list[itype], binedges, weights=weight_list[itype])
        hists.append(hist)
        # Calculate the errors
        bin_indices = np.digitize(var_list[itype], binedges) - 1
        bin_errors = np.zeros(len(binedges) - 1)
        for bin_index in range(len(binedges) - 1):
            bin_weights = weight_list[itype][bin_indices == bin_index]
            bin_errors[bin_index] = np.sqrt(np.sum(bin_weights**2))
        hists_err.append(bin_errors)
    if len(stack_hist_idx):
        plt.xlabel(xlabel)
        plt.ylabel("Weighted counts")
        plt.legend()
        plt.savefig("test_h.png")
        plt.clf()
    return hists, hists_err, binedges

def bkg_subtraction(data_hist, data_hist_err, bkg_hists, bkg_hists_err, mc2data_scale=1, bkg_scale=None, bkg_scale_err=None, include_bkg_err=True): # expect bkg_scale as an array with the same length as bkg_hists (nbkgs)
    sig_hist = np.array(data_hist)
    sig_hist_err_sq = np.power(data_hist_err, 2)
    nbins = len(data_hist)
    nbkgs = len(bkg_hists)
    if bkg_scale is None:
        bkg_scale = np.ones(nbkgs)
    if (not include_bkg_err) or (bkg_scale_err is None):
        bkg_scale_err = np.zeros(nbkgs)
    for ib in range(nbkgs):
        sig_hist -= (bkg_hists[ib]*mc2data_scale * bkg_scale[ib])
    if include_bkg_err:
        for ib in range(nbkgs):
            sig_hist_err_sq += (np.power(bkg_hists_err[ib]*mc2data_scale * bkg_scale[ib], 2) + np.power(bkg_hists[ib]*mc2data_scale * bkg_scale_err[ib], 2))
    sig_hist_err = np.sqrt(sig_hist_err_sq)
    return sig_hist, sig_hist_err

# def bkg_sideband_fit():

if __name__ == "__main__":
    with open('processedVars.pkl', 'rb') as procfile:
        processedVars = pickle.load(procfile)

    true_initial_energy = processedVars["true_initial_energy"]
    true_end_energy = processedVars["true_end_energy"]
    reco_initial_energy = processedVars["reco_initial_energy"]
    reco_end_energy = processedVars["reco_end_energy"]
    mask_SelectedPart = processedVars["mask_SelectedPart"]
    mask_FullSelection = processedVars["mask_FullSelection"]
    combined_mask = mask_SelectedPart & mask_FullSelection
    particle_type = processedVars["particle_type"]
    reweight = processedVars["reweight"]

    divided_vars, divided_weights = divide_vars_by_partype(reco_end_energy, particle_type, mask=(mask_SelectedPart&mask_FullSelection), weight=reweight)
    hists, hists_err, binedges = get_vars_hists(divided_vars, divided_weights, binedges=np.linspace(0, 1200, 25), stack_hist_idx=np.arange(1,10), xlabel="reco_end_energy [MeV]")
    #print(hists)
    sig_hist, sig_hist_err = bkg_subtraction(hists[0], hists_err[0], hists[3:-1], hists_err[3:-1], mc2data_scale=1, bkg_scale=None, bkg_scale_err=None)
    print(sig_hist, sig_hist_err)
    plt.bar((binedges[:-1]+binedges[1:])/2, sig_hist, width=binedges[1:]-binedges[:-1], color='lightblue', edgecolor='black', alpha=0.7)
    plt.errorbar((binedges[:-1]+binedges[1:])/2, sig_hist, yerr=sig_hist_err, fmt='o', color='red', markersize=1)
    plt.show()
