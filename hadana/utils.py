import numpy as np
import iminuit
from .get_histograms import divide_vars_by_partype, get_vars_hists, bkg_subtraction

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