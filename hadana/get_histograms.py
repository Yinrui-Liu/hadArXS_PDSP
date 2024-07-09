from .packages import *

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
