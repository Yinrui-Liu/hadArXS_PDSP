from .packages import *
import ROOT
from . import slicing_method as slicing
from . import get_histograms as get_hists
from . import parameters

def map_index_to_combined_variable(f_N3D, f_N3D_err, Nbins):
    Nbins_3D = Nbins**3
    ### mapping index from 3D to 1D
    f_3D1D_map = np.zeros(Nbins_3D, dtype=np.int32)
    tmpidx = 0
    for ibin in range(Nbins_3D):
        if f_N3D[ibin] > 0:
            tmpidx += 1
            f_3D1D_map[ibin] = tmpidx
    
    f_N1D = f_N3D[f_N3D>0]
    f_N1D_err = f_N3D_err[f_N3D>0]
    Nbins_1D = len(f_N1D)
    return f_3D1D_map, f_N1D, f_N1D_err, Nbins_1D

def get_efficiency(f_true_N3D, f_true_N1D, f_true_SID3D, Nbins_3D, pass_selection, weight):
    f_true_SID3D_sel = f_true_SID3D[pass_selection]
    f_true_N3D_sel, _ = np.histogram(f_true_SID3D_sel, bins=np.arange(Nbins_3D+1), weights=weight)

    f_eff1D = utils.safe_divide(f_true_N3D_sel[f_true_N3D>0], f_true_N1D)
    return f_eff1D, f_true_SID3D_sel

def get_response_matrix(f_Nmeasbins, f_Ntruebins, f_meas_hist, f_true_hist, weight):
    f_response = ROOT.RooUnfoldResponse(f_Nmeasbins, 1, f_Nmeasbins+1, f_Ntruebins, 1, f_Ntruebins+1) # 1D index starts from 1
    for ievt in range(len(f_meas_hist)):
        f_response.Fill(f_meas_hist[ievt], f_true_hist[ievt], weight[ievt])
    f_response_matrix = np.zeros([f_Ntruebins, f_Nmeasbins])
    for ibin in range(f_Ntruebins):
        for jbin in range(f_Nmeasbins):
            f_response_matrix[ibin, jbin] = f_response.Hresponse().GetBinContent(jbin+1, ibin+1)
    return f_response_matrix, f_response

def map_data_to_MC_bins(f_N3D, f_N3D_err, f_3D1D_map):
    Nbins_3D = len(f_3D1D_map)
    for ibin in range(Nbins_3D):
        if f_N3D[ibin] > 0 and f_3D1D_map[ibin] == 0:
            print("Not empty in data but empty in MC.")
    f_N1D = f_N3D[f_3D1D_map>0]
    f_N1D_err = f_N3D_err[f_3D1D_map>0]
    return f_N1D, f_N1D_err

def unfolding(f_data_meas_N1D, f_data_meas_V1D, f_response, niter=4, Nmeasbins=None, Ntruebins=None):
    if Nmeasbins is None:
        Nmeasbins = f_response.Hresponse().GetNbinsX()
    if Ntruebins is None:
        Ntruebins = f_response.Hresponse().GetNbinsY()
    hMeas = ROOT.TH1D ("hmeas", "", Nmeasbins, 0, Nmeasbins)
    for ibin in range(Nmeasbins):
        hMeas.SetBinContent(ibin+1, f_data_meas_N1D[ibin])
    uf = ROOT.RooUnfoldBayes(f_response, hMeas, niter=niter)
    data_meas_V1D_TM = ROOT.TMatrix(Nmeasbins, Nmeasbins)
    for ibin in range(Nmeasbins):
        for jbin in range(Nmeasbins):
            data_meas_V1D_TM[ibin, jbin] = f_data_meas_V1D[ibin, jbin]
    uf.SetMeasuredCov(data_meas_V1D_TM)
    
    hUnfold = uf.Hunfold()
    VUnfold = uf.Eunfold()
    
    f_data_unfold = np.zeros(Ntruebins)
    f_data_unfold_cov = np.zeros([Ntruebins, Ntruebins])
    for ibin in range(Ntruebins):
        f_data_unfold[ibin] = hUnfold.GetBinContent(ibin+1)
        for jbin in range(Ntruebins):
            f_data_unfold_cov[ibin, jbin] = VUnfold[ibin, jbin]
    return f_data_unfold, f_data_unfold_cov

def efficiency_correct_1Dvar(f_data_unfold, f_data_unfold_cov, f_eff1D, f_true_3D1D_map, f_Ntruebins_3D, f_true_N3D, f_true_N3D_Vcov, f_data_MC_scale, MCstat_fluc_to_nominal=None):
    if MCstat_fluc_to_nominal is None:
        MCstat_fluc_to_nominal = np.ones_like(f_eff1D)
    f_unfd_N3D = np.zeros(f_Ntruebins_3D)
    f_unfd_N3D_Vcov = np.zeros([f_Ntruebins_3D, f_Ntruebins_3D])
    for ibin in range(f_Ntruebins_3D):
        idx_1D_i = f_true_3D1D_map[ibin]
        if idx_1D_i > 0:
            if f_data_unfold[idx_1D_i-1] > 0:
                f_unfd_N3D[ibin] = f_data_unfold[idx_1D_i-1] / f_eff1D[idx_1D_i-1]
                for jbin in range(f_Ntruebins_3D):
                    idx_1D_j = f_true_3D1D_map[jbin]
                    if idx_1D_j > 0 and f_data_unfold[idx_1D_j-1] > 0:
                        f_unfd_N3D_Vcov[ibin, jbin] = f_data_unfold_cov[idx_1D_i-1, idx_1D_j-1] / (f_eff1D[idx_1D_i-1]*f_eff1D[idx_1D_j-1])
            elif f_eff1D[idx_1D_i-1] == 0:
                #print(data_unfold[true_3D1D_map[ibin]-1], true_N1D[true_3D1D_map[ibin]-1])
                f_unfd_N3D[ibin] = f_true_N3D[ibin] * f_data_MC_scale * MCstat_fluc_to_nominal[idx_1D_i-1]
                f_unfd_N3D_Vcov[ibin, ibin] = f_true_N3D_Vcov[ibin, ibin] * f_data_MC_scale*f_data_MC_scale * MCstat_fluc_to_nominal[idx_1D_i-1]*MCstat_fluc_to_nominal[idx_1D_i-1]
    #f_unfd_N3D_err = np.sqrt(np.diag(f_unfd_N3D_Vcov))
    return f_unfd_N3D, f_unfd_N3D_Vcov

def get_unfold_histograms(f_unfd_N3D, f_Ntruebins):
    f_unfd_N3D_real3D = np.reshape(f_unfd_N3D, [f_Ntruebins, f_Ntruebins, f_Ntruebins])
    f_unfd_Nini = f_unfd_N3D_real3D.sum((0,1))[1:]
    f_unfd_Nend = f_unfd_N3D_real3D.sum((0,2))[1:]
    f_unfd_Nint_ex = f_unfd_N3D_real3D.sum((1,2))[1:]
    f_unfd_Ninc = np.zeros_like(f_unfd_Nini)
    for ibin in range(f_Ntruebins-1):
        ## two equivalent way to calculate the incident histogram
        for itmp in range(0, ibin+1):
            f_unfd_Ninc[ibin] += f_unfd_Nini[itmp]
        for itmp in range(0, ibin):
            f_unfd_Ninc[ibin] -= f_unfd_Nend[itmp]
        '''for itmp in range(ibin, f_Ntruebins-1):
            f_unfd_Ninc[ibin] += f_unfd_Nend[itmp]
        for itmp in range(ibin+1, f_Ntruebins-1):
            f_unfd_Ninc[ibin] -= f_unfd_Nini[itmp]'''
    return f_unfd_Nini, f_unfd_Nend, f_unfd_Nint_ex, f_unfd_Ninc


if __name__ == "__main__":
    pass