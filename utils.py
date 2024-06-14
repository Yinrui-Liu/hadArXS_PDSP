from packages import *

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
