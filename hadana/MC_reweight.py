from .packages import *

def cal_bkg_reweight(procVars):
    true_beam_PDG = procVars["true_beam_PDG"]
    reco_beam_true_byE_matched = procVars["reco_beam_true_byE_matched"]
    weight = np.ones_like(reco_beam_true_byE_matched)
    if not procVars["isMC"]:
        return weight
    
    mufrac = 1.71
    #weight[(true_beam_PDG == -13) & (reco_beam_true_byE_matched == 1)] *= mufrac
    weight = np.where((true_beam_PDG == -13) & (reco_beam_true_byE_matched == 1), weight * mufrac, weight)
    return weight

def cal_momentum_reweight(procVars, rdm_radius=0, rdm_angle=0):
    true_beam_PDG = procVars["true_beam_PDG"]
    true_beam_startP = procVars["true_beam_startP"]
    weight = np.ones_like(true_beam_startP)
    if not procVars["isMC"]:
        return weight
    
    if procVars["beamPDG"] == 211:
        mom_mu0 = 2.0046
        mom_sigma0 = 0.10217
        mom_mu = 2.0198
        mom_sigma = 0.1004
        if rdm_radius != 0:
            oval_cx = 1.0198
            oval_cy = 0.0744
            oval_a = 0.0020
            oval_b = 0.0039
            oval_phi = 2.446
            mom_mu = oval_cx + rdm_radius * (oval_a * math.cos(oval_phi) * math.cos(rdm_angle) - oval_b * math.sin(oval_phi) * math.sin(rdm_angle))
            mom_sigma = oval_cy + rdm_radius * (oval_a * math.sin(oval_phi) * math.cos(rdm_angle) + oval_b * math.cos(oval_phi) * math.sin(rdm_angle))
    elif procVars["beamPDG"] == 2212:
        mom_mu0 = 0.9941
        mom_sigma0 = 0.0547
        mom_mu = 0.9898
        mom_sigma = 0.0644
        if rdm_radius != 0:
            oval_cx = 0.9898
            oval_cy = 0.0644
            oval_a = 0.0007
            oval_b = 0.0008
            oval_phi = 3.639
            mom_mu = oval_cx + rdm_radius * (oval_a * math.cos(oval_phi) * math.cos(rdm_angle) - oval_b * math.sin(oval_phi) * math.sin(rdm_angle))
            mom_sigma = oval_cy + rdm_radius * (oval_a * math.sin(oval_phi) * math.cos(rdm_angle) + oval_b * math.cos(oval_phi) * math.sin(rdm_angle))
    else:
        raise Exception("No such particle implemented in cal_momentum_reweight() yet.")

    deno = np.exp(-np.power((true_beam_startP - mom_mu0) / mom_sigma0, 2) / 2)
    numo = np.exp(-np.power((true_beam_startP - mom_mu) / mom_sigma, 2) / 2)
    weight = np.where(true_beam_PDG == procVars["beamPDG"], weight * numo/deno, weight)
    wlimit = 3. # avoid large weight
    weight = np.clip(weight, 1/wlimit, wlimit)

    return weight

def cal_g4rw(procVars, weight):
    if procVars["beamPDG"] == 211:
        return cal_g4rw_pionp(procVars, weight)
    elif procVars["beamPDG"] == 2212:
        return cal_g4rw_proton(procVars, weight)
    else:
        raise Exception("No such particle implemented in cal_g4rw() yet.")

def cal_g4rw_pionp(procVars, weight):
    nreweibins = 20 # 20 reweightable bins for pi+ total inelastic
    if not hasattr(weight, '__len__'):
        weight = [weight]*nreweibins
    true_beam_PDG = procVars["true_beam_PDG"]
    g4rw_list = np.ones_like(true_beam_PDG)
    if np.all(np.array(weight) == 1):
        return g4rw_list
    for ievt in range(len(true_beam_PDG)):
        if true_beam_PDG[ievt] == 211:
            tot_inel = []
            g4rw = 1
            for i in range(nreweibins):
                g4rw_tot_inel = 0
                tot_inel.append(procVars["g4rw_full_grid_piplus_coeffs"][ievt][i])
                if len(tot_inel[i]) > 0:
                    sum_tot = 0
                    for j in range(len(tot_inel[i])):
                        sum_tot += tot_inel[i][j]
                        g4rw_tot_inel += tot_inel[i][j] * math.pow(weight[i], j)

                    if abs(sum_tot - 1) > 0.1:
                        print(f"Check here $$$$$ {i}\t{sum_tot}")
                        #print(procVars["g4rw_full_grid_piplus_weights"][i][9])
                        #print(f"{procVars["run"]}\t{procVars["subrun"]}\t{procVars["event"]}")
                        g4rw_tot_inel = 1
                    g4rw *= g4rw_tot_inel
            g4rw_list[ievt] = g4rw
    return g4rw_list

def cal_g4rw_proton(procVars, weight):
    nreweibins = 1 # 1 reweightable bins for proton total inelastic
    if not hasattr(weight, '__len__'):
        weight = [weight]*nreweibins
    true_beam_PDG = procVars["true_beam_PDG"]
    g4rw_list = np.ones_like(true_beam_PDG)
    if np.all(np.array(weight) == 1): 
        return g4rw_list
    for ievt in range(len(true_beam_PDG)):
        if true_beam_PDG[ievt] == 2212:
            tot_inel = []
            g4rw = 1
            for i in range(nreweibins):
                g4rw_tot_inel = 0
                tot_inel.append(procVars["g4rw_full_grid_proton_coeffs"][ievt][i])
                if len(tot_inel[i]) > 0:
                    sum_tot = 0
                    for j in range(len(tot_inel[i])):
                        sum_tot += tot_inel[i][j]
                        g4rw_tot_inel += tot_inel[i][j] * math.pow(weight[i], j)

                    if abs(sum_tot - 1) > 0.1:
                        print(f"Check here $$$$$ {i}\t{sum_tot}")
                        #print(procVars["g4rw_full_grid_proton_weights"][i][9])
                        #print(procVars["run"], "\t", procVars["subrun"], "\t", procVars["event"])
                        g4rw_tot_inel = 1
                    g4rw *= g4rw_tot_inel
            g4rw_list[ievt] = g4rw
    return g4rw_list
