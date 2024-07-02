from packages import *

def cal_bkg_reweight(eventset):
    true_beam_PDG = eventset.true_beam_PDG
    reco_beam_true_byE_matched = eventset.reco_beam_true_byE_matched
    weight = np.ones_like(reco_beam_true_byE_matched)
    if not eventset.isMC:
        return weight
    
    mufrac = 1.71
    #weight[(true_beam_PDG == -13) & (reco_beam_true_byE_matched == 1)] *= mufrac
    weight = np.where((true_beam_PDG == -13) & (reco_beam_true_byE_matched == 1), weight * mufrac, weight)
    return weight

def cal_momentum_reweight(eventset, rdm_radius=0, rdm_angle=0):
    true_beam_PDG = eventset.true_beam_PDG
    true_beam_startP = eventset.true_beam_startP
    weight = np.ones_like(true_beam_startP)
    if not eventset.isMC:
        return weight
    
    if eventset.particle.pdg == 211:
        mom_mu0 = 1.0033
        mom_sigma0 = 0.0609
        mom_mu = 1.01818
        mom_sigma = 0.07192
        if rdm_radius != 0:
            oval_cx = 1.01818
            oval_cy = 0.07192
            oval_a = 0.006
            oval_b = 0.0032
            oval_phi = 0.776
            mom_mu = oval_cx + rdm_radius * (oval_a * math.cos(oval_phi) * math.cos(rdm_angle) - oval_b * math.sin(oval_phi) * math.sin(rdm_angle))
            mom_sigma = oval_cy + rdm_radius * (oval_a * math.sin(oval_phi) * math.cos(rdm_angle) + oval_b * math.cos(oval_phi) * math.sin(rdm_angle))
    elif eventset.particle.pdg == 2212:
        mom_mu0 = 0.9940
        mom_sigma0 = 0.0545
        mom_mu = 0.98939
        mom_sigma = 0.06394
        if rdm_radius != 0:
            oval_cx = 0.98939
            oval_cy = 0.06394
            oval_a = 0.002
            oval_b = 0.0012
            oval_phi = 0
            mom_mu = oval_cx + rdm_radius * (oval_a * math.cos(oval_phi) * math.cos(rdm_angle) - oval_b * math.sin(oval_phi) * math.sin(rdm_angle))
            mom_sigma = oval_cy + rdm_radius * (oval_a * math.sin(oval_phi) * math.cos(rdm_angle) + oval_b * math.cos(oval_phi) * math.sin(rdm_angle))
    else:
        raise Exception("No such particle implemented in cal_momentum_reweight() yet.")

    deno = np.exp(-np.power((true_beam_startP - mom_mu0) / mom_sigma0, 2) / 2)
    numo = np.exp(-np.power((true_beam_startP - mom_mu) / mom_sigma, 2) / 2)
    weight = np.where(true_beam_PDG == eventset.particle.pdg, weight * numo/deno, weight)
    wlimit = 3. # avoid large weight
    weight = np.clip(weight, 1/wlimit, wlimit)

    return weight

def cal_g4rw(eventset, weight):
    if eventset.particle.pdg == 211:
        return cal_g4rw_pionp(eventset, weight)
    elif eventset.particle.pdg == 2212:
        return cal_g4rw_proton(eventset, weight)
    else:
        raise Exception("No such particle implemented in cal_g4rw() yet.")

def cal_g4rw_pionp(eventset, weight):
    nreweibins = 20 # 20 reweightable bins for pi+ total inelastic
    if not hasattr(weight, '__len__'):
        weight = [weight]*nreweibins
    true_beam_PDG = eventset.true_beam_PDG
    g4rw_list = np.ones_like(true_beam_PDG)
    if np.all(np.array(weight) == 1):
        return g4rw_list
    for ievt in range(len(true_beam_PDG)):
        if true_beam_PDG[ievt] == 211:
            tot_inel = []
            g4rw = 1
            for i in range(nreweibins):
                g4rw_tot_inel = 0
                tot_inel.append(eventset.g4rw_full_grid_piplus_coeffs[ievt][i])
                if len(tot_inel[i]) > 0:
                    sum_tot = 0
                    for j in range(len(tot_inel[i])):
                        sum_tot += tot_inel[i][j]
                        g4rw_tot_inel += tot_inel[i][j] * math.pow(weight[i], j)

                    if abs(sum_tot - 1) > 0.1:
                        print(f"Check here $$$$$ {i}\t{sum_tot}")
                        #print(eventset.g4rw_full_grid_piplus_weights[i][9])
                        #print(f"{eventset.run}\t{eventset.subrun}\t{eventset.event}")
                        g4rw_tot_inel = 1
                    g4rw *= g4rw_tot_inel
            g4rw_list[ievt] = g4rw
    return g4rw_list

def cal_g4rw_proton(eventset, weight):
    nreweibins = 1 # 1 reweightable bins for proton total inelastic
    if not hasattr(weight, '__len__'):
        weight = [weight]*nreweibins
    true_beam_PDG = eventset.true_beam_PDG
    g4rw_list = np.ones_like(true_beam_PDG)
    if np.all(np.array(weight) == 1): 
        return g4rw_list
    for ievt in range(len(true_beam_PDG)):
        if true_beam_PDG[ievt] == 2212:
            tot_inel = []
            g4rw = 1
            for i in range(nreweibins):
                g4rw_tot_inel = 0
                tot_inel.append(eventset.g4rw_full_grid_proton_coeffs[ievt][i])
                if len(tot_inel[i]) > 0:
                    sum_tot = 0
                    for j in range(len(tot_inel[i])):
                        sum_tot += tot_inel[i][j]
                        g4rw_tot_inel += tot_inel[i][j] * math.pow(weight[i], j)

                    if abs(sum_tot - 1) > 0.1:
                        print(f"Check here $$$$$ {i}\t{sum_tot}")
                        #print(eventset.g4rw_full_grid_proton_weights[i][9])
                        #print(eventset.run, "\t", eventset.subrun, "\t", eventset.event)
                        g4rw_tot_inel = 1
                    g4rw *= g4rw_tot_inel
            g4rw_list[ievt] = g4rw
    return g4rw_list