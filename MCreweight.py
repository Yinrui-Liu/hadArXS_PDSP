from packages import *

def cal_g4rw(evt, weight):
    if evt.particle.pdg == 211:
        return cal_g4rw_pionp(evt, weight)
    elif evt.particle.pdg == 2212:
        return cal_g4rw_proton(evt, weight)
    else:
        raise Exception("No such particle implemented in cal_g4rw() yet.")

def cal_g4rw_pionp(evt, weight):
    nreweibins = 20 # 20 reweightable bins for pi+ total inelastic
    if not hasattr(weight, '__len__'):
        weight = [weight]*nreweibins
    true_beam_PDG = evt.true_beam_PDG
    g4rw_list = np.ones_like(true_beam_PDG)
    if np.all(np.array(weight) == 1):
        return g4rw_list
    for ievt in range(len(true_beam_PDG)):
        if true_beam_PDG[ievt] == 211:
            tot_inel = []
            g4rw = 1
            for i in range(nreweibins):
                g4rw_tot_inel = 0
                tot_inel.append(evt.g4rw_full_grid_piplus_coeffs[ievt][i])
                if len(tot_inel[i]) > 0:
                    sum_tot = 0
                    for j in range(len(tot_inel[i])):
                        sum_tot += tot_inel[i][j]
                        g4rw_tot_inel += tot_inel[i][j] * math.pow(weight[i], j)

                    if abs(sum_tot - 1) > 0.1:
                        print(f"Check here $$$$$ {i}\t{sum_tot}")
                        #print(evt.g4rw_full_grid_piplus_weights[i][9])
                        #print(f"{evt.run}\t{evt.subrun}\t{evt.event}")
                        g4rw_tot_inel = 1
                    g4rw *= g4rw_tot_inel
            g4rw_list[ievt] = g4rw
    return g4rw_list

def cal_g4rw_proton(evt, weight):
    nreweibins = 1 # 1 reweightable bins for proton total inelastic
    if not hasattr(weight, '__len__'):
        weight = [weight]*nreweibins
    true_beam_PDG = evt.true_beam_PDG
    g4rw_list = np.ones_like(true_beam_PDG)
    if np.all(np.array(weight) == 1): 
        return g4rw_list
    for ievt in range(len(true_beam_PDG)):
        if true_beam_PDG[ievt] == 2212:
            tot_inel = []
            g4rw = 1
            for i in range(nreweibins):
                g4rw_tot_inel = 0
                tot_inel.append(evt.g4rw_full_grid_proton_coeffs[ievt][i])
                if len(tot_inel[i]) > 0:
                    sum_tot = 0
                    for j in range(len(tot_inel[i])):
                        sum_tot += tot_inel[i][j]
                        g4rw_tot_inel += tot_inel[i][j] * math.pow(weight[i], j)

                    if abs(sum_tot - 1) > 0.1:
                        print(f"Check here $$$$$ {i}\t{sum_tot}")
                        #print(evt.g4rw_full_grid_proton_weights[i][9])
                        #print(evt.run, "\t", evt.subrun, "\t", evt.event)
                        g4rw_tot_inel = 1
                    g4rw *= g4rw_tot_inel
            g4rw_list[ievt] = g4rw
    return g4rw_list