from packages import *
import get_hists



if __name__ == "__main__":
    with open('processedVars.pkl', 'rb') as procfile:
        processedVars = pickle.load(procfile)

    mask_TrueSignal = processedVars["mask_TrueSignal"]
    true_initial_energy = processedVars["true_initial_energy"]
    true_end_energy = processedVars["true_end_energy"]
    true_sigflag = processedVars["true_sigflag"]
    particle_type = processedVars["particle_type"]
    reweight = processedVars["reweight"]

    divided_trueEini, divided_weights = get_hists.divide_vars_by_partype(true_initial_energy, particle_type, mask=mask_TrueSignal, weight=reweight)
    divided_trueEend, divided_weights = get_hists.divide_vars_by_partype(true_end_energy, particle_type, mask=mask_TrueSignal, weight=reweight)
    divided_trueflag, divided_weights = get_hists.divide_vars_by_partype(true_sigflag, particle_type, mask=mask_TrueSignal, weight=reweight)
    true_Eini = divided_trueEend[0]
    true_Eend = divided_trueEend[0]
    true_flag = divided_trueflag[0]
    true_weight = divided_weights[0]
    print(len(true_Eini), true_Eini, true_Eend, true_flag, true_weight, sep='\n')