from .packages import *
from .BetheBloch import BetheBloch
from . import parameters

class Processor:
    def __init__(self, ntuple, particle, isMC, **kwargs):
        self.ntuple = ntuple # ttree read from uproot
        self.variables_to_load = [] # list of strings
        self.particle = particle # Particle
        self.bb = BetheBloch(self.particle.pdg)
        self.isMC = isMC
        if self.isMC:
            self.fidvol_low = parameters.fidvol_low
        else:
            self.fidvol_low = parameters.fidvol_low + self.particle.parBQ["beam_startZ_data"] - self.particle.parBQ["beam_startZ_mc"]
        
        self.selection = kwargs.get('selection', [True]*10)
        self.fake_data = kwargs.get('fake_data', None) # for MC, fake_data is True for all fake data, False for all true MC, None for half-half
        self.incBQcut = kwargs.get('incBQcut', [True]*3) # include beam start XYZ cut, beam angle cut, beam scraper cut
        self.runPassStoppingProtonCut = kwargs.get('runPassStoppingProtonCut', False)
        self.extra_correct_KEi = kwargs.get('extra_correct_KEi', True)
        self.rng = np.random.RandomState(1)

        # output variables
        self.true_frontface_energy = []
        self.true_initial_energy = []
        self.true_end_energy = []
        self.true_sigflag = []
        self.true_containing = []
        self.true_track_length = []
        self.reco_frontface_energy = []
        self.reco_initial_energy = []
        self.reco_end_energy = []
        self.reco_sigflag = []
        self.reco_containing = []
        self.reco_track_length = []
        self.true_beam_PDG = np.array([])
        self.mask_TrueSignal = np.array([])
        self.mask_SelectedPart = np.array([])
        self.mask_FullSelection = np.array([])
        self.particle_type = []
        self.g4rw_full_grid_piplus_coeffs = np.array([])
        self.g4rw_full_grid_proton_coeffs = np.array([])
        self.true_beam_startP = np.array([])
        self.reco_beam_true_byE_matched = np.array([])
        self.beam_inst_KE = np.array([])
        self.Michel_score_bkgfit_mu = np.array([])
        self.proton_chi2_bkgfit_p = np.array([])
        self.costheta_bkgfit_spi = np.array([])
        self.chi2_stopping_proton = np.array([])
        self.trklen_csda_proton = np.array([])
        self.costheta_bkgfit_sp = np.array([])
        
    def LoadVariables(self, variable_list): # load more variables
        self.variables_to_load += variable_list

    def ProcessEvent(self, Nevents=None): # calculating all variables compulsory for the analysis
        eventset = self.ntuple.iterate(expressions=self.variables_to_load, entry_stop=Nevents, library="np") # entry_start=10000, step_size=1000

        Nevt_tot = 0
        Nevt_truesig = 0
        Nevt_isPar = 0
        Nevt_selected = 0

        for evt in eventset: # each step is a batch of events
            evtno = evt["event"]
            Nbatch = len(evtno)

            true_beam_traj_X = evt["true_beam_traj_X"]
            true_beam_traj_Y = evt["true_beam_traj_Y"]
            true_beam_traj_Z = evt["true_beam_traj_Z"]
            true_beam_traj_KE = evt["true_beam_traj_KE"]
            reco_beam_calo_X = evt["reco_beam_calo_X"]
            reco_beam_calo_Y = evt["reco_beam_calo_Y"]
            reco_beam_calo_Z = evt["reco_beam_calo_Z"]
            beam_inst_P = evt["beam_inst_P"]
            beam_inst_KE = np.sqrt( np.power(beam_inst_P*1000, 2) + self.particle.mass**2 ) - self.particle.mass
            upstream_energy_loss = GetUpstreamEnergyLoss(beam_inst_KE, self.particle.pdg)
            if self.isMC and self.extra_correct_KEi:
                if self.particle.pdg == 211:
                    beam_inst_KE += np.random.normal(-10.2, 0, Nbatch)
                elif self.particle.pdg == 2212:
                    beam_inst_KE += np.random.normal(2.9, 0, Nbatch)
            reco_frontfaceKE = beam_inst_KE - upstream_energy_loss
            reco_trklen_batch = np.zeros(Nbatch)

            reco_beam_true_byE_matched = evt["reco_beam_true_byE_matched"]
            reco_beam_true_byE_origin = evt["reco_beam_true_byE_origin"]
            reco_beam_true_byE_PDG = evt["reco_beam_true_byE_PDG"]
            true_beam_PDG = evt["true_beam_PDG"]
            true_beam_endProcess = evt["true_beam_endProcess"]

            g4rw_full_grid_piplus_coeffs = evt["g4rw_full_grid_piplus_coeffs"]
            g4rw_full_grid_proton_coeffs = evt["g4rw_full_grid_proton_coeffs"]
            true_beam_startP = evt["true_beam_startP"]
            
            true_endZ = []
            if self.fake_data is None:
                isFake = evtno%2
                #isFake = self.rng.uniform(size=len(evtno)) > 0.5
            elif self.fake_data is True:
                isFake = [True]*Nbatch
            elif self.fake_data is False:
                isFake = [False]*Nbatch
            for ievt in range(Nbatch):
                if self.isMC:
                    ## calculate true length and true energies
                    trueX = true_beam_traj_X[ievt]
                    trueY = true_beam_traj_Y[ievt]
                    trueZ = true_beam_traj_Z[ievt]
                    true_accum_len = np.zeros_like(trueZ)
                    Ntrue_traj_pts = len(trueZ)
                    true_endZ.append(trueZ[-1])

                    # calculate true length
                    true_trklen = 0.
                    for ii in range(1, Ntrue_traj_pts):
                        true_trklen += np.sqrt(
                            np.power(trueX[ii] - trueX[ii-1], 2) +
                            np.power(trueY[ii] - trueY[ii-1], 2) +
                            np.power(trueZ[ii] - trueZ[ii-1], 2)
                        )
                        true_accum_len[ii] = true_trklen
                    
                    # calculate true energies
                    trueKE = true_beam_traj_KE[ievt]
                    true_KEff = -999.
                    true_Eini = -999.
                    true_Eend = -999.
                    start_idx = -1
                    start_idx_ff = -1
                    for ii in range(Ntrue_traj_pts):
                        if trueZ[ii] > 0:
                            start_idx_ff = ii
                            break
                    for ii in range(Ntrue_traj_pts):
                        if trueZ[ii] > self.fidvol_low:
                            start_idx = ii
                            break

                    if start_idx >= 0:
                        traj_max = Ntrue_traj_pts - 1
                        temp = traj_max
                        while trueKE[temp] == 0:
                            temp -= 1
                        if start_idx == traj_max:
                            true_Eini = trueKE[temp]
                        else:
                            true_Eini = trueKE[start_idx]
                        if start_idx_ff == traj_max:
                            true_KEff = trueKE[temp]
                        else:
                            true_KEff = trueKE[start_idx_ff]
                    
                        if trueZ[-1] < parameters.fidvol_upp:
                            true_Eend = self.bb.KE_at_length(trueKE[temp], true_accum_len[traj_max] - true_accum_len[temp])
                        else: # non-containing tracks
                            idx = temp
                            while trueZ[idx] > parameters.fidvol_upp:
                                idx -= 1
                            true_Eend = self.bb.KE_at_length(trueKE[idx], (true_accum_len[idx+1]-true_accum_len[idx])*(parameters.fidvol_upp-trueZ[idx])/(trueZ[idx+1]-trueZ[idx]) )

                    self.true_frontface_energy.append(true_KEff)
                    self.true_initial_energy.append(true_Eini)
                    self.true_end_energy.append(true_Eend)
                    self.true_containing.append(trueZ[-1] < parameters.fidvol_upp)
                    self.true_track_length.append(true_trklen)

                ## calculate reco length and reco energies
                recoX = reco_beam_calo_X[ievt]
                recoY = reco_beam_calo_Y[ievt]
                recoZ = reco_beam_calo_Z[ievt]
                reco_accum_len = np.zeros_like(recoZ)
                Nreco_traj_pts = len(recoZ)

                # calculate reco length
                reco_trklen = 0.
                for ii in range(1, Nreco_traj_pts):
                    reco_trklen += np.sqrt(
                        np.power(recoX[ii] - recoX[ii-1], 2) +
                        np.power(recoY[ii] - recoY[ii-1], 2) +
                        np.power(recoZ[ii] - recoZ[ii-1], 2)
                    )
                    reco_accum_len[ii] = reco_trklen
                reco_trklen_batch[ievt] = reco_trklen

                # calculate reco energies
                reco_KEff = reco_frontfaceKE[ievt]
                reco_Eini = -999.
                reco_Eend = -999.
                reco_isContaining = True
                start_idx = -1
                for ii in range(Nreco_traj_pts):
                    if recoZ[ii] > self.fidvol_low:
                        start_idx = ii
                        break

                if start_idx >= 0:
                    if recoZ[start_idx] < parameters.fidvol_upp:
                        reco_Eini = self.bb.KE_at_length(reco_KEff, reco_accum_len[start_idx])
                        
                        if recoZ[-1] < parameters.fidvol_upp:
                            reco_Eend = self.bb.KE_at_length(reco_KEff, reco_trklen)
                        else: # non-containing tracks
                            idx = Nreco_traj_pts - 1
                            while recoZ[idx] > parameters.fidvol_upp:
                                idx -= 1
                            reco_Eend = self.bb.KE_at_length(reco_KEff, reco_accum_len[idx] + (reco_accum_len[idx+1]-reco_accum_len[idx])*(parameters.fidvol_upp-recoZ[idx])/(recoZ[idx+1]-recoZ[idx]) )
                            reco_isContaining = False
                
                self.reco_initial_energy.append(reco_Eini)
                self.reco_end_energy.append(reco_Eend)
                self.reco_containing.append(reco_isContaining)
                self.reco_track_length.append(reco_trklen)

                # get particle type
                par_type = GetParticleType(self.particle.pdg, self.isMC, isFake[ievt], reco_beam_true_byE_matched[ievt], reco_beam_true_byE_origin[ievt]==2, reco_beam_true_byE_PDG[ievt], true_beam_PDG[ievt], true_beam_endProcess[ievt])
                self.particle_type.append(par_type)

                inclusive = True
                if self.particle.pdg == 211:
                    if inclusive and true_beam_endProcess[ievt]=="pi+Inelastic":
                        true_flag = 1
                    else:
                        true_flag = 0
                    reco_flag = 1
                elif self.particle.pdg == 2212:
                    if inclusive and true_beam_endProcess[ievt]=="protonInelastic":
                        true_flag = 1
                    else:
                        true_flag = 0
                    reco_flag = 1
                else:
                    raise Exception(f"No mode implemented for pdg={self.particle.pdg}.")
                self.true_sigflag.append(true_flag)
                self.reco_sigflag.append(reco_flag)


            # selection
            if self.isMC:
                mask_TrueSignal = (true_beam_PDG==self.particle.pdg) & (np.array(true_endZ) > self.fidvol_low)
            else:
                mask_TrueSignal = np.zeros_like(true_beam_PDG, dtype=bool)
            mask_SelectedPart = self.particle.IsSelectedPart(evt)
            mask_FullSelection = self.particle.PassSelection(evt, self.isMC, self.selection, reco_trklen=reco_trklen_batch, xyz_cut=self.incBQcut[0], angle_cut=self.incBQcut[1], scraper_cut=self.incBQcut[2], runPassStoppingProtonCut=True)

            # bkg fit variables
            if self.particle.pdg == 211:
                self.Michel_score_bkgfit_mu = np.concatenate([self.Michel_score_bkgfit_mu, self.particle.var_dict["daughter_michel_score"]])
                self.proton_chi2_bkgfit_p = np.concatenate([self.proton_chi2_bkgfit_p, self.particle.var_dict["chi2_protons"]])
                self.costheta_bkgfit_spi = np.concatenate([self.costheta_bkgfit_spi, self.particle.var_dict["beam_costh"]])
            elif self.particle.pdg == 2212:
                self.chi2_stopping_proton = np.concatenate([self.chi2_stopping_proton, self.particle.var_dict["chi2_stopping_proton"]])
                self.trklen_csda_proton = np.concatenate([self.trklen_csda_proton, self.particle.var_dict["trklen_csda_proton"]])
                self.costheta_bkgfit_sp = np.concatenate([self.costheta_bkgfit_sp, self.particle.var_dict["beam_costh"]])

            Nevt_tot += Nbatch
            Nevt_truesig += len(mask_TrueSignal[mask_TrueSignal])
            Nevt_isPar += len(mask_SelectedPart[mask_SelectedPart])
            Nevt_selected += len(mask_FullSelection[mask_SelectedPart & mask_FullSelection])
            self.reco_frontface_energy = np.concatenate([self.reco_frontface_energy, reco_frontfaceKE])
            self.mask_TrueSignal = np.concatenate([self.mask_TrueSignal, mask_TrueSignal])
            self.mask_SelectedPart = np.concatenate([self.mask_SelectedPart, mask_SelectedPart])
            self.mask_FullSelection = np.concatenate([self.mask_FullSelection, mask_FullSelection])
            self.true_beam_PDG = np.concatenate([self.true_beam_PDG, true_beam_PDG])
            self.g4rw_full_grid_piplus_coeffs = np.concatenate([self.g4rw_full_grid_piplus_coeffs, g4rw_full_grid_piplus_coeffs])
            self.g4rw_full_grid_proton_coeffs = np.concatenate([self.g4rw_full_grid_proton_coeffs, g4rw_full_grid_proton_coeffs])
            self.true_beam_startP = np.concatenate([self.true_beam_startP, true_beam_startP])
            self.reco_beam_true_byE_matched = np.concatenate([self.reco_beam_true_byE_matched, reco_beam_true_byE_matched])
            self.beam_inst_KE = np.concatenate([self.beam_inst_KE, beam_inst_KE])
            
            print(f"{Nevt_tot} events processed.")
        
        print(Nevt_tot, Nevt_truesig, Nevt_isPar, Nevt_selected)
        self.true_frontface_energy = np.array(self.true_frontface_energy)
        self.true_initial_energy = np.array(self.true_initial_energy)
        self.true_end_energy = np.array(self.true_end_energy)
        self.true_sigflag = np.array(self.true_sigflag, dtype=bool)
        self.true_containing = np.array(self.true_containing, dtype=bool)
        self.true_track_length = np.array(self.true_track_length)
        self.reco_initial_energy = np.array(self.reco_initial_energy)
        self.reco_end_energy = np.array(self.reco_end_energy)
        self.reco_sigflag = np.array(self.reco_sigflag, dtype=bool)
        self.reco_containing = np.array(self.reco_containing, dtype=bool)
        self.reco_track_length = np.array(self.reco_track_length)
        self.mask_TrueSignal = np.array(self.mask_TrueSignal, dtype=bool)
        self.mask_SelectedPart = np.array(self.mask_SelectedPart, dtype=bool)
        self.mask_FullSelection = np.array(self.mask_FullSelection, dtype=bool)
        self.particle_type = np.array(self.particle_type)

    def GetOutVarsDict(self):
        outVars = {}
        outVars["isMC"] = self.isMC
        outVars["beamPDG"] = self.particle.pdg
        outVars["true_frontface_energy"] = self.true_frontface_energy
        outVars["true_initial_energy"] = self.true_initial_energy
        outVars["true_end_energy"] = self.true_end_energy
        outVars["true_sigflag"] = self.true_sigflag
        outVars["true_containing"] = self.true_containing
        outVars["true_track_length"] = self.true_track_length
        outVars["reco_frontface_energy"] = self.reco_frontface_energy
        outVars["reco_initial_energy"] = self.reco_initial_energy
        outVars["reco_end_energy"] = self.reco_end_energy
        outVars["reco_sigflag"] = self.reco_sigflag
        outVars["reco_containing"] = self.reco_containing
        outVars["reco_track_length"] = self.reco_track_length
        outVars["true_beam_PDG"] = self.true_beam_PDG
        outVars["mask_TrueSignal"] = self.mask_TrueSignal
        outVars["mask_SelectedPart"] = self.mask_SelectedPart
        outVars["mask_FullSelection"] = self.mask_FullSelection
        outVars["particle_type"] = self.particle_type
        outVars["g4rw_full_grid_piplus_coeffs"] = self.g4rw_full_grid_piplus_coeffs
        outVars["g4rw_full_grid_proton_coeffs"] = self.g4rw_full_grid_proton_coeffs
        outVars["true_beam_startP"] = self.true_beam_startP
        outVars["reco_beam_true_byE_matched"] = self.reco_beam_true_byE_matched
        outVars["beam_inst_KE"] = self.beam_inst_KE
        return outVars


def GetUpstreamEnergyLoss(beamKE, pdg, momentum=1): # 2nd polynominal parameters derived from model_upEloss.py
    upEloss = 0
    if pdg == 211 and momentum == 1:
        tmpKE = np.clip(beamKE, 700, 1050)
        upEloss = 467.1 - 1.171*tmpKE + 0.000729*np.power(tmpKE, 2)
    elif pdg == 2212 and momentum == 1:
        tmpKE = np.clip(beamKE, 300, 600)
        upEloss = 110.5 - 0.4571*tmpKE + 0.0006024*np.power(tmpKE, 2)
    else:
        raise Exception(f"No mode implemented for pdg={pdg} momentum={momentum}.")
    return upEloss

def GetParticleType(pdg_mode, isMC, isFake, beam_matched, isCosmic, true_particle_PDG, true_beam_PDG, true_beam_endProcess):
    if not isMC:
        return 0 # Data
    
    if pdg_mode == 211:
        if isFake:
            return 0 # Data (fake)
        elif not beam_matched:
            if isCosmic:
                return 4 # misID:cosmic
            elif abs(true_particle_PDG) == 211:
                return 6 # misID:pi
            elif true_particle_PDG == 2212:
                return 5 # misID:p
            elif abs(true_particle_PDG) == 13:
                return 7 # misID:mu
            elif abs(true_particle_PDG) == 11 or true_particle_PDG == 22:
                return 8 # misID:e/γ
            else:
                return 9 # misID:other
        elif true_beam_PDG == -13:
            return 3 # Muon
        elif true_beam_PDG == 211:
            if true_beam_endProcess == "pi+Inelastic":
                return 1 # PiInel (signal)
            else:
                return 2 # PiDecay
        return 9
            
    elif pdg_mode == 2212:
        if isFake:
            return 0 # Data (fake)
        elif not beam_matched:
            if isCosmic:
                return 3 # misID:cosmic
            elif abs(true_particle_PDG) == 211:
                return 5 # misID:pi
            elif true_particle_PDG == 2212:
                return 4 # misID:p
            elif abs(true_particle_PDG) == 13:
                return 6 # misID:mu
            elif abs(true_particle_PDG) == 11 or true_particle_PDG == 22:
                return 7 # misID:e/γ
            else:
                return 8 # misID:other
        elif true_beam_PDG == 2212:
            if true_beam_endProcess == "protonInelastic":
                return 1 # PInel (signal)
            else:
                return 2 # PElas
        return 8