from packages import *
from BetheBloch import BetheBloch
import selection_parameters

class Processor:
    def __init__(self, ntuple, particle, isMC):
        self.ntuple = ntuple # ttree read from uproot
        self.variables_to_load = [] # list of strings
        self.particle = particle # Particle
        self.bb = BetheBloch(self.particle.pdg)
        self.isMC = isMC
        if self.isMC:
            self.fidvol_low = selection_parameters.fidvol_low
        else:
            self.fidvol_low = selection_parameters.fidvol_low + self.particle.parBQ["beam_startZ_data"] - self.particle.parBQ["beam_startZ_mc"]

        # output variables
        self.true_initial_energy = []
        self.true_end_energy = []
        self.true_track_length = []
        self.reco_initial_energy = []
        self.reco_end_energy = []
        self.reco_track_length = []
        self.true_beam_PDG = np.array([])
        self.mask_SelectedPart = np.array([])
        self.mask_FullSelection = np.array([])
        self.particle_type = []
        self.g4rw_full_grid_piplus_coeffs = np.array([])
        self.g4rw_full_grid_proton_coeffs = np.array([])
        self.true_beam_startP = np.array([])
        self.reco_beam_true_byE_matched = np.array([])
        
    def LoadVariables(self, variable_list): # load more variables
        self.variables_to_load += variable_list

    def ProcessEvent(self, Nevents=None): # calculating all variables compulsory for the analysis
        eventset = self.ntuple.iterate(expressions=self.variables_to_load, entry_stop=Nevents, library="np") # entry_start=10000, step_size=1000

        Nevt_tot = 0
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
            
            for ievt in range(Nbatch):
                ## calculate true length and true energies
                trueX = true_beam_traj_X[ievt]
                trueY = true_beam_traj_Y[ievt]
                trueZ = true_beam_traj_Z[ievt]
                true_accum_len = np.zeros_like(trueZ)
                Ntrue_traj_pts = len(trueZ)
                
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
                true_Eini = -999.
                true_Eend = -999.
                start_idx = -1
                for ii in range(Ntrue_traj_pts):
                    if trueZ[ii] > self.fidvol_low:
                        start_idx = ii
                        break

                if start_idx >= 0:
                    traj_max = Ntrue_traj_pts - 1
                    temp = traj_max
                    while trueKE[temp] == 0:
                        temp -= 1
                    true_Eend = self.bb.KE_at_length(trueKE[temp], true_accum_len[traj_max] - true_accum_len[temp])

                    if start_idx == traj_max:
                        true_Eini = trueKE[temp]
                    else:
                        true_Eini = trueKE[start_idx]
                
                self.true_initial_energy.append(true_Eini)
                self.true_end_energy.append(true_Eend)
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
                start_idx = -1
                for ii in range(Nreco_traj_pts):
                    if recoZ[ii] > self.fidvol_low:
                        start_idx = ii
                        break

                if start_idx >= 0:
                    reco_Eini = self.bb.KE_at_length(reco_KEff, reco_accum_len[start_idx])
                    reco_Eend = self.bb.KE_at_length(reco_KEff, reco_trklen)
                
                self.reco_initial_energy.append(reco_Eini)
                self.reco_end_energy.append(reco_Eend)
                self.reco_track_length.append(reco_trklen)

                # get particle type
                par_type = GetParticleType(self.particle.pdg, self.isMC, evtno[ievt]%2, reco_beam_true_byE_matched[ievt], reco_beam_true_byE_origin[ievt]==2, reco_beam_true_byE_PDG[ievt], true_beam_PDG[ievt], true_beam_endProcess[ievt])
                self.particle_type.append(par_type)


            # selection
            mask_SelectedPart = self.particle.IsSelectedPart(evt)
            mask_FullSelection = self.particle.PassSelection(evt, reco_trklen=reco_trklen_batch)

            Nevt_tot += Nbatch
            Nevt_isPar += len(mask_SelectedPart[mask_SelectedPart])
            Nevt_selected += len(mask_FullSelection[mask_SelectedPart & mask_FullSelection])
            self.mask_SelectedPart = np.concatenate([self.mask_SelectedPart, mask_SelectedPart])
            self.mask_FullSelection = np.concatenate([self.mask_FullSelection, mask_FullSelection])
            self.true_beam_PDG = np.concatenate([self.true_beam_PDG, true_beam_PDG])
            self.g4rw_full_grid_piplus_coeffs = np.concatenate([self.g4rw_full_grid_piplus_coeffs, g4rw_full_grid_piplus_coeffs])
            self.g4rw_full_grid_proton_coeffs = np.concatenate([self.g4rw_full_grid_proton_coeffs, g4rw_full_grid_proton_coeffs])
            self.true_beam_startP = np.concatenate([self.true_beam_startP, true_beam_startP])
            self.reco_beam_true_byE_matched = np.concatenate([self.reco_beam_true_byE_matched, reco_beam_true_byE_matched])
            
            print(f"{Nevt_tot} events processed.")
        
        print(Nevt_tot, Nevt_isPar, Nevt_selected)
        self.true_initial_energy = np.array(self.true_initial_energy)
        self.true_end_energy = np.array(self.true_end_energy)
        self.true_track_length = np.array(self.true_track_length)
        self.reco_initial_energy = np.array(self.reco_initial_energy)
        self.reco_end_energy = np.array(self.reco_end_energy)
        self.reco_track_length = np.array(self.reco_track_length)
        self.particle_type = np.array(self.particle_type)


def GetUpstreamEnergyLoss(beamKE, pdg, momentum=1):
    upEloss = 0
    if pdg == 211 and momentum == 1:
        upEloss = 95.8 - 0.408*beamKE + 0.000347*np.power(beamKE, 2)
    elif pdg == 2212 and momentum == 1:
        upEloss = 26.9
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
            
