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
        self.mask_SelectedPart = np.array([])
        self.mask_FullSelection = np.array([])

    def LoadVariables(self, variable_list): # load more variables
        self.variables_to_load += variable_list

    def ProcessEvent(self, Nevents=None): # calculating all variables compulsory for the analysis
        eventset = self.ntuple.iterate(expressions=self.variables_to_load, entry_stop=Nevents, library="np") # entry_start=10000, step_size=1000

        Nevt_tot = 0
        Nevt_isPar = 0
        Nevt_selected = 0

        for evt in eventset: # each step is a batch of events
            Nbatch = len(evt["MC"])
            
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

            # selection
            mask_SelectedPart = self.particle.IsSelectedPart(evt)
            mask_FullSelection = self.particle.PassSelection(evt, reco_trklen=reco_trklen_batch)

            Nevt_tot += Nbatch
            Nevt_isPar += len(mask_SelectedPart[mask_SelectedPart])
            Nevt_selected += len(mask_FullSelection[mask_SelectedPart & mask_FullSelection])
            self.mask_SelectedPart = np.concatenate([self.mask_SelectedPart, mask_SelectedPart])
            self.mask_FullSelection = np.concatenate([self.mask_FullSelection, mask_FullSelection])

            print(f"{Nevt_tot} events processed.")
        
        print(Nevt_tot, Nevt_isPar, Nevt_selected)
        self.true_initial_energy = np.array(self.true_initial_energy)
        self.true_end_energy = np.array(self.true_end_energy)
        self.true_track_length = np.array(self.true_track_length)
        self.reco_initial_energy = np.array(self.reco_initial_energy)
        self.reco_end_energy = np.array(self.reco_end_energy)
        self.reco_track_length = np.array(self.reco_track_length)


def GetUpstreamEnergyLoss(beamKE, pdg, momentum=1):
    upEloss = 0
    if pdg == 211 and momentum == 1:
        upEloss = 95.8 - 0.408*beamKE + 0.000347*np.power(beamKE, 2)
    elif pdg == 2212 and momentum == 1:
        upEloss = 26.9
    else:
        raise Exception(f"No mode implemented for pdg={pdg} momentum={momentum}.")
    return upEloss