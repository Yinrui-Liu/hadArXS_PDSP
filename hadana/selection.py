from .packages import *
import ROOT
from . import parameters

class Particle:
    def __init__(self, pdg, mass):
        self.pdg = pdg
        self.mass = mass  # in MeV/c^2
        self.parBQ = {}  # parameter set for beam quality cut

        self.candidatePDGlist = []
        self.SetBQparameters()

    def SetCandidatePDG(self, pdg):
        self.candidatePDGlist.append(pdg)
    def SetCandidatePDGlist(self, pdglist):
        self.candidatePDGlist = pdglist
    
    def IsSelectedPart(self, evt):
        reco_reconstructable_beam_event = evt["reco_reconstructable_beam_event"]
        pass_BeamEntranceBox = reco_reconstructable_beam_event

        true_beam_PDG = evt["true_beam_PDG"]
        beam_inst_trigger = evt["beam_inst_trigger"]
        beam_inst_nMomenta = evt["beam_inst_nMomenta"]
        beam_inst_nTracks = evt["beam_inst_nTracks"]
        beam_inst_PDG_candidates = evt["beam_inst_PDG_candidates"]

        if evt["MC"].all():
            pass_BeamlinePID = np.isin(true_beam_PDG, self.candidatePDGlist)
        elif np.all(~evt["MC"]):
            pass_BeamlinePID = ( 
                (beam_inst_trigger!=8) # cosmogenic
                & ((beam_inst_nMomenta==1) & (beam_inst_nTracks==1))
                & ([np.any(np.isin(pdglist, self.candidatePDGlist)) for pdglist in beam_inst_PDG_candidates])
            )
        return pass_BeamEntranceBox & pass_BeamlinePID
    
    def SetBQparameters(self):
        if self.pdg == 211:
            self.parBQ = parameters.pionBQ
        elif self.pdg == 2212:
            self.parBQ = parameters.protonBQ
    
    def PassSelection(self, evt, isMC, selection=[True]*10, **kwargs):
        self.var_dict = {"daughter_michel_score":[], "chi2_protons":[], "beam_costh":[], "chi2_stopping_proton":[], "trklen_csda_proton":[]} # some temporary calculated variables can be passed to Processor for further use
        pass_selection = 1
        if self.pdg == 211:
            if selection[0] is True:
                pass_selection *= self.PassPandoraSliceCut(evt)
            if selection[1] is True:
                pass_selection *= self.PassCaloSizeCut(evt)
            if selection[2] is True:
                pass_selection *= self.PassBeamQualityCut(evt, self.parBQ, isMC, kwargs.get('xyz_cut', True), kwargs.get('angle_cut', True), kwargs.get('scraper_cut', True))
            if selection[3] is True:
                pass_selection *= self.PassFidVolCut(evt, self.parBQ, isMC)
            if selection[4] is True:
                pass_selection *= self.PassMichelScoreCut(evt)
            if selection[5] is True:
                pass_selection *= self.PassProtonCut(evt)
        elif self.pdg == 2212:
            if selection[0] is True:
                pass_selection *= self.PassPandoraSliceCut(evt)
            if selection[1] is True:
                pass_selection *= self.PassCaloSizeCut(evt)
            if selection[2] is True:
                pass_selection *= self.PassBeamQualityCut(evt, self.parBQ, isMC, kwargs.get('xyz_cut', True), kwargs.get('angle_cut', True), kwargs.get('scraper_cut', True))
            if selection[3] is True:
                pass_selection *= self.PassFidVolCut(evt, self.parBQ, isMC)
            if selection[4] is True:
                pass_selection *= self.PassStoppingProtonCut(evt, kwargs.get('reco_trklen'))
            elif kwargs.get('runPassStoppingProtonCut', False): # not adding the cut, but to calculate the variables chi2_stopping_proton and trklen_csda_proton
                self.PassStoppingProtonCut(evt, kwargs.get('reco_trklen'))
        pass_selection = np.array(pass_selection, dtype=bool)
        return pass_selection
        
    def PassPandoraSliceCut(self, evt): # track-like Pandora slice
        return evt["reco_beam_type"] == 13

    def PassCaloSizeCut(self, evt): # the collection plane hits not empty
        reco_beam_calo_wire = evt["reco_beam_calo_wire"]
        
        return np.array([len(calo_wire) != 0 for calo_wire in reco_beam_calo_wire])

    def PassBeamQualityCut(self, evt, parBQ, isMC, xyz_cut=True, angle_cut=True, scraper_cut=True):
        reco_beam_calo_startX = evt["reco_beam_calo_startX"]
        reco_beam_calo_startY = evt["reco_beam_calo_startY"]
        reco_beam_calo_startZ = evt["reco_beam_calo_startZ"]
        reco_beam_calo_endX = evt["reco_beam_calo_endX"]
        reco_beam_calo_endY = evt["reco_beam_calo_endY"]
        reco_beam_calo_endZ = evt["reco_beam_calo_endZ"]
        beam_inst_X = evt["beam_inst_X"]
        beam_inst_Y = evt["beam_inst_Y"]

        reco_beam_calo_wire = evt["reco_beam_calo_wire"]
        non_empty_mask = np.array([len(calo_wire) != 0 for calo_wire in reco_beam_calo_wire])

        pt0 = np.array([reco_beam_calo_startX, reco_beam_calo_startY, reco_beam_calo_startZ])
        pt1 = np.array([reco_beam_calo_endX, reco_beam_calo_endY, reco_beam_calo_endZ])
        dir = pt1 - pt0
        norms = np.linalg.norm(dir, axis=0)
        norms = np.where(norms == 0, 1, norms) # to prevent 0 due to reco_beam_calo_start==-999 and reco_beam_calo_end==-999
        dir /= norms
        dir = np.transpose(dir)

        beam_dx = np.full_like(reco_beam_calo_startX, -999, dtype=np.float64)
        beam_dy = np.full_like(reco_beam_calo_startX, -999, dtype=np.float64)
        beam_dz = np.full_like(reco_beam_calo_startX, -999, dtype=np.float64)
        if isMC:
            beam_dx = (reco_beam_calo_startX-parBQ["beam_startX_mc"])/parBQ["beam_startX_rms_mc"]
            beam_dy = (reco_beam_calo_startY-parBQ["beam_startY_mc"])/parBQ["beam_startY_rms_mc"]
            beam_dz = (reco_beam_calo_startZ-parBQ["beam_startZ_mc"])/parBQ["beam_startZ_rms_mc"]
        else:
            beam_dx = (reco_beam_calo_startX-parBQ["beam_startX_data"])/parBQ["beam_startX_rms_data"]
            beam_dy = (reco_beam_calo_startY-parBQ["beam_startY_data"])/parBQ["beam_startY_rms_data"]
            beam_dz = (reco_beam_calo_startZ-parBQ["beam_startZ_data"])/parBQ["beam_startZ_rms_data"]
        beam_dxy = np.sqrt(np.power(beam_dx, 2) + np.power(beam_dy, 2))

        beamdir_mc = np.array([np.cos(np.deg2rad(parBQ["beam_angleX_mc"])), np.cos(np.deg2rad(parBQ["beam_angleY_mc"])), np.cos(np.deg2rad(parBQ["beam_angleZ_mc"]))])
        beamdir_mc /= np.linalg.norm(beamdir_mc)
        beamdir_data = np.array([np.cos(np.deg2rad(parBQ["beam_angleX_data"])), np.cos(np.deg2rad(parBQ["beam_angleY_data"])), np.cos(np.deg2rad(parBQ["beam_angleZ_data"]))])
        beamdir_data /= np.linalg.norm(beamdir_data)
        beamdir = np.full_like(dir, -999, dtype=np.float64)
        if isMC:
            beamdir = [beamdir_mc.tolist()]*len(dir)
        else:
            beamdir = [beamdir_data.tolist()]*len(dir)
        beam_costh = np.einsum('ij,ij->i', dir, beamdir)
        self.var_dict["beam_costh"] = beam_costh

        inst_dxy = np.full_like(reco_beam_calo_startX, -999, dtype=np.float64)
        if isMC:
            inst_dxy = np.power( (beam_inst_X-parBQ["beam_startX_mc_inst"])/parBQ["beam_startX_rms_mc_inst"], 2) + np.power( (beam_inst_Y-parBQ["beam_startY_mc_inst"])/parBQ["beam_startY_rms_mc_inst"], 2)
        else:
            inst_dxy = np.power( (beam_inst_X-parBQ["beam_startX_data_inst"])/parBQ["beam_startX_rms_data_inst"], 2) + np.power( (beam_inst_Y-parBQ["beam_startY_data_inst"])/parBQ["beam_startY_rms_data_inst"], 2)

        passBQcut = non_empty_mask
        if xyz_cut:
            passBQcut &= PassBeamQualityCut_xyz(beam_dxy, beam_dz)
        if angle_cut:
            passBQcut &= PassBeamQualityCut_angle(beam_costh)
        if scraper_cut:
            passBQcut &= PassBeamQualityCut_inst(inst_dxy)
        return passBQcut

    def PassFidVolCut(self, evt, parBQ, isMC):
        reco_beam_calo_endZ = evt["reco_beam_calo_endZ"]

        pass_upper = reco_beam_calo_endZ < parameters.fidvol_upp

        pass_lower = np.zeros_like(reco_beam_calo_endZ, dtype=bool)
        if isMC:
            pass_lower = reco_beam_calo_endZ > parameters.fidvol_low
        else:
            pass_lower = reco_beam_calo_endZ > (parameters.fidvol_low + parBQ["beam_startZ_data"] - parBQ["beam_startZ_mc"])

        return pass_upper & pass_lower

    def PassMichelScoreCut(self, evt):
        reco_beam_vertex_nHits = evt["reco_beam_vertex_nHits"]
        reco_beam_vertex_michel_score_weight_by_charge = evt["reco_beam_vertex_michel_score_weight_by_charge"]

        mask = reco_beam_vertex_nHits != 0
        daughter_michel_score = np.where(mask, reco_beam_vertex_michel_score_weight_by_charge, -999)
        self.var_dict["daughter_michel_score"] = daughter_michel_score
        return daughter_michel_score < 0.55

    def PassProtonCut(self, evt):
        reco_beam_calo_wire = evt["reco_beam_calo_wire"]
        reco_beam_Chi2_proton = evt["reco_beam_Chi2_proton"]
        reco_beam_Chi2_ndof = evt["reco_beam_Chi2_ndof"]

        mask = np.array([len(calo_wire) != 0 for calo_wire in reco_beam_calo_wire])
        chi2_protons = np.where(mask, reco_beam_Chi2_proton / reco_beam_Chi2_ndof, -1)
        self.var_dict["chi2_protons"] = chi2_protons
        return chi2_protons > 80

    def PassStoppingProtonCut(self, evt, reco_trklen):
        beam_inst_P = evt["beam_inst_P"]
        reco_beam_calibrated_dEdX_SCE = evt["reco_beam_calibrated_dEdX_SCE"]
        reco_beam_resRange_SCE = evt["reco_beam_resRange_SCE"]

        csda_file = ROOT.TFile.Open("/Users/lyret/proton_mom_csda_converter.root") # uproot does not support Eval()
        csda_range_vs_mom_sm = csda_file.Get("csda_range_vs_mom_sm")
        csda = np.array([csda_range_vs_mom_sm.Eval(p) for p in beam_inst_P])
        trklen_csda_proton = reco_trklen / csda
        
        ppid_file = ROOT.TFile.Open("/Users/lyret/dEdxrestemplates.root")
        dedx_range_pro = ppid_file.Get("dedx_range_pro")
        chi2_stopping_proton = np.zeros_like(beam_inst_P)
        for ievt in range(len(chi2_stopping_proton)):
            trkdedx = reco_beam_calibrated_dEdX_SCE[ievt]
            trkres = reco_beam_resRange_SCE[ievt]
            chi2_stopping_proton[ievt] = utils.GetStoppingProtonChi2PID(trkdedx, trkres, dedx_range_pro)

        short_indices = trklen_csda_proton < 0.75
        long_indices = trklen_csda_proton > 0.75
        pass_cut = np.zeros_like(beam_inst_P, dtype=bool)
        pass_cut[short_indices] = chi2_stopping_proton[short_indices] > 7.5
        pass_cut[long_indices] = chi2_stopping_proton[long_indices] > 10
        self.var_dict["chi2_stopping_proton"] = chi2_stopping_proton
        self.var_dict["trklen_csda_proton"] = trklen_csda_proton
        return pass_cut

def PassBeamQualityCut_xyz(beam_dxy, beam_dz):
    return (beam_dxy > parameters.dxy_min) & (beam_dxy < parameters.dxy_max) & (beam_dz > parameters.dz_min) & (beam_dz < parameters.dz_max)

def PassBeamQualityCut_angle(beam_costh):
    return (beam_costh > parameters.costh_min) & (beam_costh < parameters.costh_max)
    
def PassBeamQualityCut_inst(inst_dxy): # beam scraper cut
    return inst_dxy < parameters.dxy_inst_sq_max