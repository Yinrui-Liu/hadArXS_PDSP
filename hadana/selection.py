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
    
    def PassSelection(self, evt, **kwargs):
        if self.pdg == 211:
            return PassPandoraSliceCut(evt) & PassCaloSizeCut(evt) & PassBeamQualityCut(evt, self.parBQ) & PassFidVolCut(evt, self.parBQ) & PassMichelScoreCut(evt) & PassProtonCut(evt)
        elif self.pdg == 2212:
            return PassPandoraSliceCut(evt) & PassCaloSizeCut(evt) & PassBeamQualityCut(evt, self.parBQ) & PassFidVolCut(evt, self.parBQ) & PassStoppingProtonCut(evt, **kwargs)
        

def PassPandoraSliceCut(evt): # track-like Pandora slice
    return evt["reco_beam_type"] == 13

def PassCaloSizeCut(evt): # the collection plane hits not empty
    reco_beam_calo_wire = evt["reco_beam_calo_wire"]
    
    return np.array([len(calo_wire) != 0 for calo_wire in reco_beam_calo_wire])

def PassBeamQualityCut(evt, parBQ):
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

    isMC = evt["MC"]
    mc_indices = (isMC == True)
    dt_indices = (isMC == False)

    pt0 = np.array([reco_beam_calo_startX, reco_beam_calo_startY, reco_beam_calo_startZ])
    pt1 = np.array([reco_beam_calo_endX, reco_beam_calo_endY, reco_beam_calo_endZ])
    dir = pt1 - pt0
    norms = np.linalg.norm(dir, axis=0)
    norms = np.where(norms == 0, 1, norms) # to prevent 0 due to reco_beam_calo_start==-999 and reco_beam_calo_end==-999
    dir /= norms
    dir = np.transpose(dir)

    beam_dx = np.full_like(isMC, -999, dtype=np.float64)
    beam_dy = np.full_like(isMC, -999, dtype=np.float64)
    beam_dz = np.full_like(isMC, -999, dtype=np.float64)
    beam_dx[mc_indices] = (reco_beam_calo_startX[mc_indices]-parBQ["beam_startX_mc"])/parBQ["beam_startX_rms_mc"]
    beam_dy[mc_indices] = (reco_beam_calo_startY[mc_indices]-parBQ["beam_startY_mc"])/parBQ["beam_startY_rms_mc"]
    beam_dz[mc_indices] = (reco_beam_calo_startZ[mc_indices]-parBQ["beam_startZ_mc"])/parBQ["beam_startZ_rms_mc"]
    beam_dx[dt_indices] = (reco_beam_calo_startX[dt_indices]-parBQ["beam_startX_data"])/parBQ["beam_startX_rms_data"]
    beam_dy[dt_indices] = (reco_beam_calo_startY[dt_indices]-parBQ["beam_startY_data"])/parBQ["beam_startY_rms_data"]
    beam_dz[dt_indices] = (reco_beam_calo_startZ[dt_indices]-parBQ["beam_startZ_data"])/parBQ["beam_startZ_rms_data"]
    beam_dxy = np.sqrt(np.power(beam_dx, 2) + np.power(beam_dy, 2))

    beamdir_mc = np.array([np.cos(np.deg2rad(parBQ["beam_angleX_mc"])), np.cos(np.deg2rad(parBQ["beam_angleY_mc"])), np.cos(np.deg2rad(parBQ["beam_angleZ_mc"]))])
    beamdir_mc /= np.linalg.norm(beamdir_mc)
    beamdir_data = np.array([np.cos(np.deg2rad(parBQ["beam_angleX_data"])), np.cos(np.deg2rad(parBQ["beam_angleY_data"])), np.cos(np.deg2rad(parBQ["beam_angleZ_data"]))])
    beamdir_data /= np.linalg.norm(beamdir_data)
    beamdir = np.full_like(dir, -999, dtype=np.float64)
    beamdir[mc_indices] = beamdir_mc
    beamdir[dt_indices] = beamdir_data
    beam_costh = np.einsum('ij,ij->i', dir, beamdir)

    inst_dxy = np.full_like(isMC, -999, dtype=np.float64)
    inst_dxy[mc_indices] = np.power( (beam_inst_X[mc_indices]-parBQ["beam_startX_mc_inst"])/parBQ["beam_startX_rms_mc_inst"], 2) + np.power( (beam_inst_Y[mc_indices]-parBQ["beam_startY_mc_inst"])/parBQ["beam_startY_rms_mc_inst"], 2)
    inst_dxy[dt_indices] = np.power( (beam_inst_X[dt_indices]-parBQ["beam_startX_data_inst"])/parBQ["beam_startX_rms_data_inst"], 2) + np.power( (beam_inst_Y[dt_indices]-parBQ["beam_startY_data_inst"])/parBQ["beam_startY_rms_data_inst"], 2)

    return PassBeamQualityCut_xyz(beam_dxy, beam_dz) & PassBeamQualityCut_angle(beam_costh) & PassBeamQualityCut_inst(inst_dxy) & non_empty_mask

def PassBeamQualityCut_xyz(beam_dxy, beam_dz):
    return (beam_dxy > parameters.dxy_min) & (beam_dxy < parameters.dxy_max) & (beam_dz > parameters.dz_min) & (beam_dz < parameters.dz_max)

def PassBeamQualityCut_angle(beam_costh):
    return (beam_costh > parameters.costh_min) & (beam_costh < parameters.costh_max)
    
def PassBeamQualityCut_inst(inst_dxy): # beam scraper cut
    return inst_dxy < parameters.dxy_inst_sq_max

def PassFidVolCut(evt, parBQ):
    reco_beam_calo_endZ = evt["reco_beam_calo_endZ"]
    isMC = evt["MC"] # may not be necessary since the full set is either True or False
    mc_indices = isMC == True
    dt_indices = isMC == False

    pass_upper = reco_beam_calo_endZ < parameters.fidvol_upp

    pass_lower = np.zeros_like(reco_beam_calo_endZ, dtype=bool)
    pass_lower[mc_indices] = reco_beam_calo_endZ[mc_indices] > parameters.fidvol_low
    pass_lower[dt_indices] = reco_beam_calo_endZ[dt_indices] > (parameters.fidvol_low + parBQ["beam_startZ_data"] - parBQ["beam_startZ_mc"])

    return pass_upper & pass_lower

def PassMichelScoreCut(evt):
    reco_beam_vertex_nHits = evt["reco_beam_vertex_nHits"]
    reco_beam_vertex_michel_score_weight_by_charge = evt["reco_beam_vertex_michel_score_weight_by_charge"]

    mask = reco_beam_vertex_nHits != 0
    daughter_michel_score = np.where(mask, reco_beam_vertex_michel_score_weight_by_charge, -999)
    return daughter_michel_score < 0.55

def PassProtonCut(evt):
    reco_beam_calo_wire = evt["reco_beam_calo_wire"]
    reco_beam_Chi2_proton = evt["reco_beam_Chi2_proton"]
    reco_beam_Chi2_ndof = evt["reco_beam_Chi2_ndof"]

    mask = np.array([len(calo_wire) != 0 for calo_wire in reco_beam_calo_wire])
    chi2_protons = np.where(mask, reco_beam_Chi2_proton / reco_beam_Chi2_ndof, -1)
    return chi2_protons > 80

def PassStoppingProtonCut(evt, reco_trklen):
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

    return pass_cut
