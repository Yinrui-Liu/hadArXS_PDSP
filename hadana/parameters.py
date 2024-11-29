from .packages import *

### energy slicing/binning
true_bins_pionp = np.array([1000, 900, 850, 800, 750, 700, 650, 600, 550, 500, 0])
meas_bins_pionp = np.array([1000, 900, 850, 800, 750, 700, 650, 600, 550, 500, 0])
true_bins_proton = np.array([500,450,400,350,300,250,200,150,100,70,40,10,0])
meas_bins_proton = np.array([500,450,400,350,300,250,200,150,100,70,40,10,0])

### fiducial volume
fidvol_low = 30
fidvol_upp = 220

### selection parameters
dxy_min = -1
dxy_max = 3
dz_min = -3
dz_max = 3
costh_min = 0.95
costh_max = 2
dxy_inst_sq_max = 4.5

pionBQ = {
    "beam_startX_mc": -30.47,
    "beam_startX_rms_mc": 4.29,
    "beam_startY_mc": 422.21,
    "beam_startY_rms_mc": 4.03,
    "beam_startZ_mc": 0.108,
    "beam_startZ_rms_mc": 0.192,
    "beam_startX_mc_inst": -29.05,
    "beam_startX_rms_mc_inst": 4.14,
    "beam_startY_mc_inst": 421.71,
    "beam_startY_rms_mc_inst": 3.73,
    "beam_angleX_mc": 101.90,
    "beam_angleY_mc": 101.07,
    "beam_angleZ_mc": 16.31,

    "beam_startX_data": -28.64,
    "beam_startX_rms_data": 4.03,
    "beam_startY_data": 423.84,
    "beam_startY_rms_data": 4.29,
    "beam_startZ_data": 3.145,
    "beam_startZ_rms_data": 0.968,
    "beam_startX_data_inst": -30.79,
    "beam_startX_rms_data_inst": 3.95,
    "beam_startY_data_inst": 422.37,
    "beam_startY_rms_data_inst": 3.58,
    "beam_angleX_data": 100.98,
    "beam_angleY_data": 103.56,
    "beam_angleZ_data": 17.91,
}

protonBQ = {
    "beam_startX_mc": -30.60,
    "beam_startX_rms_mc": 4.30,
    "beam_startY_mc": 422.42,
    "beam_startY_rms_mc": 3.98,
    "beam_startZ_mc": 0.053,
    "beam_startZ_rms_mc": 0.189,
    "beam_startX_mc_inst": -29.21,
    "beam_startX_rms_mc_inst": 4.20,
    "beam_startY_mc_inst": 421.80,
    "beam_startY_rms_mc_inst": 3.72,
    "beam_angleX_mc": 101.92,
    "beam_angleY_mc": 101.06,
    "beam_angleZ_mc": 16.49,

    "beam_startX_data": -28.67,
    "beam_startX_rms_data": 3.76,
    "beam_startY_data": 423.84,
    "beam_startY_rms_data": 4.20,
    "beam_startZ_data": 3.131,
    "beam_startZ_rms_data": 0.963,
    "beam_startX_data_inst": -31.15,
    "beam_startX_rms_data_inst": 3.68,
    "beam_startY_data_inst": 422.12,
    "beam_startY_rms_data_inst": 3.42,
    "beam_angleX_data": 101.00,
    "beam_angleY_data": 104.01,
    "beam_angleZ_data": 18.09,
}
