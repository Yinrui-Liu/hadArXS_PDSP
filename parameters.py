from packages import *

### energy slicing/binning
true_bins_pionp = np.array([1000, 900, 850, 800, 750, 700, 650, 600, 550, 500, 0])
meas_bins_pionp = np.array([1000, 900, 850, 800, 750, 700, 650, 600, 550, 500, 0])

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
    "beam_startX_data": -28.3483,
    "beam_startY_data": 424.553,
    "beam_startZ_data": 3.19841,
    "beam_startX_rms_data": 4.63594,
    "beam_startY_rms_data": 5.21649,
    "beam_startZ_rms_data": 1.2887,
    "beam_startX_mc": -30.6692,
    "beam_startY_mc": 422.263,
    "beam_startZ_mc": 0.1106,
    "beam_startX_rms_mc": 5.172,
    "beam_startY_rms_mc": 4.61689,
    "beam_startZ_rms_mc": 0.212763,
    "beam_angleX_data": 100.464,
    "beam_angleY_data": 103.442,
    "beam_angleZ_data": 17.6633,
    "beam_angleX_mc": 101.547,
    "beam_angleY_mc": 101.247,
    "beam_angleZ_mc": 16.5864,
    "beam_startX_data_inst": -30.9033,
    "beam_startY_data_inst": 422.406,
    "beam_startX_rms_data_inst": 4.17987,
    "beam_startY_rms_data_inst": 3.73181,
    "beam_startX_mc_inst": -28.8615,
    "beam_startY_mc_inst": 421.662,
    "beam_startX_rms_mc_inst": 4.551,
    "beam_startY_rms_mc_inst": 3.90137
}

protonBQ = {
    "beam_startX_data": -28.722,
    "beam_startY_data": 424.216,
    "beam_startZ_data": 3.17107,
    "beam_startX_rms_data": 3.86992,
    "beam_startY_rms_data": 4.59281,
    "beam_startZ_rms_data": 1.22898,
    "beam_startX_mc": -30.734,
    "beam_startY_mc": 422.495,
    "beam_startZ_mc": 0.0494442,
    "beam_startX_rms_mc": 4.75219,
    "beam_startY_rms_mc": 4.23009,
    "beam_startZ_rms_mc": 0.206941,
    "beam_angleX_data": 100.834,
    "beam_angleY_data": 104.119,
    "beam_angleZ_data": 18.2308,
    "beam_angleX_mc": 101.667,
    "beam_angleY_mc": 101.14,
    "beam_angleZ_mc": 16.5398,
    "beam_startX_data_inst": -31.2908,
    "beam_startY_data_inst": 422.105,
    "beam_startX_rms_data_inst": 3.79071,
    "beam_startY_rms_data_inst": 3.43891,
    "beam_startX_mc_inst": -29.0375,
    "beam_startY_mc_inst": 421.82,
    "beam_startX_rms_mc_inst": 4.61462,
    "beam_startY_rms_mc_inst": 3.78259
}
