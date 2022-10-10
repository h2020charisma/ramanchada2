import numpy as np


def shift_cm_1_to_abs_nm_dict(deltas, laser_wave_length_nm):
    arr = np.array(list(deltas.items()), dtype=float)
    arr[:, 0] = 1/(1/laser_wave_length_nm - arr[:, 0] * 1e-7)
    return dict(arr)


def abs_nm_to_shift_cm_1_dict(deltas, laser_wave_length_nm):
    arr = np.array(list(deltas.items()), dtype=float)
    arr[:, 0] = 1e7*(1/laser_wave_length_nm - 1/arr[:, 0])
    return dict(arr)


def abs_nm_to_shift_cm_1(wl, laser_wave_length_nm):
    return 1e7*(1/laser_wave_length_nm - 1/wl)


def shift_cm_1_to_abs_nm(wn, laser_wave_length_nm):
    shift_nm = wn * 1e-7
    absolute_nm = 1/(1/laser_wave_length_nm - shift_nm)
    return absolute_nm


def laser_wl_nm(raman_shift_cm_1, wave_length_nm):
    return 1/(1/wave_length_nm+raman_shift_cm_1*1e-7)
