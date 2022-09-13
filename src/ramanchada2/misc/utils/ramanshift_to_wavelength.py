import numpy as np


def abs_nm_to_shift_cm_1(deltas, laser_wave_length_nm):
    arr = np.array(list(deltas.items()), dtype=float)
    arr[:, 0] = 1e7*(1/laser_wave_length_nm - 1/arr[:, 0])
    return dict(arr)


def shift_cm_1_to_abs_nm(wn, laser_wave_length_nm):
    shift_nm = wn * 1e-7
    absolute_nm = 1/(1/laser_wave_length_nm - shift_nm)
    return absolute_nm
