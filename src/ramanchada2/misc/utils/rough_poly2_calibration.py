import logging

import numpy as np

from .argmin2d import argmin2d
from .poly_params import poly2_by_points
from .significant_peaks import select_peaks

logger = logging.getLogger(__name__)


def _rough_calib_all_to_all(spe_pos, ref_pos, spe_sel_idx, ref_sel_idx):
    last = 0
    for r1_i, r1 in enumerate(ref_pos[ref_sel_idx]):
        for r3_i, r3 in reversed(list(enumerate(ref_pos[ref_sel_idx]))[r1_i+1:]):
            for r2_i, r2 in list(enumerate(ref_pos[ref_sel_idx]))[r1_i+1:r3_i-1]:
                for s1_i, s1 in enumerate(spe_pos[spe_sel_idx]):
                    for s3_i, s3 in reversed(list(enumerate(spe_pos[spe_sel_idx]))[s1_i+1:]):
                        for s2_i, s2 in list(enumerate(spe_pos[spe_sel_idx]))[s1_i+1:s3_i-1]:
                            a, b, c = poly2_by_points(np.array([s1, s2, s3]), np.array([r1, r2, r3]))
                            test_spe_pos = spe_pos**2*a + spe_pos*b + c
                            if np.any(np.diff(test_spe_pos) <= 0):
                                # should be strictly increasing
                                continue
                            w = np.subtract.outer(ref_pos, test_spe_pos)
                            ridx, sidx = argmin2d(np.abs(w)).T
                            mer = - len(ridx)
                            mer = - np.log(len(ridx))
                            if last > mer:
                                deviation = np.inf
                                last = mer
                            if last == mer:
                                d = np.average(np.abs(w[ridx, sidx]))
                                if d < deviation:
                                    deviation = d
                                    logger.info(f'matches={-mer}, deviation={deviation:8.3f}, a={a}, b={b}, c={c}')
                                    print(f'matches={len(ridx)}, deviation={deviation:8.3f}, a={a}, b={b}, c={c}')
                                    lasta = a
                                    lastb = b
                                    lastc = c
    return lasta, lastb, lastc


def rough_poly2_calibration(spe_dict, ref_dict, npeaks=10, **kwargs):
    spe_pos = np.array(list(spe_dict.keys()))
    spe_amp = np.array(list(spe_dict.values()))
    ref_pos = np.array(list(ref_dict.keys()))
    ref_amp = np.array(list(ref_dict.values()))
    npeaks = np.min([npeaks, len(spe_pos), len(ref_pos)])
    spe_sel_idx = select_peaks(spe_pos, spe_amp, npeaks=npeaks, **kwargs)
    ref_sel_idx = select_peaks(ref_pos, ref_amp, npeaks=npeaks, **kwargs)
    return _rough_calib_all_to_all(spe_pos, ref_pos, spe_sel_idx, ref_sel_idx)
