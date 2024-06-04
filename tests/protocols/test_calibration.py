#!/usr/bin/env python3

import numpy as np
import pytest
import ramanchada2 as rc2
from ramanchada2.protocols.calibration import CalibrationModel

from sklearn.metrics.pairwise import cosine_similarity
import ramanchada2.misc.constants as rc2const

NEON_WL = {
    785: rc2const.neon_wl_785_nist_dict,
    633: rc2const.neon_wl_633_nist_dict,
    532: rc2const.neon_wl_532_nist_dict
}

def calibration_model_x(laser_wl,spe_neon,spe_sil,neon_wl = NEON_WL):
    calmodel = CalibrationModel(laser_wl)
    calmodel.prominence_coeff = 3
    model_neon = calmodel.derive_model_curve(spe_neon,neon_wl[laser_wl],spe_units="cm-1",ref_units="nm",find_kw={},fit_peaks_kw={},should_fit = False,name="Neon calibration")
    return calmodel, model_neon

def resample(spe,xmin,xmax,npoints):
    x_values = np.linspace(xmin, xmax, npoints)
    dist = spe.spe_distribution(trim_range=(xmin, xmax))
    y_values = dist.pdf(x_values)
    scale = np.max(spe.y) / np.max(y_values)
        # pdf sampling is normalized to area unity, scaling back
        #tbd - make resample a filter
    return y_values *  scale


def test_xcalibration():
    spe_neon = rc2.spectrum.from_test_spe(sample=['Neon'], provider=['FNMT'], OP=['03'], laser_wl=['785'])
    spe_pst2 = rc2.spectrum.from_test_spe(sample=['PST'], provider=['FNMT'], OP=['02'], laser_wl=['785'])
    spe_pst3 = rc2.spectrum.from_test_spe(sample=['PST'], provider=['FNMT'], OP=['03'], laser_wl=['785'])
    spe_sil = rc2.spectrum.from_test_spe(sample=['S0B'], provider=['FNMT'], OP=['03'], laser_wl=['785'])
    spe_nCal = rc2.spectrum.from_test_spe(sample=['nCAL'], provider=['FNMT'], OP=['03'], laser_wl=['785'])

    spe_sil = spe_sil.trim_axes(method='x-axis',boundaries=(520.45-200,520.45+200))
    spe_neon = spe_neon.trim_axes(method='x-axis',boundaries=(100,max(spe_neon.x)))    
    kwargs = {"niter" : 40 }
    spe_neon = spe_neon.subtract_baseline_rc1_snip(**kwargs)  
    spe_sil = spe_sil.subtract_baseline_rc1_snip(**kwargs)    

    ## normalize min/max
    spe_neon = spe_neon.normalize()        
    spe_sil = spe_sil.normalize()        

    spe_y_original = []
    _max = 2000
    for spe in [spe_pst2,spe_pst3]:
        spe_y_original.append(resample(spe,100,_max+100,_max))

    calmodel, model_neon = calibration_model_x(785,spe_neon,spe_sil)
    spe_y = []
    for spe in [spe_pst2,spe_pst3]:
        spe = calmodel.apply_calibration_x(
                spe,
                spe_units="cm-1"
                )
        spe_y.append(resample(spe,100,_max+100,_max))
    cos_sim_matrix_original =  cosine_similarity(spe_y_original)
    print(cos_sim_matrix_original)
    cos_sim_matrix =  cosine_similarity(spe_y)
    print(cos_sim_matrix)
    print(np.mean(cos_sim_matrix_original),np.mean(cos_sim_matrix))
    assert(np.mean(cos_sim_matrix_original) <= np.mean(cos_sim_matrix))
    
