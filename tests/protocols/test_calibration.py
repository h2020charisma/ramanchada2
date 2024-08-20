#!/usr/bin/env python3

import numpy as np
import pytest
import ramanchada2 as rc2
from ramanchada2.protocols.calibration import CalibrationModel, XCalibrationComponent
from ramanchada2.protocols.calibration_io import xcalibration_model_to_json

from sklearn.metrics.pairwise import cosine_similarity
import ramanchada2.misc.constants as rc2const
import matplotlib.pyplot as plt

NEON_WL = {
    785: rc2const.neon_wl_785_nist_dict,
    633: rc2const.neon_wl_633_nist_dict,
    532: rc2const.neon_wl_532_nist_dict
}

from ramanchada2.protocols.calibration_io import to_json


class SetupModule:
    def __init__(self):
        self.spe_neon = rc2.spectrum.from_test_spe(sample=['Neon'], provider=['FNMT'], OP=['03'], laser_wl=['785'])
        self.spe_pst2 = rc2.spectrum.from_test_spe(sample=['PST'], provider=['FNMT'], OP=['02'], laser_wl=['785'])
        self.spe_pst3 = rc2.spectrum.from_test_spe(sample=['PST'], provider=['FNMT'], OP=['03'], laser_wl=['785'])
        self.spe_sil = rc2.spectrum.from_test_spe(sample=['S0B'], provider=['FNMT'], OP=['03'], laser_wl=['785'])
        self.spe_nCal = rc2.spectrum.from_test_spe(sample=['nCAL'], provider=['FNMT'], OP=['03'], laser_wl=['785'])

        self.spe_sil = self.spe_sil.trim_axes(method='x-axis',boundaries=(520.45-200,520.45+200))
        self.spe_neon = self.spe_neon.trim_axes(method='x-axis',boundaries=(100,max(self.spe_neon.x)))    
        kwargs = {"niter" : 40 }
        self.spe_neon = self.spe_neon.subtract_baseline_rc1_snip(**kwargs)  
        self.spe_sil = self.spe_sil.subtract_baseline_rc1_snip(**kwargs)    

        ## normalize min/max
        self.spe_neon = self.spe_neon.normalize()        
        self.spe_sil = self.spe_sil.normalize()        

        self.calmodel, model_neon = calibration_model_x(785,self.spe_neon,self.spe_sil)
        

@pytest.fixture(scope='module')
def setup_module():
    return SetupModule()

def test_serialization(setup_module):
    xcal = setup_module.calmodel.components[0]
    print(f"Methods in xcal: {dir(xcal)}")
    print(type(xcal))
    print(xcal.to_json())
    print(xcalibration_model_to_json(xcal))
    
    

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


def test_xcalibration(setup_module):
    spe_y_original = []
    _min = 200
    _max = 2000
    spe_calibrated = []
    fig, ax = plt.subplots(figsize=(24, 8))
    for spe in [setup_module.spe_pst2,setup_module.spe_pst3]:
        spe_norm = spe.normalize()
        spe_norm.plot(ax=ax,label="original",color="blue")
        spe_y_original.append(resample(spe_norm,_min,_max,_max-_min))
        
        spe = setup_module.calmodel.apply_calibration_x(
                spe,
                spe_units="cm-1"
                )
        #returns spectra in nm!
        spe = setup_module.calmodel.components[0].convert_units(spe, "nm", "cm-1")
        spe_norm = spe.normalize()        
        spe_norm.plot(ax=ax,label="calibrated",color="red")
        spe_calibrated.append(resample(spe_norm,_min,_max,_max-_min))
    cos_sim_matrix_original =  cosine_similarity(spe_y_original)
    cos_sim_matrix =  cosine_similarity(spe_calibrated)

    plt.savefig("{}.png".format("calibration"))
    print(np.mean(cos_sim_matrix_original),np.mean(cos_sim_matrix))
    assert(np.mean(cos_sim_matrix_original) <= np.mean(cos_sim_matrix))
    

    
