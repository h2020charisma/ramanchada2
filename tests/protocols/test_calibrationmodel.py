#!/usr/bin/env python3

import numpy as np
import pytest
import ramanchada2 as rc2
from ramanchada2.protocols.calibration import CalibrationModel, XCalibrationComponent

from sklearn.metrics.pairwise import cosine_similarity
import ramanchada2.misc.constants as rc2const
import matplotlib.pyplot as plt
import traceback


class SetupModule:
    def __init__(self):
        self.spe_neon = rc2.spectrum.from_test_spe(sample=['Neon'], provider=['FNMT'], OP=['03'], laser_wl=['785'])
        self.spe_pst2 = rc2.spectrum.from_test_spe(sample=['PST'], provider=['FNMT'], OP=['02'], laser_wl=['785'])
        self.spe_pst3 = rc2.spectrum.from_test_spe(sample=['PST'], provider=['FNMT'], OP=['03'], laser_wl=['785'])
        self.spe_sil = rc2.spectrum.from_test_spe(sample=['S0B'], provider=['FNMT'], OP=['03'], laser_wl=['785'])
        self.spe_nCal = rc2.spectrum.from_test_spe(sample=['nCAL'], provider=['FNMT'], OP=['03'], laser_wl=['785'])

        self.spe_sil = self.spe_sil.trim_axes(method='x-axis',boundaries=(520.45-100,520.45+100))
        self.spe_neon = self.spe_neon.trim_axes(method='x-axis',boundaries=(100,max(self.spe_neon.x)))    
        kwargs = {"niter" : 40 }
        self.spe_neon = self.spe_neon.subtract_baseline_rc1_snip(**kwargs)  
        self.spe_sil = self.spe_sil.subtract_baseline_rc1_snip(**kwargs)    

        ## normalize min/max
        self.spe_neon = self.spe_neon.normalize()        
        self.spe_sil = self.spe_sil.normalize()        

        try:
            neon_wl = rc2const.NEON_WL[785]
            self.calmodel = CalibrationModel.calibration_model_factory(785,
                                            self.spe_neon,self.spe_sil,neon_wl = neon_wl,
                                            find_kw={"wlen" : 200, "width" :  1}, fit_peaks_kw={},should_fit=False)
            assert len(self.calmodel.components)==2
            #print(self.calmodel.components[1].profile, self.calmodel.components[1].peaks)
        except Exception as err:
            self.calmodel = None
            traceback.print_exc()
        

@pytest.fixture(scope='module')
def setup_module():
    return SetupModule()

def test_laser_zeroing(setup_module):
    assert setup_module.calmodel is not None
    fig, ax =plt.subplots(1,1,figsize=(12,2))
    spe_sil_calib = setup_module.calmodel.apply_calibration_x(
                setup_module.spe_sil,
                spe_units="cm-1"
                )
    setup_module.spe_sil.plot(label="Si original",ax=ax)
    spe_sil_calib.plot(ax = ax,label="Si laser zeroed",fmt=":")
    #ax.set_xlim(520.45-50,520.45+50)    

    ax.set_xlabel(setup_module.calmodel.components[1].model_units)
    #print(setup_module.calmodel.components[1])
    plt.savefig("{}.png".format("laser_zeroing"))
    
  

def resample(spe,xmin,xmax,npoints):
    x_values = np.linspace(xmin, xmax, npoints)
    dist = spe.spe_distribution(trim_range=(xmin, xmax))
    y_values = dist.pdf(x_values)
    scale = np.max(spe.y) / np.max(y_values)
    # pdf sampling is normalized to area unity, scaling back
    #tbd - make resample a filter
    return y_values *  scale


def resample_NUDFT(spe,xmin,xmax,npoints):
    spe_resampled = spe.resample_NUDFT_filter(x_range=(xmin,xmax), xnew_bins=npoints)
    return spe_resampled.y

def compare_calibrated_spe(setup_module,spectra,name="calibration"):
    fig, ax = plt.subplots(2,1,figsize=(24, 8))
    setup_module.calmodel.plot(ax = ax[1])
    crl = [("blue","red"),("green","black")]
    spe_y_original = []
    _min = 200
    _max = 2000
    spe_calibrated = []
    for index,spe in enumerate(spectra):
        spe_norm = spe.normalize()
        spe_norm.plot(ax=ax[0],label=f"original {index}",color=crl[index][0])

        #resample with NUDFT
        #spe_y_original.append(resample_NUDFT(spe_norm,_min,_max,_max-_min))
        
        #resample with histogram
        spe_y_original.append(resample(spe_norm,_min,_max,_max-_min))
        
        spe_c = setup_module.calmodel.apply_calibration_x(
                spe,
                spe_units="cm-1"
                )
        spe_c_norm = spe_c.normalize()        
        spe_c_norm.plot(ax=ax[0],label=f"calibrated {index}",color=crl[index][1])
        spe_calibrated.append(resample(spe_c_norm,_min,_max,_max-_min))
    cos_sim_matrix_original =  cosine_similarity(spe_y_original)
    cos_sim_matrix =  cosine_similarity(spe_calibrated)

    plt.savefig("{}.png".format(name))
    print(name,np.mean(cos_sim_matrix_original),np.mean(cos_sim_matrix))
    assert(np.mean(cos_sim_matrix_original) <= np.mean(cos_sim_matrix))
    
def test_xcalibration_pst(setup_module):
    assert setup_module.calmodel is not None
    compare_calibrated_spe(setup_module,[setup_module.spe_pst2,setup_module.spe_pst3],"PST") 


def test_xcalibration_si(setup_module):
    assert setup_module.calmodel is not None
    compare_calibrated_spe(setup_module,[setup_module.spe_sil,setup_module.spe_sil],"Sil") 
    
