from ramanchada2.protocols.calibration import CalibrationComponent
from ramanchada2.spectrum import Spectrum

class TwinningComponent(CalibrationComponent):
    def __init__(self, laser_wl, spe, spe_units, ref, ref_units, sample="Ti"):
        super(TwinningComponent, self).__init__(
            laser_wl, spe, spe_units, ref, ref_units, sample
        )
        

    def process(self, old_spe: Spectrum, spe_units="cm-1", convert_back=False):
        raise NotImplementedError(self)

    def derive_model(self, find_kw={}, fit_peaks_kw={}, should_fit=False, name=None):
        raise NotImplementedError(self)
    
    def _plot(self, ax, **kwargs):
        pass    