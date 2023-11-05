from ..spectrum import Spectrum
from ..misc import utils as rc2utils
import numpy as np
from scipy import interpolate
from ramanchada2.misc.plottable import Plottable
import matplotlib.pyplot as plt

class ProcessingModel:

    def __init__(self):
        pass

class XCalibrationComponent(Plottable):
    def __init__(self, laser_wl, spe, spe_units, ref, ref_units):
        super(Plottable, self).__init__()
        self.laser_wl = laser_wl
        self.spe = spe
        self.spe_units = spe_units
        self.ref = ref
        self.model_units = ref_units
        self.name = "not estimated"
        self.model = None
        self.peaks = None

    def set_model(self, model, model_units, peaks, name=None):
        self.model = model
        self.model_units = model_units
        self.peaks = peaks
        self.name = "calibration component" if name is None else name

    def __str__(self):
        return("{} spe ({}) model ({}) {}".format(self.name,self.spe_units,self.model_units,self.model))
   
    def convert_units(self,old_spe,spe_unit="cm-1",newspe_unit="nm"):
        print("convert {} --> {}".format(spe_unit,newspe_unit))
        if spe_unit!=newspe_unit:
            new_spe = old_spe.__copy__() 
            if spe_unit == "nm":
                new_spe = old_spe.abs_nm_to_shift_cm_1_filter(laser_wave_length_nm=self.laser_wl)
            elif spe_unit == "cm-1":
                new_spe = old_spe.shift_cm_1_to_abs_nm_filter(laser_wave_length_nm=self.laser_wl)
            else:
                raise Exception("Unsupported conversion {} to {}",spe_unit,new_spe_unit)
        else:
            new_spe = old_spe
        #    new_spe = old_spe.__copy__() 
        return new_spe
            

    def process(self,old_spe: Spectrum, spe_units="cm-1"):
        print("convert to ", spe_units, self.model_units)
        new_spe = self.convert_units(old_spe,spe_units,self.model_units)
        print("process", self)
        if self.model is None: 
            return new_spe        
        elif isinstance(self.model, interpolate.RBFInterpolator):
            new_spe.x = self.model(new_spe.x.reshape(-1, 1)) 
        elif isinstance(self.model, float):
            new_spe.x = new_spe.x + self.model
        #convert back
        print("convert back", spe_units)
        return self.convert_units(new_spe,self.model_units,spe_units)
     
    def derive_model(self,find_kw={},fit_peaks_kw={},should_fit = False,name=None):
        #convert to ref_units
        print("convert to ref_units",self.spe_units,self.model_units)
        spe_to_process = self.convert_units(self.spe,self.spe_units,self.model_units)
        print(max(spe_to_process.x))
        fig, ax = plt.subplots(3,1,figsize=(12,4))
        self.spe.plot(ax=ax[0].twinx(),label=self.spe_units)    
        spe_to_process.plot(ax=ax[1],label=self.model_units)
 
        #if should_fit:
        #    spe_pos_dict = spe_to_process.fit_peak_positions(center_err_threshold=1, 
        #                        find_peaks_kw=find_kw,  fit_peaks_kw=fit_peaks_kw)  # type: ignore   
        #else:
        #    find_kw = dict(sharpening=None)
        #    spe_pos_dict = spe_to_process.find_peak_multipeak(**find_kw).get_pos_ampl_dict()  # type: ignore
        #prominence=prominence, wlen=wlen, width=width
        find_kw = dict(sharpening=None)
        if should_fit:
            spe_pos_dict = spe_to_process.fit_peak_positions(center_err_threshold=10, 
                                find_peaks_kw=find_kw,  fit_peaks_kw=fit_peaks_kw)  # type: ignore           
            #fit_res = spe_to_process.fit_peak_multimodel(candidates=cand,**fit_peaks_kw)
            #pos, amp = fit_res.center_amplitude(threshold=1)
            #spe_pos_dict = dict(zip(pos, amp))        
        else:
            #prominence=prominence, wlen=wlen, width=width
            cand = spe_to_process.find_peak_multipeak(**find_kw)
            spe_pos_dict = cand.get_pos_ampl_dict()        

        ax[2].stem(spe_pos_dict.keys(),spe_pos_dict.values(),linefmt='b-', basefmt=' ')
        ax[2].twinx().stem(self.ref.keys(),self.ref.values(),linefmt='r-', basefmt=' ')
    
        x_spe,x_reference,x_distance,df = rc2utils.match_peaks(spe_pos_dict, self.ref)
        sum_of_differences = np.sum(np.abs(x_spe - x_reference)) / len(x_spe)
        #print("sum_of_differences original {} {}".format(sum_of_differences, ref_units))
        if len(x_reference)==1:
            _offset = ( x_reference[0] - x_spe[0])
            print("ref",x_reference[0],"sample", x_spe[0],"offset", _offset, self.model_units)
            self.set_model(_offset, self.model_units, df,name)
        else:
            fig, ax = plt.subplots(1,1,figsize=(3,3))
            ax.scatter(x_spe,x_reference,marker='o')
            ax.set_xlabel("spectrum x ".format(self.model_units))
            ax.set_ylabel("reference x ".format(self.model_units))
            try:
                kwargs = {"kernel" : "thin_plate_spline"}
                interp = interpolate.RBFInterpolator(x_spe.reshape(-1, 1),x_reference,**kwargs)
                self.set_model( interp, self.model_units, df,name)
      
            except Exception as err:
                raise(err)     

    def plot(self, ax=None, label=' ', **kwargs) -> plt.axes:
        if ax is None:
            fig, ax = plt.subplots(3,1)
        self._plot(ax[0], label=label, **kwargs)
        ax.legend()
        return ax
    
    def _plot(self, ax, **kwargs):
        self.spe.plot(ax=ax)

    def _plot_peaks(self, ax, **kwargs):
        pass
        #fig, ax = plt.subplots(3,1,figsize=(12,4))
        #spe.plot(ax=ax[0].twinx(),label=spe_units)    
        #spe_to_process.plot(ax=ax[1],label=ref_units)

class CalibrationModel(ProcessingModel,Plottable):
    def __init__(self, laser_wl:int):
        super(ProcessingModel, self).__init__()
        super(Plottable, self).__init__()
        self.set_laser_wavelength(laser_wl)

    def set_laser_wavelength(self,laser_wl):
        self.clear()
        self.laser_wl = laser_wl
        
    def clear(self):
        self.laser_wl = None
        self.components = []

    def create(self,spe,ref,spe_units="cm-1",ref_units="nm",find_kw={},fit_peaks_kw={},should_fit = False,name=None):
        calibration_x = XCalibrationComponent(self.laser_wl, spe, spe_units, ref, ref_units)
        print(calibration_x)
        calibration_x.derive_model(find_kw=find_kw,fit_peaks_kw=fit_peaks_kw,should_fit = should_fit,name=name)
        print(calibration_x)
        self.components.append(calibration_x)
        return calibration_x
        #if should_fit:
        #    spe_pos_dict = spe_to_process.fit_peak_positions(center_err_threshold=1, 
        #                        find_peaks_kw=find_kw,  fit_peaks_kw=fit_peaks_kw)  # type: ignore   
        #else:
        #    find_kw = dict(sharpening=None)
        #    spe_pos_dict = spe_to_process.find_peak_multipeak(**find_kw).get_pos_ampl_dict()  # type: ignore
        #prominence=prominence, wlen=wlen, width=width
       
    def apply_calibration_x(self,old_spe: Spectrum, spe_units="cm-1"):
        new_spe = old_spe
        for model in self.components:
            new_spe = model.process(new_spe,spe_units)
        return new_spe

    def _plot(self, ax, **kwargs):
        pass