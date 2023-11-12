from ..spectrum import Spectrum
from ..misc import utils as rc2utils
import numpy as np
from scipy import interpolate
from ramanchada2.misc.plottable import Plottable
import matplotlib.pyplot as plt
import ramanchada2.misc.constants  as rc2const

class ProcessingModel:

    def __init__(self):
        pass

class CalibrationComponent(Plottable):
    def __init__(self, laser_wl, spe, spe_units, ref, ref_units):
        super(CalibrationComponent, self).__init__()
        self.laser_wl = laser_wl
        self.spe = spe
        self.spe_units = spe_units
        self.ref = ref
        self.ref_units = ref_units
        self.name = "not estimated"
        self.model = None
        self.model_units = None        
        self.peaks = None

    def set_model(self, model, model_units, peaks, name=None):
        self.model = model
        self.model_units = model_units
        self.peaks = peaks
        self.name = "calibration component" if name is None else name        

    def __str__(self):
        return("{} spe ({}) reference ({}) model ({}) {}".format(self.name,self.spe_units,self.ref_units, self.model_units,self.model))

   
    def convert_units(self,old_spe,spe_unit="cm-1",newspe_unit="nm",laser_wl=None):
        if laser_wl is None:
            laser_wl = self.laser_wl
        print("convert laser_wl {} {} --> {}".format(laser_wl,spe_unit,newspe_unit))
        if spe_unit!=newspe_unit:
            new_spe = old_spe.__copy__() 
            if spe_unit == "nm":
                new_spe = old_spe.abs_nm_to_shift_cm_1_filter(laser_wave_length_nm=laser_wl)
            elif spe_unit == "cm-1":
                new_spe = old_spe.shift_cm_1_to_abs_nm_filter(laser_wave_length_nm=laser_wl)
            else:
                raise Exception("Unsupported conversion {} to {}",spe_unit,newspe_unit)
        else:
            new_spe = old_spe
        #    new_spe = old_spe.__copy__() 
        return new_spe
        
    def process(self,old_spe: Spectrum, spe_units="cm-1",convert_back=False):
        raise NotImplemented(self)
    
    def derive_model(self,find_kw={},fit_peaks_kw={},should_fit = False,name=None):    
        raise NotImplemented(self)
    
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

class XCalibrationComponent(CalibrationComponent):
    def __init__(self, laser_wl, spe, spe_units, ref, ref_units):
        super(XCalibrationComponent, self).__init__( laser_wl, spe, spe_units, ref, ref_units)

    def process(self,old_spe: Spectrum, spe_units="cm-1",convert_back=False):
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
        if convert_back:
            return self.convert_units(new_spe,self.model_units,spe_units)
        else:
            return new_spe
     
    def derive_model(self,find_kw={},fit_peaks_kw={},should_fit = False,name=None):

        #convert to ref_units
        print("convert to ref_units",self.spe_units,self.ref_units)
        spe_to_process = self.convert_units(self.spe,self.spe_units,self.ref_units)
        print(max(spe_to_process.x))
        fig, ax = plt.subplots(3,1,figsize=(12,4))
        self.spe.plot(ax=ax[0].twinx(),label=self.spe_units)    
        spe_to_process.plot(ax=ax[1],label=self.ref_units)
 
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
            print("ref",x_reference[0],"sample", x_spe[0],"offset", _offset, self.ref_units)
            self.set_model(_offset, self.ref_units, df,name)
        else:
            fig, ax = plt.subplots(1,1,figsize=(3,3))
            ax.scatter(x_spe,x_reference,marker='o')
            ax.set_xlabel("spectrum x ".format(self.ref_units))
            ax.set_ylabel("reference x ".format(self.ref_units))
            try:
                kwargs = {"kernel" : "thin_plate_spline"}
                interp = interpolate.RBFInterpolator(x_spe.reshape(-1, 1),x_reference,**kwargs)
                self.set_model( interp, self.ref_units, df,name)
      
            except Exception as err:
                raise(err)     



class ShiftCalibrationComponent(CalibrationComponent):
    def __init__(self, laser_wl, spe, spe_units, ref, ref_units):
        super(ShiftCalibrationComponent, self).__init__( laser_wl, spe, spe_units, ref, ref_units)

    def derive_model(self,find_kw={},fit_peaks_kw={},should_fit = True,name=None):    
        find_kw = dict(sharpening=None)

        cand = self.spe.find_peak_multipeak(**find_kw)
        print(cand)
        #init_guess = self.spe.fit_peak_multimodel(profile='Pearson4', candidates=cand, no_fit=False)
        fit_res = self.spe.fit_peak_multimodel(profile='Pearson4', candidates=cand, **fit_peaks_kw)
        df = fit_res.to_dataframe_peaks()
        df  = df.sort_values(by='height', ascending=False)
        self.set_model( df.iloc[0]["position"], "nm", df,"Shift calibration using".format(self.ref,self.ref_units))

    def process(self,old_spe: Spectrum, spe_units="cm-1",convert_back=False):
        print(self)
        spe_to_process = self.convert_units(old_spe,"nm",self.ref_units,laser_wl=self.model)
        print(self.ref.keys())
        spe_to_process.x = spe_to_process.x +  list(self.ref.keys())[0]
        return spe_to_process   

class CalibrationModel(ProcessingModel,Plottable):
    def __init__(self, laser_wl:int):
        super(ProcessingModel, self).__init__()
        super(Plottable, self).__init__()
        self.set_laser_wavelength(laser_wl)
        self.neon_wl = {
            785: rc2const.neon_wl_785_nist_dict,
            633: rc2const.neon_wl_633_nist_dict,
            532: rc2const.neon_wl_532_nist_dict
        }    
        self.prominence_coeff = 10

    def set_laser_wavelength(self,laser_wl):
        self.clear()
        self.laser_wl = laser_wl
        
    def clear(self):
        self.laser_wl = None
        self.components = []

    def derive_model_x(self,spe_neon,spe_neon_units="cm-1",ref_neon=None,ref_neon_units="nm",spe_sil=None,spe_sil_units="cm-1",ref_sil=None,ref_sil_units="cm-1"):
        model_neon = self.derive_model_curve(spe_neon,self.neon_wl[self.laser_wl],spe_units=spe_neon_units,ref_units=ref_neon_units,find_kw={},fit_peaks_kw={},should_fit = False,name="Neon calibration")
        spe_sil_ne_calib = model_neon.process(spe_sil,spe_units=spe_sil_units,convert_back=False)
        find_kw = {"prominence" :spe_sil_ne_calib.y_noise * prominence_coeff , "wlen" : 200, "width" :  1 }
        model_si = calmodel.derive_model_shift(spe_sil_ne_calib,ref={520.45:1},spe_units="nm",ref_units=ref_sil_units,find_kw=find_kw,fit_peaks_kw={},should_fit=True,name="Si calibration")
        return (model_neon,model_si)

    def derive_model_curve(self,spe,ref,spe_units="cm-1",ref_units="nm",find_kw={},fit_peaks_kw={},should_fit = False,name="X calibration"):
        calibration_x = XCalibrationComponent(self.laser_wl, spe, spe_units, ref, ref_units)
        calibration_x.derive_model(find_kw=find_kw,fit_peaks_kw=fit_peaks_kw,should_fit = should_fit,name=name)
        self.components.append(calibration_x)
        return calibration_x

    def derive_model_shift(self,spe,ref,spe_units="cm-1",ref_units="cm-1",find_kw={},fit_peaks_kw={},should_fit = False,name="X Shift"):
        calibration_shift = ShiftCalibrationComponent(self.laser_wl, spe, spe_units, ref, ref_units)
        print(calibration_shift)
        calibration_shift.derive_model(find_kw=find_kw,fit_peaks_kw=fit_peaks_kw,should_fit = should_fit,name=name)
        print(calibration_shift)
        self.components.append(calibration_shift)
        return calibration_shift
 
    def apply_calibration_x(self,old_spe: Spectrum, spe_units="cm-1"):
        new_spe = old_spe
        for model in self.components:
            #find out if ot convert units
            new_spe = model.process(new_spe,spe_units,convert_back=False)
        return new_spe

    def _plot(self, ax, **kwargs):
        pass