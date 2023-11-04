from ..spectrum import Spectrum
from ...misc import utils as rc2utils
import numpy as np
from scipy import interpolate

def calibration_model_x(laser_wl,spe,ref,spe_units="cm-1",ref_units="nm",find_kw={},fit_peaks_kw={},should_fit = False):
    print("calibration_model laser_wl {} spe ({}) reference ({})".format(laser_wl,spe_units,ref_units))
    #convert to ref_units
    spe_to_process = None #spe_to_process.__copy__()
    if ref_units == "nm":
        if spe_units != "nm":
            spe_to_process = spe.shift_cm_1_to_abs_nm_filter(laser_wave_length_nm=laser_wl)
    else: #assume cm-1
        if spe_units != "cm-1":
            spe_to_process = spe.abs_nm_to_shift_cm_1_filter(laser_wave_length_nm=laser_wl)
    if spe_to_process is None:
       spe_to_process = spe.__copy__() 
    #fig, ax = plt.subplots(3,1,figsize=(12,4))
    #spe.plot(ax=ax[0].twinx(),label=spe_units)    
    #spe_to_process.plot(ax=ax[1],label=ref_units)
    
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

    #ax[2].stem(spe_pos_dict.keys(),spe_pos_dict.values(),linefmt='b-', basefmt=' ')
    #ax[2].twinx().stem(ref.keys(),ref.values(),linefmt='r-', basefmt=' ')
   
   
    x_spe,x_reference,x_distance,df = rc2utils.match_peaks(spe_pos_dict, ref)
    sum_of_differences = np.sum(np.abs(x_spe - x_reference)) / len(x_spe)
    #print("sum_of_differences original {} {}".format(sum_of_differences, ref_units))
    if len(x_reference)==1:
        _offset = ( x_reference[0] - x_spe[0])
        print("ref",x_reference[0],"sample", x_spe[0],"offset", _offset, ref_units)
        return ( _offset ,ref_units, df)
    else:
        #fig, ax = plt.subplots(1,1,figsize=(3,3))
        #ax.scatter(x_spe,x_reference,marker='o')
        #ax.set_xlabel("spectrum x ".format(ref_units))
        #ax.set_ylabel("reference x ".format(ref_units))
        try:
            kwargs = {"kernel" : "thin_plate_spline"}
            return (interpolate.RBFInterpolator(x_spe.reshape(-1, 1),x_reference,**kwargs) ,ref_units,  df)
        except Exception as err:
            raise(err)


def apply_calibration(old_spe: Spectrum,
                    laser_wl, interp_cal=None, offset=0,spe_units="cm-1",model_units="nm"):
    print("apply_calibration laser_wl {} spe ({}) model ({}) interp {} offset {}".format(laser_wl,spe_units,model_units,interp_cal,offset))
    new_spe = old_spe.__copy__()    
    if spe_units!=model_units:
        if model_units == "nm":
            new_spe = new_spe.shift_cm_1_to_abs_nm_filter(laser_wave_length_nm=laser_wl)     
        else:
            new_spe =  new_spe.abs_nm_to_shift_cm_1_filter(laser_wave_length_nm=laser_wl)
    if interp_cal != None:
        new_spe.x = interp_cal(new_spe.x.reshape(-1, 1)) 
    new_spe.x = new_spe.x + offset
    #convert back
    if spe_units!=model_units:
        if model_units == "nm":
            new_spe = new_spe.abs_nm_to_shift_cm_1_filter(laser_wave_length_nm=laser_wl)
        else:
            new_spe = new_spe.shift_cm_1_to_abs_nm_filter(laser_wave_length_nm=laser_wl)  
    return new_spe    
