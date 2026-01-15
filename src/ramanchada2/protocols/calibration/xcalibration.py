import logging
from typing import Dict, Literal

import numpy as np
import pandas as pd
from ramanchada2.protocols.calibration import qmatch
from ramanchada2.protocols.calibration.interpolators import (
    CustomCubicSplineInterpolator, 
    CustomPChipInterpolator,
    CustomPolyInterpolator,
    CustomRBFInterpolator,
    get_interpolator
)

from ramanchada2.misc.utils import find_closest_pairs_idx

from ramanchada2.misc.utils.matchsets import (
    match_peaks_optimized, match_peaks_monotonic, 
    match_peaks_monotonic_simple, 
    match_peaks_cluster, match_peaks_ready_wrapper
)
from ramanchada2.spectrum import Spectrum
from .calibration_component import CalibrationComponent

logger = logging.getLogger(__name__)


class XCalibrationComponent(CalibrationComponent):
    def __init__(
        self,
        laser_wl,
        spe: Spectrum,
        ref: Dict[float, float],
        spe_units: Literal["cm-1", "nm", "pixel"] = "cm-1",
        ref_units: Literal["cm-1", "nm"] = "nm",
        sample="Neon",
        match_method: Literal["cluster", "argmin2d", "assignment", "monotonic", "dynamicp", "qargmin2d"] = "qargmin2d",
        interpolator_method: Literal["rbf", "pchip", "pchipinverse", "cubic_spline", "pchippolyinverse", "poly"] = "pchipinverse",
        extrapolate=True,
    ):
        super(XCalibrationComponent, self).__init__(
            laser_wl, spe, spe_units, ref, ref_units, sample
        )
        self.spe_pos_dict = None
        self.match_method = match_method
        self.cost_function = None
        self.interpolator_method = interpolator_method
        self.extrapolate = extrapolate

    
    # @staticmethod
    # def from_json(filepath: str):
    #    rbf_intrpolator, other_data = load_xcalibration_model(filepath)
    #    calibration_x = XCalibrationComponent(laser_wl, spe, spe_units, ref, ref_units)
    #    calibration_x.model = rbf_intrpolator
    #    return calibration_x

    def process(
        self,
        old_spe: Spectrum,
        spe_units: Literal["cm-1", "nm", "pixel"] = "cm-1",
        convert_back=False,
    ):
        new_spe = self.convert_units(old_spe, spe_units, self.model_units)
        logger.debug(
            "convert spe_units {} --> model units {}".format(
                spe_units, self.model_units
            )
        )
        if self.model is None:
            return new_spe
        elif self.enabled:
            if isinstance(self.model, float):
                new_spe.x = new_spe.x + self.model
            else:
                if isinstance(self.model, CustomPChipInterpolator):
                    new_spe.x = self.model(new_spe.x)
                elif isinstance(self.model, CustomPolyInterpolator):
                    new_spe.x = self.model(new_spe.x)                    
                elif isinstance(self.model, CustomRBFInterpolator):
                    new_spe.x = self.model(new_spe.x.reshape(-1, 1))
                    if not self.extrapolate:
                        min_train, max_train = self.model.y.min(), self.model.y.max()
                        out_of_bounds = (new_spe.x < min_train) | (
                            new_spe.x > max_train
                        )
                        new_spe.x[out_of_bounds] = np.nan

                elif isinstance(self.model, CustomCubicSplineInterpolator):
                    new_spe.x = self.model(new_spe.x)

                if np.any(np.diff(new_spe.x[np.isfinite(new_spe.x)]) <= 0):
                    if self.nonmonotonic == "error":
                        raise ValueError(f"Non-monotonic values detected (mode={self.nonmonotonic})")
                    elif (self.nonmonotonic == "nan") or (self.nonmonotonic == "drop"):
                        # this is a patch, mostly intended at extrapolation
                        _newx = np.asarray(new_spe.x, dtype=float)
                        is_nonmonotonic = np.diff(_newx, prepend=_newx[0]) <= 0
                        _newx[is_nonmonotonic] = np.nan
                        new_spe.x = _newx
                        if self.nonmonotonic == "drop":
                            new_spe = new_spe.dropna()
                        # we don't necessary ensure monotonicity by setting nans
                        if np.any(np.diff(new_spe.x[np.isfinite(new_spe.x)]) <= 0):
                            raise ValueError(f"Non-monotonic values detected (mode={self.nonmonotonic})")
        if convert_back:
            return self.convert_units(new_spe, self.model_units, spe_units)
        else:
            return new_spe
        

    def _plot(self, ax, **kwargs):
        # Normalize x-positions to [0, 1] for comparison
        ref_keys = np.array(list(self.ref.keys()))
        spe_keys = np.array(list(self.spe_pos_dict.keys()))
        
        ref_norm_x = (ref_keys - ref_keys.min()) / (ref_keys.max() - ref_keys.min())
        spe_norm_x = (spe_keys - spe_keys.min()) / (spe_keys.max() - spe_keys.min())
        
        # Normalize y-values to [0, 1] for each dataset
        ref_vals = np.array(list(self.ref.values()))
        spe_vals = np.array(list(self.spe_pos_dict.values()))
        
        ref_norm_y = (ref_vals - ref_vals.min()) / (ref_vals.max() - ref_vals.min()) if ref_vals.max() > ref_vals.min() else ref_vals
        spe_norm_y = (spe_vals - spe_vals.min()) / (spe_vals.max() - spe_vals.min()) if spe_vals.max() > spe_vals.min() else spe_vals
        
        # Plot spectrum peaks going UP
        ax.stem(
            spe_norm_x,
            spe_norm_y,
            linefmt="b-",
            basefmt="k-",
            label=f"{self.sample} peaks (measured)",
            markerfmt="bo"
        )
        
        # Plot reference peaks going DOWN (negative)
        ax.stem(
            ref_norm_x,
            -ref_norm_y,  # Negative for mirror effect
            linefmt="r-",
            basefmt="k-",
            label=f"Reference peaks",
            markerfmt="ro"
        )
        
        ax.axhline(y=0, color='k', linewidth=0.8)
        ax.set_xlabel("Normalized position [0-1]")
        ax.set_ylabel("Normalized intensity (measured ↑, reference ↓)")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.1, 1.1)  # Give some padding
        
        # Add annotation showing original ranges
        ax.text(0.02, 0.98, f"Spectrum: {spe_keys.min():.1f} - {spe_keys.max():.1f} [{self.spe_units}]",
                transform=ax.transAxes, va='top', fontsize=8, color='b',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if self.ref_units == "cm-1":
            ref_label = r"$\mathrm{cm^{-1}}$"
        else:
            ref_label = self.ref_units
        ax.text(0.02, 0.02, f"Reference: {ref_keys.min():.1f} - {ref_keys.max():.1f} [{ref_label}]",
                transform=ax.transAxes, va='bottom', fontsize=8, color='r',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        

    def _plot_peaks(self, ax, **kwargs):
        # self.model.peaks
        pass
        # fig, ax = plt.subplots(3,1,figsize=(12,4))
        # spe.plot(ax=ax[0].twinx(),label=spe_units)
        # spe_to_process.plot(ax=ax[1],label=ref_units)

    def derive_model(
        self, find_kw=None, fit_peaks_kw=None, should_fit=False, name=None
    ):
        if find_kw is None:
            find_kw = {"sharpening": None}
        if fit_peaks_kw is None:
            fit_peaks_kw = {}
        # convert to ref_units
        if self.spe_units == "pixel":
            logger.debug(self.spe_units)
            pass
        else:
            logger.debug(
                "[{}]: convert spe_units {} to ref_units {}".format(
                    self.name, self.spe_units, self.ref_units
                )
            )
        peaks_df = self.fit_peaks(find_kw, fit_peaks_kw, should_fit)
        x_spe, x_reference, x_distance, cost_matrix, df = match_peaks(
            self.spe_pos_dict, self.ref, self.spe_units, match_method=self.match_method)
        self.cost_matrix = cost_matrix
        self.matched_peaks = df
        # if df is None:
        #    self.matched_peaks = pd.DataFrame({
        #        'spe': x_spe,
        #        'reference': x_reference,
        #        'distances': x_distance
        #    })

        sum_of_differences = np.sum(np.abs(x_spe - x_reference)) / len(x_spe)
        logger.debug(
            "sum_of_differences original {} {}".format(
                sum_of_differences, self.ref_units
            )
        )
        if len(x_reference) == 1:
            _offset = x_reference[0] - x_spe[0]
            logger.debug(
                "ref {} sample {} offset {} {}".format(
                    x_reference[0],
                    x_spe[0],
                    _offset,
                    self.ref_units
                )
            )
            self.set_model(_offset, self.ref_units, peaks_df, name)
        else:
            try:
                interp = get_interpolator(x_spe, x_reference,interpolator_method="pchip")
                self.set_model(interp, self.ref_units, peaks_df, name)
            except Exception as err:
                logger.error(err)
                raise err

    def fit_peaks(self, find_kw, fit_peaks_kw, should_fit):
        spe_to_process = self.convert_units(self.spe, self.spe_units, self.ref_units)
        logger.debug("max x {} {}".format(max(spe_to_process.x), self.ref_units))
        fit_res, spe_pos_dict = fit_peaks(spe_to_process, find_kw, fit_peaks_kw, profile="Gaussian", should_fit=should_fit)
        self.spe_pos_dict = spe_pos_dict
        self.fit_res = fit_res
        return self.fit_res.to_dataframe_peaks()


class LazerZeroingComponent(CalibrationComponent):
    def __init__(
        self,
        laser_wl,
        spe: Spectrum,
        spe_units: Literal["cm-1", "nm"] = "nm",
        ref=None,
        ref_units: Literal["cm-1", "nm"] = "cm-1",
        sample="Silicon",
        profile="Pearson4",
    ):
        if ref is None:
            ref = {520.45: 1}
        super(LazerZeroingComponent, self).__init__(
            laser_wl, spe, spe_units, ref, ref_units, sample
        )
        self.profile = profile

    def derive_model(self, find_kw=None, fit_peaks_kw=None, should_fit=True, name=None):
        if find_kw is None:
            find_kw = {}
        if fit_peaks_kw is None:
            fit_peaks_kw = {}

        cand = self.spe.find_peak_multipeak(**find_kw)
        self.fit_res = self.spe.fit_peak_multimodel(
            profile=self.profile, candidates=cand, **fit_peaks_kw,
            bound_centers_to_group=True
        )

        df = self.fit_res.to_dataframe_peaks()
        # df = self.fitres2df(self.spe)
        # highest peak first
        logger.debug(f"{df.shape} {df.columns}")
        
        # df = df.sort_values(by='amplitude', ascending=False)
        if df.empty:
            raise Exception("No peaks found")
        else:
            df = df.sort_values(by="height", ascending=False)            
            if "position" in df.columns:
                zero_peak_nm = df.iloc[0]["position"]
            elif "center" in df.columns:
                zero_peak_nm = df.iloc[0]["center"]
            # https://www.elodiz.com/calibration-and-validation-of-raman-instruments/
            zero_peak_cm1 = self.zero_nm_to_shift_cm_1(
                zero_peak_nm, zero_peak_nm, list(self.ref.keys())[0]
            )
            self.set_model(
                zero_peak_nm,
                "nm",
                df,
                "Laser zeroing using {} nm {} cm¯¹ ({}) ".format(
                    zero_peak_nm, zero_peak_cm1, self.profile
                ),
            )
            logger.info(f"{self.name} peak {self.profile} at {zero_peak_nm} nm")
        # laser_wl should be calculated  based on the peak position and set instead of the nominal

    def zero_nm_to_shift_cm_1(self, wl, zero_pos_nm, zero_ref_cm_1=520.45):
        return 1e7 * (1 / zero_pos_nm - 1 / wl) + zero_ref_cm_1

    # we do not do shift (as initially implemented)
    # just convert the spectrum nm->cm-1 using the Si measured peak in nm and reference in cm-1
    # https://www.elodiz.com/calibration-and-validation-of-raman-instruments/
    def process(
        self,
        old_spe: Spectrum,
        spe_units: Literal["cm-1", "nm"] = "nm",
        convert_back=False,
    ):
        wl_si_ref = list(self.ref.keys())[0]
        logger.debug(f"{self.name}, process, {self.model}, {wl_si_ref}")
        new_x = self.zero_nm_to_shift_cm_1(old_spe.x, self.model, wl_si_ref)
        new_spe = Spectrum(x=new_x, y=old_spe.y, metadata=old_spe.meta)
        # new_spe = old_spe.lazer_zero_nm_to_shift_cm_1(self.model, wl_si_ref)
        # print("old si", old_spe.x)
        # print("new si", new_spe.x)
        return new_spe

    def _plot(self, ax, **kwargs):
        # spe_sil.plot(label="{} original".format(si_tag),ax=ax)
        # spe_sil_calib.plot(ax = ax,label="{} laser zeroed".format(si_tag),fmt=":")
        # ax.set_xlim(520.45-50,520.45+50)
        # ax.set_xlabel("cm-1")
        pass


def match_peaks(spe_pos_dict, ref_dict, spe_units, match_method="qargmin2d"):
    _match_method = match_method
    if spe_units == "pixel" and match_method != "qargmin2d":
        _match_method = "dynamicp"
    logger.debug(f"{_match_method} spe_pos_dict {spe_pos_dict}, \nref {ref_dict}")
    if _match_method == "cluster":
        x_spe, x_reference, x_distance, _ = match_peaks_cluster(
            spe_pos_dict, ref_dict,
            #_filter_range = self.spe_units != "pixel"
        )
        x_inliers, y_inliers, inlier_mask = qmatch.iterative_linear_filter(
            x_spe, x_reference, n_sigma=3
        )            
        logger.debug(f"Outliers found {len(x_spe)-len(x_inliers)}")
        cost_matrix = None
        df = pd.DataFrame(
            {
                "spe": x_spe,
                "reference": x_reference,
                "distances": x_spe - x_reference,
                "inlier_mask" : inlier_mask
            }
        )
        return x_inliers, y_inliers, x_inliers-y_inliers, cost_matrix, df
    elif _match_method == "dynamicp":
        x_spe, x_reference, cost_matrix, df = match_peaks_ready_wrapper(
            spe_pos_dict, ref_dict,
        )
        x_inliers, y_inliers, inlier_mask = qmatch.iterative_linear_filter(
            x_spe, x_reference, n_sigma=3
        )
        df["inlier_mask"] = inlier_mask           
        logger.debug(f"Outliers found {len(x_spe)-len(x_inliers)}")        
        return x_inliers, y_inliers, x_inliers - y_inliers, cost_matrix, df
    elif _match_method == "qargmin2d":
        x = np.array(list(spe_pos_dict.keys()))
        y = np.array(list(ref_dict.keys()))
        if spe_units == "pixel":
            x_idx, y_idx = qmatch.find_closest_pairs_quantile_idx(x,y,n_sigma=3)
        else:
            x_idx, y_idx = find_closest_pairs_idx(x, y)
        x_spe = x[x_idx]
        x_reference = y[y_idx]
        # Sort by x
        idx = np.argsort(x_spe)
        x_spe = x_spe[idx]
        x_reference = x_reference[idx]     
        #iterative_linear_filter       
        x_inliers, y_inliers, inlier_mask = qmatch.linear_residual_filter(
            x_spe, x_reference, n_sigma=3
        )
        logger.debug(f"Outliers found {len(x_spe)-len(x_inliers)}")
        df = pd.DataFrame(
            {
                "spe": x_spe,
                "reference": x_reference,
                "distances": x_spe - x_reference,
                "inlier_mask" : inlier_mask
            }
        )
        return  x_inliers, y_inliers, x_inliers-y_inliers, None, df        
    elif _match_method == "argmin2d":
        x = np.array(list(spe_pos_dict.keys()))
        y = np.array(list(ref_dict.keys()))
        x_idx, y_idx = find_closest_pairs_idx(x, y)
        x_spe = x[x_idx]
        x_reference = y[y_idx]
        df = pd.DataFrame(
            {
                "spe": x_spe,
                "reference": x_reference,
                "distances": x_spe - x_reference,
                "inlier_mask": True
            }
        )
        return x_spe, x_reference, x_spe - x_reference, None, df
    elif _match_method == "assignment":  # https://en.wikipedia.org/wiki/Hungarian_algorithm
        try:
            x_spe, x_reference, x_distance, cost_matrix, df = match_peaks_optimized(
                spe_pos_dict=spe_pos_dict,
                ref=ref_dict,
                tolerance=100, relative=False, weight_intensity=0.9
            )
            return x_spe, x_reference, x_distance, cost_matrix, df
        except Exception as err:
            logger.warning(f"{err} Reverting to monotonic match")
            x_spe, x_reference, x_distance,  df = match_peaks_monotonic(
                spe_pos_dict=spe_pos_dict,
                ref=ref_dict,
                tolerance=None, relative=False, weight_intensity=0.25
            )
            return x_spe, x_reference, x_distance, None, df
    else:  # self.match_method == "monotonic":
        try:
            x_spe, x_reference, x_distance,  df = match_peaks_monotonic_simple(
                spe_pos_dict=spe_pos_dict,
                ref=ref_dict,
                tolerance=100,
                relative=False,
                weight_intensity=.5
            )
            x_inliers, y_inliers, inlier_mask = qmatch.linear_residual_filter(
                x_spe, x_reference, n_sigma=3
            )
            df["inlier_mask"] = inlier_mask
            logger.debug(f"Outliers found {len(x_spe)-len(x_inliers)}")            
            return x_inliers, y_inliers, x_inliers-y_inliers, None, df
        except Exception as err:
            raise err


def fit_peaks(spe_to_process, find_kw, fit_peaks_kw, profile="Gaussian", should_fit=True):

    fit_res = None

    # instead of fit_peak_positions - we don't want movmin here
    # baseline removal might be done during preprocessing
    center_err_threshold = 0.5
    find_kw.update(dict(sharpening=None))
    cand = spe_to_process.find_peak_multipeak(**find_kw)
    # print(cand.get_ampl_pos_fwhm())

    fit_res = spe_to_process.fit_peak_multimodel(
        profile=profile, candidates=cand, **fit_peaks_kw, no_fit=not should_fit,
        bound_centers_to_group=True
    )
    if should_fit:
        pos, amp = fit_res.center_amplitude(threshold=center_err_threshold)
        spe_pos_dict = dict(zip(pos, amp))
    else:
        spe_pos_dict = cand.get_pos_ampl_dict()
    return fit_res, spe_pos_dict


def match_peaks4analysis(
        spectra, ref=None, spe_units="nm", 
        find_kw=None, fit_peaks_kw=None, profile="Gaussian", should_fit=True,
        match_method = "qargmin2d",
        stages=["1.original"]):
    if spectra is None or ref is None:
        return None
    matched_peaks = None
    for spe, stage in list(zip(spectra, stages)):
        fit_res, spe_pos_dict = fit_peaks(
            spe, find_kw, fit_peaks_kw, profile=profile, should_fit=should_fit)
        _x, _ref, _, _, df_calib = match_peaks(
            spe_pos_dict, ref, spe_units =spe_units, match_method=match_method)    
        df_calib["match_mode"] = match_method
        df_calib["before_after"] = stage
        matched_peaks = df_calib if matched_peaks is None else pd.concat([matched_peaks, df_calib])
    return matched_peaks
