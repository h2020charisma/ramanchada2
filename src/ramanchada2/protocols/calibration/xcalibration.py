import json
import logging
from typing import Dict, Literal

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator, RBFInterpolator
from ramanchada2.protocols.calibration import qmatch

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
        match_method: Literal["cluster", "argmin2d", "assignment", "monotonic", "dynamicp", "qargmin2d"] = "cluster",
        interpolator_method: Literal["rbf", "pchip", "cubic_spline", "pchippoly"] = "pchip",
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
            print(self.spe_units)
            pass
        else:
            logger.debug(
                "[{}]: convert spe_units {} to ref_units {}".format(
                    self.name, self.spe_units, self.ref_units
                )
            )
        peaks_df = self.fit_peaks(find_kw, fit_peaks_kw, should_fit)
        x_spe, x_reference, x_distance, cost_matrix, df = self.match_peaks(
            threshold_max_distance=None, return_df=True
        )
        print(list(zip(x_spe, x_reference)))
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
                if self.interpolator_method in ["pchip", "pchippoly"]:
    # --------------------------------------------------------
    # Inverse spline: spe = f(reference)
    # --------------------------------------------------------                    
                    inverse = CustomPChipInterpolator(x_reference, x_spe)
    # --------------------------------------------------------
    # Dense sampling in reference space
    # --------------------------------------------------------                    
                    dense_reference = np.linspace(
                        x_reference[0],
                        x_reference[-1],
                        max(2048, 10*len(x_reference)) # N≫number of original knots
                        # Anything smaller gives you no benefit over the raw calibration.
                    )
    # --------------------------------------------------------
    # Enforce monotonic ordering (important for polyfit)
    # --------------------------------------------------------
                    dense_spe = inverse(dense_reference)
                    order = np.argsort(dense_spe)
                    dense_spe = dense_spe[order]
                    dense_reference = dense_reference[order]  
                    if  self.interpolator_method == "pchip":
                        interp = CustomPChipInterpolator(dense_spe, dense_reference)
                    else:
                        interp = CustomPolyInterpolator(dense_spe, dense_reference)
                    # direct
                    #interp_raw = CustomPChipInterpolator(x_spe, x_reference)
                elif self.interpolator_method == "cubic_spline":
                    kwargs = {"bc_type": "clamped"}
                    interp = CustomCubicSplineInterpolator(x_spe, x_reference, **kwargs)
                elif self.interpolator_method == "rbf":
                    kwargs = {
                        "kernel": "thin_plate_spline",
                        "neighbors": len(x_spe) / 3,
                        "smoothing": 0,
                    }
                    interp = CustomRBFInterpolator(
                        x_spe.reshape(-1, 1), x_reference, **kwargs
                    )
                self.set_model(interp, self.ref_units, peaks_df, name)
            except Exception as err:
                print(err)
                raise err

    def match_peaks(self, threshold_max_distance=9, return_df=False):
        _match_method = self.match_method
        if self.spe_units == "pixel" and self.match_method != "qargmin2d":
            _match_method = "dynamicp"
        print(self.match_method)
        print(f"spe_pos_dict {self.spe_pos_dict}, \nref {self.ref}")
        if _match_method == "cluster":
            x_spe, x_reference, x_distance, _ = match_peaks_cluster(
                self.spe_pos_dict, self.ref,
                #_filter_range = self.spe_units != "pixel"
            )
            x_inliers, y_inliers, inlier_mask = qmatch.iterative_linear_filter(
                x_spe, x_reference, n_sigma=3
            )            
            cost_matrix = None
            df = pd.DataFrame(
                {"spe": x_inliers, "reference": y_inliers, "distances": None}
            )
            return x_spe, x_reference, x_distance, cost_matrix, df
        elif _match_method == "dynamicp":
            x_spe, x_reference, cost_matrix, df = match_peaks_ready_wrapper(
                self.spe_pos_dict, self.ref,
                #alpha=1.0,
                #gamma=2.0,
                #skip_baseline=0.02,
                #tolerance=None,
                #normalize= self.spe_units == "pixel"
            )

            #df = pd.DataFrame(
            #    {"spe": x_spe, "reference": x_reference, "distances": None}
            #)
            return x_spe, x_reference, x_spe - x_reference, cost_matrix, df
        elif _match_method == "qargmin2d":
            x = np.array(list(self.spe_pos_dict.keys()))
            y = np.array(list(self.ref.keys()))
            if self.spe_units == "pixel":
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
            print(f"Outliers found {len(x_spe)-len(x_inliers)}")
            df = pd.DataFrame(
                {
                    "spe": x_inliers,
                    "reference": y_inliers,
                    "distances": x_inliers - y_inliers,
                }
            )
            return x_spe, x_reference, x_spe - x_reference, None, df        
        elif _match_method == "argmin2d":
            x = np.array(list(self.spe_pos_dict.keys()))
            y = np.array(list(self.ref.keys()))
            x_idx, y_idx = find_closest_pairs_idx(x, y)
            x_spe = x[x_idx]
            x_reference = y[y_idx]
            df = pd.DataFrame(
                {
                    "spe": x_spe,
                    "reference": x_reference,
                    "distances": x_spe - x_reference,
                }
            )
            return x_spe, x_reference, x_spe - x_reference, None, df
        elif _match_method == "assignment":  # https://en.wikipedia.org/wiki/Hungarian_algorithm
            try:
                x_spe, x_reference, x_distance, cost_matrix, df = match_peaks_optimized(
                    spe_pos_dict=self.spe_pos_dict,
                    ref=self.ref,
                    tolerance=100, relative=False, weight_intensity=0.9
                )
                return x_spe, x_reference, x_distance, cost_matrix, df
            except Exception as err:
                print(err)
                print("Reverting to monotonic match")
                x_spe, x_reference, x_distance,  df = match_peaks_monotonic(
                    spe_pos_dict=self.spe_pos_dict,
                    ref=self.ref,
                    tolerance=None, relative=False, weight_intensity=0.25
                )
                return x_spe, x_reference, x_distance, None, df
        else:  # self.match_method == "monotonic":
            try:
                x_spe, x_reference, x_distance,  df = match_peaks_monotonic_simple(
                    spe_pos_dict=self.spe_pos_dict,
                    ref=self.ref,
                    tolerance=100,
                    relative=False,
                    weight_intensity=.5
                )
                return x_spe, x_reference, x_distance, None, df
            except Exception as err:
                raise err

    def fit_peaks(self, find_kw, fit_peaks_kw, should_fit):
        spe_to_process = self.convert_units(self.spe, self.spe_units, self.ref_units)
        logger.debug("max x {} {}".format(max(spe_to_process.x), self.ref_units))

        peaks_df = None
        self.fit_res = None

        # instead of fit_peak_positions - we don't want movmin here
        # baseline removal might be done during preprocessing
        center_err_threshold = 0.5
        find_kw.update(dict(sharpening=None))
        cand = spe_to_process.find_peak_multipeak(**find_kw)
        # print(cand.get_ampl_pos_fwhm())

        self.fit_res = spe_to_process.fit_peak_multimodel(
            profile="Gaussian", candidates=cand, **fit_peaks_kw, no_fit=not should_fit,
            bound_centers_to_group=True
        )
        peaks_df = self.fit_res.to_dataframe_peaks()
        if should_fit:
            pos, amp = self.fit_res.center_amplitude(threshold=center_err_threshold)
            self.spe_pos_dict = dict(zip(pos, amp))
        else:
            self.spe_pos_dict = cand.get_pos_ampl_dict()
        return peaks_df


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
        print(df.shape, df.columns)
        
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


class CustomRBFInterpolator(RBFInterpolator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_dict(rbf_dict=None):
        if rbf_dict is None:
            rbf_dict = {}
        interpolator_loaded = CustomRBFInterpolator(
            rbf_dict["y"],
            rbf_dict["d"],
            epsilon=rbf_dict["epsilon"],
            smoothing=rbf_dict["smoothing"],
            kernel=rbf_dict["kernel"],
            neighbors=rbf_dict["neighbors"],
        )
        interpolator_loaded._coeffs = rbf_dict["coeffs"]
        interpolator_loaded._scale = rbf_dict["scale"]
        interpolator_loaded._shift = rbf_dict["shift"]
        return interpolator_loaded

    def to_dict(self):
        return {
            "y": self.y,
            "d": self.d,
            "d_dtype": self.d_dtype,
            "d_shape": self.d_shape,
            "epsilon": self.epsilon,
            "kernel": self.kernel,
            "neighbors": self.neighbors,
            "powers": self.powers,
            "smoothing": self.smoothing,
            "coeffs": self._coeffs,
            "scale": self._scale,
            "shift": self._shift,
        }

    def plot(self, ax):
        ax.scatter(
            self.y.reshape(-1),
            self.d.reshape(-1),
            marker="+",
            color="blue",
            label="Matched peaks",
        )

        x_range = np.linspace(self.y.min(), self.y.max(), 100)
        predicted_x = self(x_range.reshape(-1, 1))

        ax.plot(
            x_range, predicted_x, color="red", linestyle="-", label="Calibration curve"
        )
        ax.set_xlabel("Ne peaks, nm")
        ax.set_ylabel("Reference peaks, nm")
        ax.grid(which="both", linestyle="--", linewidth=0.5, color="gray")
        ax.legend()

    def __str__(self):
        return f"Calibration curve {len(self.y)} points) {self.kernel}"


class CustomPChipInterpolator(PchipInterpolator):
    def __init__(self, x, y,  **kwargs):
        super().__init__(x, y,  **kwargs)
        self.x = x  # Store x values
        self.y = y  # Store y values

    @staticmethod
    def from_dict(pchip_dict=None):
        if pchip_dict is None:
            pchip_dict = {}
        # Load the PCHIP interpolator from a dictionary
        interpolator_loaded = CustomPChipInterpolator(
            np.array(pchip_dict["x"]),  # Convert back to numpy arrays
            np.array(pchip_dict["y"]),
        )
        return interpolator_loaded

    def to_dict(self):
        # Save the current x and y data to a dictionary
        return {
            "x": self.x.tolist(),  # Convert numpy arrays to lists for JSON serialization
            "y": self.y.tolist(),
        }

    def save_coefficients(self, filename):
        """Save the x and y coefficients to a JSON file."""
        coeffs = self.to_dict()
        with open(filename, "w") as f:
            json.dump(coeffs, f)

    @classmethod
    def load_coefficients(cls, filename):
        """Load the coefficients from a JSON file."""
        with open(filename, "r") as f:
            coeffs = json.load(f)
        return cls.from_dict(coeffs)

    def plot(self, ax):
        """Plot the interpolation curve and the original points."""
        ax.scatter(self.x, self.y, marker="+", color="blue", label="Ne original data")

        x_range = np.linspace(self.x.min(), self.x.max(), 100)
        predicted_y = self(x_range)

        ax.plot(
            x_range, predicted_y, color="red", linestyle="-", label="Ne calibration curve"
        )
        ax.set_xlabel("Ne peak original/nm")
        ax.set_ylabel("Ne peak NIST/nm")
        ax.grid(which="both", linestyle="--", linewidth=0.5, color="gray")
        ax.legend()

    def __str__(self):
        return f"Calibration curve {len(self.y)} points) (PchipInterpolator)"


class CustomCubicSplineInterpolator(CubicSpline):
    def __init__(self, x, y,  **kwargs):
        super().__init__(x, y, **kwargs)
        self.x = x
        self.y = y

    @staticmethod
    def from_dict(spline_dict=None):
        if spline_dict is None:
            spline_dict = {}
        interpolator_loaded = CustomCubicSplineInterpolator(
            spline_dict["x"],
            spline_dict["y"],
            bc_type=spline_dict.get("bc_type", "clamped"),
            extrapolate=spline_dict.get("extrapolate", True),
        )
        return interpolator_loaded

    def to_dict(self):
        return {
            "x": self.x,
            "y": self.y,
            "bc_type": self.bc_type,
            "extrapolate": self.extrapolate,
        }

    def plot(self, ax):
        ax.scatter(self.x, self.y, marker="+", color="blue", label="Data points")
        x_range = np.linspace(self.x.min(), self.x.max(), 100)
        predicted_y = self(x_range)

        ax.plot(
            x_range, predicted_y, color="red", linestyle="-", label="Cubic spline curve"
        )
        ax.set_xlabel("X values")
        ax.set_ylabel("Y values")
        ax.grid(which="both", linestyle="--", linewidth=0.5, color="gray")
        ax.legend()

    def __str__(self):
        return f"Cubic Spline Interpolator with {len(self.x)} points."


class CustomPolyInterpolator:
    def __init__(self, x, y, max_degree=3):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.max_degree = max_degree

        # enforce monotonic ordering in x
        order = np.argsort(self.x)
        self.x = self.x[order]
        self.y = self.y[order]

        # normalize x for numerical stability
        self.x_min = self.x.min()
        self.x_max = self.x.max()

        u = (self.x - self.x_min) / (self.x_max - self.x_min)

        # choose degree ≤ max_degree by max residual
        best_err = np.inf
        best_coeff = None
        best_deg = None

        for deg in range(2, max_degree + 1):
            coeff = np.polyfit(u, self.y, deg)
            pred = np.polyval(coeff, u)
            err = np.max(np.abs(pred - self.y))

            if err < best_err:
                best_err = err
                best_coeff = coeff
                best_deg = deg

        self.coef = best_coeff
        self.degree = best_deg
        self.fit_error = best_err

    # --------------------------------------------------------
    # Callable interface (like PchipInterpolator)
    # --------------------------------------------------------
    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        u = (x - self.x_min) / (self.x_max - self.x_min)
        return np.polyval(self.coef, u)

    # --------------------------------------------------------
    # Serialization
    # --------------------------------------------------------
    @staticmethod
    def from_dict(poly_dict=None):
        if poly_dict is None:
            poly_dict = {}

        obj = CustomPolyInterpolator(
            np.array(poly_dict["x"]),
            np.array(poly_dict["y"]),
            max_degree=poly_dict.get("degree", 3),
        )

        # overwrite learned parameters
        obj.coef = np.array(poly_dict["coef"])
        obj.degree = poly_dict["degree"]
        obj.x_min = poly_dict["x_min"]
        obj.x_max = poly_dict["x_max"]
        obj.fit_error = poly_dict.get("fit_error", None)

        return obj

    def to_dict(self):
        return {
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "coef": self.coef.tolist(),
            "degree": int(self.degree),
            "x_min": float(self.x_min),
            "x_max": float(self.x_max),
            "fit_error": None if self.fit_error is None else float(self.fit_error),
        }

    def save_coefficients(self, filename):
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load_coefficients(cls, filename):
        with open(filename, "r") as f:
            coeffs = json.load(f)
        return cls.from_dict(coeffs)

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    def plot(self, ax):
        ax.scatter(self.x, self.y, marker="+", color="blue", label="Original data")

        x_range = np.linspace(self.x.min(), self.x.max(), 500)
        y_pred = self(x_range)

        ax.plot(
            x_range,
            y_pred,
            color="red",
            linestyle="-",
            label=f"Polynomial degree {self.degree}",
        )
        ax.set_xlabel("Original")
        ax.set_ylabel("Reference")
        ax.grid(which="both", linestyle="--", linewidth=0.5, color="gray")
        ax.legend()

    def __str__(self):
        return (
            f"Calibration curve {len(self.y)} points "
            f"(Polynomial degree {self.degree})"
        )
