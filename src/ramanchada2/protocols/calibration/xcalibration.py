import json
import logging
from typing import Dict, Literal

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator, RBFInterpolator

from ramanchada2.misc.utils import find_closest_pairs_idx

from ramanchada2.misc.utils.matchsets import (
    cost_function_position,
    match_peaks,
    match_peaks_cluster,
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
        spe_units: Literal["cm-1", "nm"] = "cm-1",
        ref_units: Literal["cm-1", "nm"] = "nm",
        sample="Neon",
        match_method: Literal["cluster", "argmin2d", "assignment"] = "cluster",
        interpolator_method: Literal["rbf", "pchip", "cubic_spline"] = "pchip",
        extrapolate=True,
    ):
        super(XCalibrationComponent, self).__init__(
            laser_wl, spe, spe_units, ref, ref_units, sample
        )
        self.spe_pos_dict = None
        self.match_method = match_method
        self.cost_function = cost_function_position
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
        spe_units: Literal["cm-1", "nm"] = "cm-1",
        convert_back=False,
    ):
        logger.debug(
            "convert spe_units {} --> model units {}".format(
                spe_units, self.model_units
            )
        )
        new_spe = self.convert_units(old_spe, spe_units, self.model_units)

        if self.model is None:
            return new_spe
        elif self.enabled:
            if isinstance(self.model, float):
                new_spe.x = new_spe.x + self.model
            else:
                if isinstance(self.model, CustomPChipInterpolator):
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
        ax.stem(
            self.spe_pos_dict.keys(),
            self.spe_pos_dict.values(),
            linefmt="b-",
            basefmt=" ",
            label="{} peaks".format(self.sample),
        )
        ax.twinx().stem(
            self.ref.keys(),
            self.ref.values(),
            linefmt="r-",
            basefmt=" ",
            label="Reference {}".format(self.sample),
        )

        if self.ref_units == "cm-1":
            _units = r"$\mathrm{{[{self.ref_units}]}}$"
        else:
            _units = self.ref_units
        ax.set_xlabel(_units)
        ax.legend()

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
        logger.debug(
            "[{}]: convert spe_units {} to ref_units {}".format(
                self.name, self.spe_units, self.ref_units
            )
        )
        peaks_df = self.fit_peaks(find_kw, fit_peaks_kw, should_fit)
        x_spe, x_reference, x_distance, cost_matrix, df = self.match_peaks(
            threshold_max_distance=8, return_df=True
        )

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
                if self.interpolator_method == "pchip":
                    # kwargs = {"kernel": "thin_plate_spline"}
                    interp = CustomPChipInterpolator(x_spe, x_reference)
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
        if self.match_method == "cluster":
            x_spe, x_reference, x_distance, _ = match_peaks_cluster(
                self.spe_pos_dict, self.ref
            )
            cost_matrix = None
            df = pd.DataFrame(
                {"spe": x_spe, "reference": x_reference, "distances": x_distance}
            )
            return x_spe, x_reference, x_distance, cost_matrix, df
        elif self.match_method == "argmin2d":
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
        else:
            try:
                x_spe, x_reference, x_distance, cost_matrix, df = match_peaks(
                    self.spe_pos_dict,
                    self.ref,
                    threshold_max_distance=threshold_max_distance,
                    df=return_df,
                    cost_func=self.cost_function,
                )
                return x_spe, x_reference, x_distance, cost_matrix, df
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
        df = df.sort_values(by="height", ascending=False)
        # df = df.sort_values(by='amplitude', ascending=False)
        if df.empty:
            raise Exception("No peaks found")
        else:
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
