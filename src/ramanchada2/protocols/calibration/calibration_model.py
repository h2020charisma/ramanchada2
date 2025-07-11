import pickle
import warnings

from typing import Dict, Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import ramanchada2.misc.constants as rc2const
from ramanchada2.misc.plottable import Plottable
from ramanchada2.spectrum import Spectrum
from .calibration_component import ProcessingModel
from .xcalibration import LazerZeroingComponent, XCalibrationComponent


class CalibrationModel(ProcessingModel, Plottable):
    nonmonotonic: Literal["ignore", "nan", "error", "drop"] = "nan"

    """
    A class representing a calibration model for Raman spectrum.
    """

    def __init__(self, laser_wl: int):
        """
        Initializes a CalibrationModel instance.

        Args:
            laser_wl:
                The wavelength of the laser used for calibration.

        Example:
        ```python
        # Create an instance of CalibrationModel
        import ramanchada2 as rc2
        import ramanchada2.misc.constants as rc2const
        from ramanchada2.protocols.calibration import CalibrationModel
        laser_wl=785
        calmodel = CalibrationModel.calibration_model_factory(
            laser_wl,
            spe_neon,
            spe_sil,
            neon_wl=rc2const.NEON_WL[laser_wl],
            find_kw={"wlen": 200, "width": 1},
            fit_peaks_kw={},
            should_fit=False,
        )
        # Store (optional)
        calmodel.save(modelfile)
        # Load (optional)
        calmodel = CalibrationModel.from_file(modelfile)
        # Apply to new spectrum
        calmodel.apply_calibration_x(
            spe_to_calibrate,
            spe_units="cm-1"
            )
        ```
        """
        super(ProcessingModel, self).__init__()
        super(Plottable, self).__init__()
        self.set_laser_wavelength(laser_wl)
        self.prominence_coeff = 3

    def set_laser_wavelength(self, laser_wl):
        """
        Sets the wavelength of the laser used for calibration.
        """
        self.clear()
        self.laser_wl = laser_wl

    def clear(self):
        """
        Clears the calibration model.
        """
        self.laser_wl = None
        self.components = []

    def save(self, filename):
        """
        Saves the calibration model to a file.
        """
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def from_file(filename):
        """
        Loads a calibration model from a file.
        """
        with open(filename, "rb") as file:
            return pickle.load(file)

    def derive_model_x(
        self,
        spe_neon: Spectrum,
        spe_neon_units: str,
        ref_neon: Dict,
        ref_neon_units: str,
        spe_sil: Spectrum,
        spe_sil_units="cm-1",
        ref_sil={520.45: 1},
        ref_sil_units="cm-1",
        find_kw={"wlen": 200, "width": 1},
        fit_kw={},
        should_fit=False,
        match_method: Literal["cluster", "argmin2d", "assignment"] = "cluster",
        interpolator_method: Literal["rbf", "pchip", "cubic_spline"] = "rbf",
        extrapolate=True,
    ):
        """
        Derives x-calibration models using Neon and Silicon spectra.
        """
        self.components.clear()
        if ref_neon_units is None:
            ref_neon_units = "nm"
        if spe_neon_units is None:
            spe_neon_units = "cm-1"
        find_kw["prominence"] = spe_neon.y_noise_MAD() * self.prominence_coeff
        model_neon = self._derive_model_curve(
            spe_neon,
            rc2const.NEON_WL[self.laser_wl] if ref_neon is None else ref_neon,
            spe_units=spe_neon_units,
            ref_units=ref_neon_units,
            find_kw=find_kw,
            fit_peaks_kw=fit_kw,
            should_fit=should_fit,
            name="Neon calibration",
            match_method=match_method,
            interpolator_method=interpolator_method,
            extrapolate=extrapolate,
        )
        model_neon.nonmonotonic = self.nonmonotonic
        spe_sil_ne_calib = model_neon.process(
            spe_sil, spe_units=spe_sil_units, convert_back=False
        )

        find_kw["prominence"] = spe_sil_ne_calib.y_noise_MAD() * self.prominence_coeff
        model_si = self._derive_model_zero(
            spe_sil_ne_calib,
            ref=ref_sil,
            spe_units=model_neon.model_units,
            ref_units=ref_sil_units,
            find_kw=find_kw,
            fit_peaks_kw=fit_kw,
            should_fit=True,
            name="Si laser zeroing",
        )
        return (model_neon, model_si)

    def _derive_model_curve(
        self,
        spe: Spectrum,
        ref=Dict[float, float],
        spe_units="cm-1",
        ref_units="nm",
        find_kw=None,
        fit_peaks_kw=None,
        should_fit=False,
        name="X calibration",
        match_method: Literal["cluster", "argmin2d", "assignment"] = "cluster",
        interpolator_method: Literal["rbf", "pchip", "cubic_spline"] = "rbf",
        extrapolate=True,
    ):
        if find_kw is None:
            find_kw = {}
        if fit_peaks_kw is None:
            fit_peaks_kw = {}
        reference_peaks = rc2const.NEON_WL[self.laser_wl] if ref is None else ref
        calibration_x = XCalibrationComponent(
            self.laser_wl,
            spe=spe,
            spe_units=spe_units,
            ref=reference_peaks,
            ref_units=ref_units,
            match_method=match_method,
            interpolator_method=interpolator_method,
            extrapolate=extrapolate,
        )
        calibration_x.nonmonotonic = self.nonmonotonic
        calibration_x.derive_model(
            find_kw=find_kw, fit_peaks_kw=fit_peaks_kw, should_fit=should_fit, name=name
        )
        self.components.append(calibration_x)
        return calibration_x

    def derive_model_curve(
        self,
        spe: Spectrum,
        ref=None,
        spe_units="cm-1",
        ref_units="nm",
        find_kw={},
        fit_peaks_kw={},
        should_fit=False,
        name="X calibration",
        match_method: Literal["cluster", "argmin2d", "assignment"] = "cluster",
        interpolator_method: Literal["rbf", "pchip", "cubic_spline"] = "rbf",
        extrapolate=True,
    ):
        warnings.warn(
            message="Do not use directly. Use derive_model_x instead.",
            category=DeprecationWarning,
        )
        return self._derive_model_curve(
            spe=spe,
            ref=ref,
            spe_units=spe_units,
            ref_units=ref_units,
            find_kw=find_kw,
            fit_peaks_kw=fit_peaks_kw,
            should_fit=should_fit,
            name=name,
            match_method=match_method,
            interpolator_method=interpolator_method,
            extrapolate=extrapolate,
        )

    def _derive_model_zero(
        self,
        spe: Spectrum,
        ref=None,
        spe_units="nm",
        ref_units="cm-1",
        find_kw=None,
        fit_peaks_kw=None,
        should_fit=False,
        name="Laser zeroing",
        profile="Pearson4",
    ):
        if ref is None:
            ref = {520.45: 1}
        if find_kw is None:
            find_kw = {}
        if fit_peaks_kw is None:
            fit_peaks_kw = {}
        calibration_shift = LazerZeroingComponent(
            self.laser_wl, spe, spe_units, ref, ref_units
        )
        calibration_shift.profile = profile
        calibration_shift.derive_model(
            find_kw=find_kw, fit_peaks_kw=fit_peaks_kw, should_fit=should_fit, name=name
        )
        _laser_zeroing_component = None
        for i, item in enumerate(self.components):
            if isinstance(item, LazerZeroingComponent):
                self.components[i] = calibration_shift
                _laser_zeroing_component = self.components[i]
        if (
            _laser_zeroing_component is None
        ):  # LaserZeroing component should present only once
            self.components.append(calibration_shift)
        return calibration_shift

    def derive_model_zero(
        self,
        spe: Spectrum,
        ref={520.45: 1},
        spe_units="nm",
        ref_units="cm-1",
        find_kw=None,
        fit_peaks_kw=None,
        should_fit=False,
        name="X Shift",
        profile="Pearson4",
    ):
        if find_kw is None:
            find_kw = {}
        if fit_peaks_kw is None:
            fit_peaks_kw = {}
        warnings.warn(
            message="Do not use directly. Use derive_model_x instead.",
            category=DeprecationWarning,
        )
        return self._derive_model_zero(
            spe=spe,
            ref=ref,
            spe_units=spe_units,
            ref_units=ref_units,
            find_kw=find_kw,
            fit_peaks_kw=fit_peaks_kw,
            should_fit=should_fit,
            name=name,
            profile=profile,
        )

    def apply_calibration_x(self, old_spe: Spectrum, spe_units="cm-1"):
        # neon calibration converts to nm
        # silicon calibration takes nm and converts back to cm-1 using laser zeroing
        new_spe = old_spe
        model_units = spe_units
        for model in self.components:
            # TODO: tbd find out if to convert units
            if model.enabled:
                new_spe = model.process(new_spe, model_units, convert_back=False)
                model_units = model.model_units
        return new_spe

    def plot(self, ax=None, label=" ", **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        self._plot(ax, **kwargs)
        return ax

    def _plot(self, ax, **kwargs):
        for index, model in enumerate(self.components):
            model._plot(ax, **kwargs)
            break

    @staticmethod
    def calibration_model_factory(
        laser_wl,
        spe_neon: Spectrum,
        spe_sil: Spectrum,
        neon_wl=None,
        find_kw=None,
        fit_peaks_kw=None,
        should_fit=False,
        prominence_coeff=3,
        si_profile="Pearson4",
        match_method: Literal["cluster", "argmin2d", "assignment"] = "argmin2d",
        interpolator_method: Literal["rbf", "pchip", "cubic_spline"] = "pchip",
        extrapolate=True,
    ):
        if neon_wl is None:
            neon_wl = rc2const.NEON_WL[laser_wl]
        if find_kw is None:
            find_kw = {"wlen": 100, "width": 1}
        if fit_peaks_kw is None:
            fit_peaks_kw = {}
        calmodel = CalibrationModel(laser_wl)
        calmodel.prominence_coeff = prominence_coeff
        find_kw["prominence"] = spe_neon.y_noise_MAD() * calmodel.prominence_coeff

        model_neon = calmodel._derive_model_curve(
            spe=spe_neon,
            ref=neon_wl,
            spe_units="cm-1",
            ref_units="nm",
            find_kw=find_kw,
            fit_peaks_kw=fit_peaks_kw,
            should_fit=should_fit,
            name="Neon calibration",
            match_method=match_method,
            interpolator_method=interpolator_method,
            extrapolate=extrapolate,
        )
        spe_sil_ne_calib = model_neon.process(
            spe_sil, spe_units="cm-1", convert_back=False
        )
        find_kw["prominence"] = (
            spe_sil_ne_calib.y_noise_MAD() * calmodel.prominence_coeff
        )
        calmodel.derive_model_zero(
            spe=spe_sil_ne_calib,
            ref={520.45: 1},
            spe_units=model_neon.model_units,
            ref_units="cm-1",
            find_kw=find_kw,
            fit_peaks_kw=fit_peaks_kw,
            should_fit=True,
            name="Si calibration",
            profile=si_profile,
        )
        return calmodel
