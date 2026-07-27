import json
import logging
import os.path
from typing import Optional, Tuple

import numpy as np
from pydantic import BaseModel, ValidationError

from ramanchada2.misc.plottable import Plottable
from ramanchada2.spectrum import Spectrum
from .calibration_component import CalibrationComponent
from .xcalibration import CustomPChipInterpolator

logger = logging.getLogger(__name__)


class YCalibrationCertificate(BaseModel, Plottable):
    """
    Class for intensity calibration certificates

    Usage:

        1. Use for specific SRM
        >>> cert = YCalibrationCertificate(
        ...             id="NIST785_SRM2241",
        ...             description="optical glass",
        ...             url="https://tsapps.nist.gov/srmext/certificates/2241.pdf",
        ...             wavelength=785,
        ...             params="A0 = 9.71937e-02, A1 = 2.28325e-04, A2 = -5.86762e-08, A3 = 2.16023e-10, A4 = -9.77171e-14, A5 = 1.15596e-17",
        ...             equation="A0 + A1 * x + A2 * x**2 + A3 * x**3 + A4 * x**4 + A5 * x**5",
        ...             temperature_c=(20, 25),
        ...             raman_shift=(200, 3500)
        ...         )
        ...
        >>> cert.plot()

    """  # noqa: E501

    id: str
    description: Optional[str] = None
    url: Optional[str] = None
    wavelength: int
    params: str
    equation: str
    temperature_c: Optional[Tuple[int, int]] = None
    raman_shift: Optional[Tuple[int, int]] = None

    @property
    def response_function(self):
        local_vars = {}
        for param in self.params.split(","):
            key, value = param.split("=")
            key = key.strip()
            value = value.strip()
            local_vars[key] = eval(value)

        def evaluate_expression(x_value):
            local_vars["x"] = x_value
            return eval(self.equation, {"np": np}, local_vars)

        return evaluate_expression

    def Y(self, x_value):
        return self.response_function(x_value)

    def _parsed_params(self):
        names, values = [], []
        for param in self.params.split(","):
            key, value = param.split("=")
            names.append(key.strip())
            values.append(float(eval(value.strip())))
        return names, values

    @property
    def param_names(self):
        return self._parsed_params()[0]

    @property
    def param_values(self):
        return self._parsed_params()[1]

    @property
    def polynomial_order(self):
        """Degree n when the equation is a polynomial with params A0..An, else None."""
        names = self.param_names
        if names == [f"A{i}" for i in range(len(names))]:
            return len(names) - 1
        return None

    def model_function(self):
        """``f(x, *param_values)`` evaluating the certificate equation.

        This is the model the measured reference is fitted against (seeded by the
        certificate's own ``param_values``), so the fit takes the certificate's
        functional form.
        """
        names = self.param_names
        equation = self.equation

        def f(x, *values):
            local_vars = dict(zip(names, values))
            local_vars["x"] = np.asarray(x, dtype=float)
            return eval(equation, {"np": np}, local_vars)

        return f

    def trim_axes(self, spe):
        return spe.trim_axes(method="x-axis", boundaries=self.raman_shift)

    def _plot(self, ax, **kwargs):
        if self.raman_shift is None:
            x = np.linspace(100, 4000)
        else:
            x = np.linspace(self.raman_shift[0], self.raman_shift[1])
        kwargs.pop("label", None)
        ax.plot(
            x, self.Y(x), label="{} ({}nm)".format(self.id, self.wavelength), **kwargs
        )
        _units = "cm^{-1}"
        ax.set_xlabel(rf"Wavenumber/$\mathrm{{{_units}}}$")
        ax.set_ylabel("Raman intensity/Arbitr.Units")

    @staticmethod
    def load(wavelength=785, key="NIST785_SRM2241"):
        return CertificatesDict().get(wavelength, key)


class CertificatesDict:
    """
    Class for loading y calibration certificates

    Usage:
       Load single certificate
       >>> cert = CertificatesDict.load(wavelength="785", key="NIST785_SRM2241")
       >>> cert.plot()

       Load all certificates for wavelength. Iterate :

        >>> certificates = CertificatesDict()
        ... plt.figure()
        ... ax=None
        ... certs = certificates.get_certificates(wavelength=532)
        ... ax = certs[cert].plot(ax=ax)
        >>> plt.show()
    """

    def __init__(self):
        self.load_certificates(
            os.path.join(os.path.dirname(__file__), "config_certs.json")
        )

    def load_certificates(self, file_path):

        with open(file_path, "r") as f:
            certificates_data = json.load(f)
            certificates = {}
            self.laser_wl = []
            for wavelength, certificates_dict in certificates_data.items():
                certificates[wavelength] = {}
                self.laser_wl.append(wavelength)
                for certificate_id, certificate_data in certificates_dict.items():
                    certificate_data["wavelength"] = int(wavelength)
                    certificate_data["id"] = certificate_id
                    try:
                        certificate = YCalibrationCertificate.model_validate(
                            certificate_data
                        )
                        certificates[wavelength][certificate_id] = certificate
                    except ValidationError as e:
                        logger.warning(f"Validation error for certificate {certificate_id}: {e}")
            self.config_certs = certificates

    def get_laser_wl(self):
        return self.laser_wl

    def get_certificates(self, wavelength=785):
        return self.config_certs[str(wavelength)]

    def get(self, wavelength=532, key="NIST532_SRM2242a"):
        return self.config_certs[str(wavelength)][key]

    @staticmethod
    def load(wavelength=785, key="NIST785_SRM2241"):
        return CertificatesDict().get(wavelength, key)


class YCalibrationComponent(CalibrationComponent):
    """
    Class for relative intensity calibration. Uses response functions loaded in
    ResponseFunctionEvaluator. Functions are defined in json file.

    Usage:

        >>> laser_wl = 785
        >>> ycert = YCalibrationCertificate.load(wavelength=785, key="SRM2241")
        >>> ycal = YCalibrationComponent(laser_wl, reference_spe_xcalibrated=spe_srm,certificate=ycert)
        >>> fig, ax = plt.subplots(1, 1, figsize=(15,4))
        >>> spe_srm.plot(ax=ax)
        >>> spe_to_correct.plot(ax=ax)
        >>> spe_ycalibrated = ycal.process(spe_to_correct)
        >>> spe_ycalibrated.plot(label="y-calibrated",color="green",ax=ax.twinx())
    """

    def __init__(
        self,
        laser_wl,
        reference_spe_xcalibrated,
        certificate: YCalibrationCertificate,
        model_method: str = "certificate",
        fit_order=None,
        normalize: bool = True,
    ):
        super(YCalibrationComponent, self).__init__(
            laser_wl,
            spe=reference_spe_xcalibrated,
            spe_units=None,
            ref=certificate,
            ref_units=None,
        )
        self.laser_wl = laser_wl
        self.spe = reference_spe_xcalibrated
        self.ref = certificate
        self.name = "Y calibration"
        # How the measured reference is represented. "certificate" fits it with
        # the certificate's own functional form (analytic, noise-free); "pchip"
        # is the legacy raw interpolation of the measured points.
        self.model_method = model_method
        self.fit_order = fit_order
        self.normalize = normalize
        self.model = self._build_model()
        self.model_units = "cm-1"

    def _trimmed_reference(self):
        # certificate.raman_shift is optional; without a certified range there is
        # nothing to trim to, so use the full measured reference spectrum instead
        # of letting trim_axes(boundaries=None) raise
        if self.ref.raman_shift is None:
            return self.spe
        return self.spe.trim_axes(method="x-axis", boundaries=self.ref.raman_shift)

    def _build_model(self):
        tmp = self._trimmed_reference()
        if self.model_method == "certificate":
            fitted = self._fit_reference(tmp)
            if fitted is not None:
                return fitted
        return CustomPChipInterpolator(tmp.x, tmp.y)

    def _fit_reference(self, tmp):
        """Fit the measured reference with the certificate's own functional form.

        Polynomial certificates -> a polynomial of the same order (linear least
        squares); other forms (the SRM log-Gaussian) -> a nonlinear fit of the
        certificate equation, seeded by the certificate's own parameters. The
        analytic fit denoises the reference without smoothing. Returns a
        ``ParametricModel``, or ``None`` to fall back to the raw PCHIP.
        """
        from .interpolators import ParametricModel
        from scipy.optimize import curve_fit

        x = np.asarray(tmp.x, dtype=float)
        y = np.asarray(tmp.y, dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if len(x) < 6:
            logger.warning("Y calibration: too few reference points to fit; using PCHIP")
            return None
        if self.normalize:
            ymax = float(np.max(np.abs(y)))
            if ymax > 0:
                y = y / ymax

        # fit_order forces a polynomial of that order; otherwise the certificate's
        # own order (polynomial) or None (fit the certificate's equation directly)
        order = self.fit_order if self.fit_order is not None else self.ref.polynomial_order
        try:
            if order is not None:
                order = int(order)
                coef_desc = np.polyfit(x, y, order)
                names = [f"A{i}" for i in range(order + 1)]
                terms = ["A0"] + [f"A{i}*x**{i}" for i in range(1, order + 1)]
                model = ParametricModel(
                    " + ".join(terms), names, list(coef_desc[::-1])
                )
            else:
                names = self.ref.param_names
                popt, _ = curve_fit(
                    self.ref.model_function(), x, y, p0=self.ref.param_values,
                    maxfev=20000,
                )
                model = ParametricModel(self.ref.equation, names, list(popt))
            probe = np.asarray(model(x), dtype=float)
            if not np.all(np.isfinite(probe)):
                logger.warning("Y calibration: fitted reference not finite; using PCHIP")
                return None
            return model
        except Exception as err:  # noqa: BLE001 - degrade to PCHIP, never crash
            logger.warning(f"Y calibration: reference fit failed ({err}); using PCHIP")
            return None

    def derive_model(self, find_kw=None, fit_peaks_kw=None, should_fit=True, name=None):
        self.model = self._build_model()

    def safe_divide(self, spe_to_correct, spe_reference_resampled):
        numerator = spe_to_correct.y
        # numerator_noise = spe_to_correct.y_noise

        scaling_denominator = spe_reference_resampled.y / self.ref.Y(
            spe_reference_resampled.x
        )
        # print(np.median(scaling_denominator), np.mean(scaling_denominator), np.std(scaling_denominator))

        # denominator_noise = spe_reference_resampled.y_noise
        denominator = spe_reference_resampled.y
        # Create a mask for dividing only where value is above noise !
        # mask = (abs(scaling_denominator) > 0) & (kind_of_snr > 0.9)
        # mask =  (abs(denominator) > abs(denominator_noise)) &
        mask = (abs(scaling_denominator) > 0) & (numerator > 0) & (denominator > 0)
        # & (abs(numerator) > numerator_noise) & (abs(scaling_denominator) > 0)
        # & (abs(denominator-numerator) > min(denominator_noise,numerator_noise))
        result = np.zeros_like(numerator)
        # Perform division where mask is true
        result[mask] = numerator[mask] / scaling_denominator[mask]
        return result

    def safe_mask(self, spe_to_correct, spe_reference_resampled):
        ref_noise = spe_reference_resampled.y_noise_MAD()
        mask = (spe_reference_resampled.y >= 0) & (
            abs(spe_reference_resampled.y) > ref_noise
        )
        # Outside the certified range, self.model (CustomPChipInterpolator) falls
        # back to a unit-slope extrapolation meant for x-calibration curves, not
        # measured reference intensity -- exclude those points rather than treat
        # the extrapolated value as a real intensity factor.
        if self.ref.raman_shift is not None:
            lo, hi = self.ref.raman_shift
            x = spe_reference_resampled.x
            mask = mask & (x >= lo) & (x <= hi)
        return mask

    def safe_factor(self, spe_to_correct, spe_reference_resampled):
        numerator = spe_to_correct.y
        # numerator_noise = spe_to_correct.y_noise

        Y = self.ref.Y(spe_reference_resampled.x)
        mask = self.safe_mask(spe_to_correct, spe_reference_resampled)
        if mask is None:
            scaling_factor = Y / spe_reference_resampled.y
        else:
            scaling_factor = np.zeros_like(spe_reference_resampled.y)
            scaling_factor[mask] = Y[mask] / spe_reference_resampled.y[mask]

        result = numerator * scaling_factor
        return result

    def process(self, old_spe: Spectrum, spe_units="nm", convert_back=False):
        # resample using probability density function
        _tmp = self.model(old_spe.x)
        spe_reference_resampled = Spectrum(old_spe.x, _tmp)
        # new_spe = Spectrum(old_spe.x,self.safe_divide(old_spe,spe_reference_resampled))
        new_spe = Spectrum(
            old_spe.x, self.safe_factor(old_spe, spe_reference_resampled)
        )
        return new_spe

    def _plot(self, ax, **kwargs):
        if self.ref is not None:
            self.ref.plot(ax, **kwargs)

    def to_dict(self):
        """Portable (JSON-clean) representation: certificate + measured-reference model."""
        from .interpolators import interpolator_to_tagged_dict
        return {
            "type": "YCalibrationComponent",
            "name": self.name,
            "enabled": bool(self.enabled),
            "laser_wl": int(self.laser_wl) if self.laser_wl is not None else None,
            "model_units": self.model_units,
            "model_method": getattr(self, "model_method", "certificate"),
            "certificate": self.ref.model_dump(),
            "model": interpolator_to_tagged_dict(self.model),
        }

    @classmethod
    def from_dict(cls, d):
        from .interpolators import interpolator_from_tagged_dict
        obj = object.__new__(cls)  # __init__ requires the reference spectrum
        obj.laser_wl = d["laser_wl"]
        obj.spe = None
        obj.spe_units = None
        obj.ref = YCalibrationCertificate.model_validate(d["certificate"])
        obj.ref_units = None
        obj.name = d.get("name", "Y calibration")
        obj.model_method = d.get("model_method", "certificate")
        obj.fit_order = None
        obj.normalize = True
        obj.model = interpolator_from_tagged_dict(d["model"])
        obj.model_units = d.get("model_units", "cm-1")
        obj.peaks = None
        obj.sample = None
        obj.enabled = d.get("enabled", True)
        obj.fit_res = None
        return obj
