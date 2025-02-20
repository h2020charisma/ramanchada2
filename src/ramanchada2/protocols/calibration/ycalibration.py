import json
import os.path
from typing import Optional, Tuple

import numpy as np
from pydantic import BaseModel, ValidationError

from ramanchada2.misc.plottable import Plottable
from ramanchada2.spectrum import Spectrum
from .calibration_component import CalibrationComponent


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
    description: Optional[str]
    url: Optional[str]
    wavelength: int
    params: str
    equation: str
    temperature_c: Optional[Tuple[int, int]]
    raman_shift: Optional[Tuple[int, int]]

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
                        certificate = YCalibrationCertificate.model_construct(
                            **certificate_data
                        )
                        certificates[wavelength][certificate_id] = certificate
                    except ValidationError as e:
                        print(f"Validation error for certificate {certificate_id}: {e}")
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
        self, laser_wl, reference_spe_xcalibrated, certificate: YCalibrationCertificate
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
        self.model = self.spe.spe_distribution(trim_range=certificate.raman_shift)
        self.model_units = "cm-1"

    def derive_model(self, find_kw=None, fit_peaks_kw=None, should_fit=True, name=None):
        # measured reference spectrum as distribution, so we can resample
        self.model = self.spe.spe_distribution(trim_range=self.ref.raman_shift)

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
        return (spe_reference_resampled.y >= 0) & (
            abs(spe_reference_resampled.y) > ref_noise
        )

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
        _tmp = self.model.pdf(old_spe.x)
        _tmp = (
            _tmp * max(self.spe.y) / max(_tmp)
        )  # pdf sampling is normalized to area unity, scaling back
        spe_reference_resampled = Spectrum(old_spe.x, _tmp)
        # new_spe = Spectrum(old_spe.x,self.safe_divide(old_spe,spe_reference_resampled))
        new_spe = Spectrum(
            old_spe.x, self.safe_factor(old_spe, spe_reference_resampled)
        )
        return new_spe

    def _plot(self, ax, **kwargs):
        if self.ref is not None:
            self.ref.plot(ax, **kwargs)
