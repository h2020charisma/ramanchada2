from typing import Optional, Tuple
from ramanchada2.misc.plottable import Plottable
from pydantic import BaseModel, ValidationError
import json
import os.path
import numpy as np


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
        ax.set_xlabel(rf"Raman shift $\mathrm{{[{_units}]}}$")
        ax.set_ylabel("Intensity")

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
