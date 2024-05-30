import matplotlib.pyplot as plt
import numpy as np
import pickle
import ramanchada2.misc.constants as rc2const
from ..misc import utils as rc2utils
from ..spectrum import Spectrum
from matplotlib.axes import Axes
from ramanchada2.misc.plottable import Plottable
from scipy import interpolate
import logging
import json
import os
from typing import Tuple, Optional
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class ProcessingModel:

    def __init__(self):
        pass


class CalibrationComponent(Plottable):

    def __init__(self, laser_wl, spe, spe_units, ref, ref_units, sample=None):
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
        self.sample = sample
        self.enabled = True

    def set_model(self, model, model_units, peaks, name=None):
        self.model = model
        self.model_units = model_units
        self.peaks = peaks
        self.name = "calibration component" if name is None else name

    def __str__(self):
        return (f"{self.name} spe ({self.spe_units}) reference ({self.ref_units}) "
                f"model ({self.model_units}) {self.model}")

    def convert_units(self, old_spe, spe_unit="cm-1", newspe_unit="nm", laser_wl=None):
        if laser_wl is None:
            laser_wl = self.laser_wl
        logger.debug("convert laser_wl {} {} --> {}".format(laser_wl, spe_unit, newspe_unit))
        if spe_unit != newspe_unit:
            new_spe = old_spe.__copy__()
            if spe_unit == "nm":
                new_spe = old_spe.abs_nm_to_shift_cm_1_filter(laser_wave_length_nm=laser_wl)
            elif spe_unit == "cm-1":
                new_spe = old_spe.shift_cm_1_to_abs_nm_filter(laser_wave_length_nm=laser_wl)
            else:
                raise Exception("Unsupported conversion {} to {}", spe_unit, newspe_unit)
        else:
            new_spe = old_spe.__copy__()
        #    new_spe = old_spe.__copy__()
        return new_spe

    def process(self, old_spe: Spectrum, spe_units="cm-1", convert_back=False):
        raise NotImplementedError(self)

    def derive_model(self, find_kw={}, fit_peaks_kw={}, should_fit=False, name=None):
        raise NotImplementedError(self)

    def plot(self, ax=None, label=' ', **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots(3, 1, figsize=(12, 4))
        self._plot(ax[0], label=label, **kwargs)
        ax.legend()
        return ax

    def _plot(self, ax, **kwargs):
        pass


class XCalibrationComponent(CalibrationComponent):
    def __init__(self, laser_wl, spe, spe_units, ref, ref_units, sample="Neon"):
        super(XCalibrationComponent, self).__init__(laser_wl, spe, spe_units, ref, ref_units, sample)
        self.spe_pos_dict = None

    def process(self, old_spe: Spectrum, spe_units="cm-1", convert_back=False):
        logger.debug("convert spe_units {} --> model units {}".format(spe_units, self.model_units))
        new_spe = self.convert_units(old_spe, spe_units, self.model_units)
        logger.debug("process", self)
        if self.model is None:
            return new_spe
        elif self.enabled:
            if isinstance(self.model, interpolate.RBFInterpolator):
                new_spe.x = self.model(new_spe.x.reshape(-1, 1))
            elif isinstance(self.model, float):
                new_spe.x = new_spe.x + self.model
        else:
            return new_spe
        # convert back
        if convert_back:
            # print("convert back", spe_units)
            return self.convert_units(new_spe, self.model_units, spe_units)
        else:
            return new_spe

    def _plot(self, ax, **kwargs):
        # self.spe.plot(ax=ax[0].twinx(), label=self.spe_units)

        ax.stem(self.spe_pos_dict.keys(), self.spe_pos_dict.values(), linefmt='b-', basefmt=' ',
                label="{} peaks".format(self.sample))
        ax.twinx().stem(self.ref.keys(), self.ref.values(), linefmt='r-', basefmt=' ',
                        label="Reference {}".format(self.sample))
        ax.set_xlabel("{}".format(self.ref_units))
        ax.legend()
        # fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        # ax.scatter(x_spe, x_reference, marker='o')
        # ax.set_xlabel("spectrum x ({})".format(self.ref_units))
        # ax.set_ylabel("reference x ({})".format(self.ref_units))

        # x_plot = np.linspace(min(x_spe), max(x_spe), 20)
        # y_plot = interp(x_plot.reshape(-1, 1))
        # ax.scatter(x_plot, y_plot, marker='.', label=kwargs["kernel"])

    def _plot_peaks(self, ax, **kwargs):
        # self.model.peaks
        pass
        # fig, ax = plt.subplots(3,1,figsize=(12,4))
        # spe.plot(ax=ax[0].twinx(),label=spe_units)
        # spe_to_process.plot(ax=ax[1],label=ref_units)

    def derive_model(self, find_kw={}, fit_peaks_kw={}, should_fit=False, name=None):

        # convert to ref_units
        logger.debug("[{}]: convert spe_units {} to ref_units {}".format(self.name, self.spe_units, self.ref_units))
        spe_to_process = self.convert_units(self.spe, self.spe_units, self.ref_units)
        logger.debug("max x", max(spe_to_process.x), self.ref_units)
        # spe_to_process.plot(ax=ax[1], label=self.ref_units)

        # if should_fit:
        #     spe_pos_dict = spe_to_process.fit_peak_positions(center_err_threshold=1,
        #                    find_peaks_kw=find_kw, fit_peaks_kw=fit_peaks_kw)  # type: ignore
        # else:
        #     find_kw = dict(sharpening=None)
        #     spe_pos_dict = spe_to_process.find_peak_multipeak(**find_kw).get_pos_ampl_dict()  # type: ignore
        # prominence = prominence, wlen=wlen, width=width
        find_kw = dict(sharpening=None)
        if should_fit:
            self.spe_pos_dict = spe_to_process.fit_peak_positions(
                    center_err_threshold=10, find_peaks_kw=find_kw, fit_peaks_kw=fit_peaks_kw)  # type: ignore
            # fit_res = spe_to_process.fit_peak_multimodel(candidates=cand, **fit_peaks_kw)
            # pos, amp = fit_res.center_amplitude(threshold=1)
            # spe_pos_dict = dict(zip(pos, amp))
        else:
            # prominence = prominence, wlen=wlen, width=width
            print("find_peak_multipeak")
            cand = spe_to_process.find_peak_multipeak(**find_kw)
            self.spe_pos_dict = cand.get_pos_ampl_dict()
        x_spe, x_reference, x_distance, df = rc2utils.match_peaks(self.spe_pos_dict, self.ref)
        sum_of_differences = np.sum(np.abs(x_spe - x_reference)) / len(x_spe)
        logger.debug("sum_of_differences original {} {}".format(sum_of_differences, self.ref_units))
        if len(x_reference) == 1:
            _offset = (x_reference[0] - x_spe[0])
            logger.debug("ref", x_reference[0], "sample", x_spe[0], "offset", _offset, self.ref_units)
            self.set_model(_offset, self.ref_units, df, name)
        else:
            try:
                kwargs = {"kernel": "thin_plate_spline"}
                interp = interpolate.RBFInterpolator(x_spe.reshape(-1, 1), x_reference, **kwargs)
                self.set_model(interp, self.ref_units, df, name)
            except Exception as err:
                raise err


class LazerZeroingComponent(CalibrationComponent):
    def __init__(self, laser_wl, spe, spe_units="nm", ref={520.45: 1}, ref_units="cm-1", sample="Silicon"):
        super(LazerZeroingComponent, self).__init__(laser_wl, spe, spe_units, ref, ref_units, sample)
        self.profile = "Pearson4"

    def derive_model(self, find_kw={}, fit_peaks_kw={}, should_fit=True, name=None):
        find_kw = dict(sharpening=None)
        cand = self.spe.find_peak_multipeak(**find_kw)
        logger.debug(self.name, cand)
        # init_guess = self.spe.fit_peak_multimodel(profile='Pearson4', candidates=cand, no_fit=False)
        fit_res = self.spe.fit_peak_multimodel(profile=self.profile, candidates=cand, **fit_peaks_kw)
        df = fit_res.to_dataframe_peaks()
        df = df.sort_values(by='height', ascending=False)
        if df.empty:
            raise Exception("No peaks found")
        else:
            if "position" in df.columns:
                zero_peak_nm = df.iloc[0]["position"]
            elif "center" in df.columns:
                zero_peak_nm = df.iloc[0]["center"]
            # print(self.name, "peak", zero_peak_nm)
            # https://www.elodiz.com/calibration-and-validation-of-raman-instruments/
            self.set_model(zero_peak_nm, "nm", df, "Lazer zeroing using {} nm".format(zero_peak_nm))
            logger.info(self.name, "peak", zero_peak_nm)
        # laser_wl should be calculated  based on the peak position and set instead of the nominal

    def zero_nm_to_shift_cm_1(self, wl, zero_pos_nm, zero_ref_cm_1=520.45):
        return 1e7*(1/zero_pos_nm - 1/wl) + zero_ref_cm_1

    # we do not do shift (as initially implemented)
    # just convert the spectrum nm->cm-1 using the Si measured peak in nm and reference in cm-1
    # https://www.elodiz.com/calibration-and-validation-of-raman-instruments/
    def process(self, old_spe: Spectrum, spe_units="nm", convert_back=False):
        wl_si_ref = list(self.ref.keys())[0]
        logger.debug(self.name, "process", self.model, wl_si_ref)
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
        for param in self.params.split(','):
            key, value = param.split('=')
            key = key.strip()
            value = value.strip()
            local_vars[key] = eval(value)

        def evaluate_expression(x_value):
            local_vars['x'] = x_value
            return eval(self.equation, {"np": np}, local_vars)

        return evaluate_expression

    def Y(self,  x_value):
        return self.response_function(x_value)

    def _plot(self, ax, **kwargs):
        if self.raman_shift is None:
            x = np.linspace(100, 4000)
        else:
            x = np.linspace(self.raman_shift[0], self.raman_shift[1])
        kwargs.pop('label', None)
        ax.plot(x, self.Y(x), label="{} ({}nm)".format(self.id, self.wavelength), **kwargs)
        ax.set_xlabel('Raman shift cm-1')
        ax.set_ylabel('Intensity')

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
        self.load_certificates(os.path.join(os.path.dirname(__file__), "config_certs.json"))

    def load_certificates(self, file_path):

        with open(file_path, 'r') as f:
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
                        certificate = YCalibrationCertificate(**certificate_data)
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
    Class for intensity calibration. Uses response functions loaded in ResponseFunctionEvaluator. Functions are defined
    in json file.

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

    def __init__(self, laser_wl, reference_spe_xcalibrated, certificate: YCalibrationCertificate):
        super(YCalibrationComponent, self).__init__(laser_wl, spe=reference_spe_xcalibrated, spe_units=None,
                                                    ref=certificate, ref_units=None)
        self.laser_wl = laser_wl
        self.spe = reference_spe_xcalibrated
        self.ref = certificate
        self.name = "Y calibration"
        self.model = self.spe.spe_distribution(trim_range=certificate.raman_shift)
        self.model_units = "cm-1"

    def derive_model(self, find_kw={}, fit_peaks_kw={}, should_fit=True, name=None):
        # measured reference spectrum as distribution, so we can resample
        self.model = self.spe.spe_distribution(trim_range=self.ref.raman_shift)

    def safe_divide(self, spe_to_correct, spe_reference_resampled):
        numerator = spe_to_correct.y
        # numerator_noise = spe_to_correct.y_noise

        scaling_denominator = spe_reference_resampled.y / self.ref.Y(spe_reference_resampled.x)
        print(np.median(scaling_denominator), np.mean(scaling_denominator), np.std(scaling_denominator))

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
        ref_noise = spe_reference_resampled.y_noise
        return (spe_reference_resampled.y >= 0) & (abs(spe_reference_resampled.y) > ref_noise)

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
        _tmp = _tmp * max(self.spe.y) / max(_tmp)  # pdf sampling is normalized to area unity, scaling back
        spe_reference_resampled = Spectrum(old_spe.x, _tmp)
        # new_spe = Spectrum(old_spe.x,self.safe_divide(old_spe,spe_reference_resampled))
        new_spe = Spectrum(old_spe.x, self.safe_factor(old_spe, spe_reference_resampled))
        return new_spe

    def _plot(self, ax, **kwargs):
        if self.ref is not None:
            self.ref.plot(ax, **kwargs)


class CalibrationModel(ProcessingModel, Plottable):
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
        calmodel = CalibrationModel(laser_wl=785)
        calmodel.derive_model_x(
            spe_neon,
            spe_neon_units="cm-1",
            ref_neon=None,
            ref_neon_units="nm",
            spe_sil=None,
            spe_sil_units="cm-1",
            ref_sil=None,
            ref_sil_units="cm-1"
            )
        # Store
        calmodel.save(modelfile)
        # Load
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
        self.neon_wl = {
            785: rc2const.neon_wl_785_nist_dict,
            633: rc2const.neon_wl_633_nist_dict,
            532: rc2const.neon_wl_532_nist_dict
        }
        self.prominence_coeff = 10

    def peaks(self, spe, profile='Gaussian', wlen=300, width=1):
        """
        Finds and fits peaks in the spectrum spe.
        """
        cand = spe.find_peak_multipeak(prominence=spe.y_noise*self.prominence_coeff, wlen=wlen, width=width)
        init_guess = spe.fit_peak_multimodel(profile=profile, candidates=cand, no_fit=True)
        fit_res = spe.fit_peak_multimodel(profile=profile, candidates=cand)
        return cand, init_guess, fit_res

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

    def derive_model_x(self, spe_neon, spe_neon_units="cm-1", ref_neon=None, ref_neon_units="nm", spe_sil=None,
                       spe_sil_units="cm-1", ref_sil=None, ref_sil_units="cm-1", find_kw={}, fit_kw={}):
        """
        Derives x-calibration models using Neon and Silicon spectra.
        """
        model_neon = self.derive_model_curve(
                spe_neon, self.neon_wl[self.laser_wl], spe_units=spe_neon_units, ref_units=ref_neon_units, find_kw={},
                fit_peaks_kw={}, should_fit=False, name="Neon calibration")
        spe_sil_ne_calib = model_neon.process(spe_sil, spe_units=spe_sil_units, convert_back=False)
        find_kw = {"prominence": spe_sil_ne_calib.y_noise * 10, "wlen": 200, "width":  1}
        model_si = self.derive_model_zero(
                spe_sil_ne_calib, ref={520.45: 1}, spe_units="nm", ref_units=ref_sil_units, find_kw=find_kw,
                fit_peaks_kw={}, should_fit=True, name="Si laser zeroing")
        return (model_neon, model_si)

    def derive_model_curve(self, spe, ref, spe_units="cm-1", ref_units="nm", find_kw={}, fit_peaks_kw={},
                           should_fit=False, name="X calibration"):
        calibration_x = XCalibrationComponent(self.laser_wl, spe, spe_units, ref, ref_units)
        calibration_x.derive_model(find_kw=find_kw, fit_peaks_kw=fit_peaks_kw, should_fit=should_fit, name=name)
        self.components.append(calibration_x)
        return calibration_x

    def derive_model_zero(self, spe, ref, spe_units="nm", ref_units="cm-1", find_kw={}, fit_peaks_kw={},
                          should_fit=False, name="X Shift", profile="Gaussian"):
        calibration_shift = LazerZeroingComponent(self.laser_wl, spe, spe_units, ref, ref_units)
        calibration_shift.profile = profile
        calibration_shift.derive_model(find_kw=find_kw, fit_peaks_kw=fit_peaks_kw, should_fit=should_fit, name=name)
        self.components.append(calibration_shift)
        return calibration_shift

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

    def plot(self, ax=None, label=' ', **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        self._plot(ax, **kwargs)
        return ax

    def _plot(self, ax, **kwargs):
        for index, model in enumerate(self.components):
            model._plot(ax, **kwargs)
            break
