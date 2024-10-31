import numpy as np
from ramanchada2.spectrum import Spectrum
from .CalibrationComponent import CalibrationComponent
from .YCalibrationCertificate import YCalibrationCertificate


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

    def derive_model(self, find_kw={}, fit_peaks_kw={}, should_fit=True, name=None):
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
