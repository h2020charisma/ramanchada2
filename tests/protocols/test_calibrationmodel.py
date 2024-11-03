#!/usr/bin/env python3

import traceback

import matplotlib.pyplot as plt
import numpy as np
import pytest

import ramanchada2.misc.constants as rc2const
from ramanchada2.protocols.calibration.calibration_model import CalibrationModel
from ramanchada2.protocols.calibration.ycalibration import (
    CertificatesDict,
    YCalibrationCertificate,
    YCalibrationComponent,
)
from ramanchada2.spectrum import from_test_spe

from sklearn.metrics.pairwise import cosine_similarity

_plots = False
si_peak = 520.45


class SetupModule:
    def __init__(self):
        self.laser_wl = 785
        _provider = "ICV"
        _optical_path = "100"
        _device = "BWtek"
        _wl_str = str(self.laser_wl)
        self.si_profile = "Pearson4"
        # self.si_profile = "Gaussian"
        self.spe_neon = from_test_spe(
            sample=["Neon"], provider=[_provider], device=[_device],  OP=[_optical_path], laser_wl=[_wl_str]
        )

        self.spe_pst2 = from_test_spe(
            sample=["PST"], provider=[_provider], device=[_device], OP=[_optical_path], laser_wl=[_wl_str]
        )
        self.spe_pst3 = from_test_spe(
            sample=["PST"], provider=[_provider], device=[_device], OP=["020"], laser_wl=[_wl_str]
        )
        self.spe_sil = from_test_spe(
            sample=["S0B"], provider=[_provider], device=[_device], OP=[_optical_path], laser_wl=[_wl_str]
        )
        self.spe_sil2 = from_test_spe(
            sample=["S0B"], provider=[_provider], device=[_device], OP=[_optical_path], laser_wl=[_wl_str]
        )
        self.spe_nCal = from_test_spe(
            sample=["nCAL"], provider=[_provider], device=[_device], OP=[_optical_path], laser_wl=[_wl_str]
        )

        self.spe_SRM2241 = from_test_spe(
            sample=["NIST785_SRM2241"], provider=[_provider], device=[_device], OP=[_optical_path], laser_wl=[_wl_str]
        )
        # NIR785_EL0-9002

        self.spe_sil = self.spe_sil.trim_axes(
            method="x-axis", boundaries=(max(100, 520.45 - 50), 520.45 + 50)
        )
        self.spe_sil2 = self.spe_sil2.trim_axes(
            method="x-axis", boundaries=(max(100, 520.45 - 50), 520.45 + 50)
        )

        self.spe_neon = self.spe_neon.trim_axes(
            method="x-axis", boundaries=(100, max(self.spe_neon.x))
        )
        kwargs = {"niter": 40}
        self.spe_neon = self.spe_neon.subtract_baseline_rc1_snip(**kwargs)
        self.spe_sil = self.spe_sil.subtract_baseline_rc1_snip(**kwargs)
        self.spe_sil2 = self.spe_sil2.subtract_baseline_rc1_snip(**kwargs)

        # don't normalize , it makes the Pearson4 fit worse!
        # self.spe_neon = self.spe_neon.normalize()
        # self.spe_sil = self.spe_sil.normalize()
        # self.spe_sil2 = self.spe_sil2.normalize()

        fit_peaks = True

        try:
            neon_wl = rc2const.NEON_WL[785]
            self.calmodel = CalibrationModel.calibration_model_factory(
                self.laser_wl,
                self.spe_neon,
                self.spe_sil,
                neon_wl=neon_wl,
                find_kw={"wlen": 200, "width": 1},
                fit_peaks_kw={},
                should_fit=fit_peaks,
                match_method="cluster",
                si_profile=self.si_profile
            )
            assert len(self.calmodel.components) == 2
            # print(self.calmodel.components[0])
            # print(self.calmodel.components[1])
            # print(self.calmodel.components[1].profile, self.calmodel.components[1].peaks)
        except Exception as _err:
            self.calmodel = None
            print(_err)
            traceback.print_exc()


@pytest.fixture(scope="module")
def setup_module():
    return SetupModule()


def fit_si(spe_sil):
    spe_sil = spe_sil.trim_axes(
            method="x-axis", boundaries=(520.45 - 50, 520.45 + 50)
        )

    find_kw = {"wlen": 200, "width": 2, "sharpening": None}
    find_kw["prominence"] = spe_sil.y_noise_MAD() * 3
    cand = spe_sil.find_peak_multipeak(**find_kw)
    fitres = spe_sil.fit_peak_multimodel(profile="Pearson4", candidates=cand, no_fit=False, vary_baseline=False)
    df = fitres.to_dataframe_peaks().sort_values(by="height", ascending=False)
    difference = abs(df.iloc[0]["center"] - si_peak)
    assert difference < 1E-2, f"Si peak found at {df.iloc[0]['center']}"


def test_laser_zeroing(setup_module):
    assert setup_module.calmodel is not None
    spe_sil_calib = setup_module.calmodel.apply_calibration_x(
        setup_module.spe_sil, spe_units="cm-1"
    )
    spe_test = spe_sil_calib  # .trim_axes(method='x-axis',boundaries=(si_peak-50,si_peak+50))
    find_kw = {"wlen": 200, "width": 1, "sharpening": None}
    find_kw["prominence"] = spe_test.y_noise_MAD() * setup_module.calmodel.prominence_coeff
    cand = spe_test.find_peak_multipeak(**find_kw)
    fitres = spe_test.fit_peak_multimodel(profile=setup_module.si_profile, candidates=cand, no_fit=False)
    df = fitres.to_dataframe_peaks().sort_values(by="height", ascending=False)

    _units = setup_module.calmodel.components[1].model_units
    assert _units == "nm"

    if _plots:
        fig, (ax, axpeak, axsifit) = plt.subplots(3, 1, figsize=(12, 4))
        setup_module.spe_sil.plot(label="Si original", ax=ax)
        spe_sil_calib.plot(ax=ax, label="Si laser zeroed", fmt=":")
        plt.grid()
        ax.set_xlim(si_peak-50, si_peak+50)
        if _units == "cm-1":
            _units = r"$\mathrm{{[cm^{-1}]}}$"
        ax.set_xlabel(_units)
        fitres.plot(ax=axpeak, label="fit res")
        spe_test.plot(ax=axpeak, label="Si laser zeroed", fmt=":")
        axpeak.set_xlim(si_peak-50, si_peak+50)
        # let's look at the peak before laser zeroing

        ne_model = setup_module.calmodel.components[0]
        si_model = setup_module.calmodel.components[1]
        spe_test_necalibrated_only = ne_model.process(spe_test)
        si_model.fit_res.plot(ax=axsifit)
        si_peak_nm = si_model.model
        axsifit.set_xlim(si_peak_nm-5, si_peak_nm+5)
        spe_test_necalibrated_only.plot(ax=axsifit, label="Si (Ne calibrated only)", fmt=":")

        print(df.sort_values(by="amplitude", ascending=False).head())
        plt.grid()
        plt.tight_layout()
        plt.savefig("test_calmodel_{}.png".format("laser_zeroing"))

    difference = abs(df.iloc[0]["center"] - si_peak)
    assert difference < 1E-2, f"Si peak found at {df.iloc[0]['center']}"


def resample(spe, xmin, xmax, npoints):
    x_values = np.linspace(xmin, xmax, npoints)
    dist = spe.spe_distribution(trim_range=(xmin, xmax))
    y_values = dist.pdf(x_values)
    scale = np.max(spe.y) / np.max(y_values)
    # pdf sampling is normalized to area unity, scaling back
    # tbd - make resample a filter
    return y_values * scale


def resample_spline(spe, xmin, xmax, npoints):
    spe_resampled = spe.resample_spline_filter(
        (xmin, xmax), xnew_bins=1000, spline="akima", cumulative=False
    )
    # spe_resampled = spe.resample_NUDFT_filter(x_range=(xmin,xmax), xnew_bins=npoints)
    return spe_resampled


def compare_calibrated_spe(setup_module, spectra, name="calibration"):
    if _plots:
        fig, ax = plt.subplots(len(spectra) + 1, 1, figsize=(24, 8))
        setup_module.calmodel.plot(ax=ax[0])
    crl = [("blue", "red"), ("green", "gray")]
    spe_y_original = []
    _min = 200
    _max = 2000
    spe_calibrated = []
    for index, spe in enumerate(spectra):
        # check if x is monotonically increasing
        assert np.all(np.diff(spe.x) > 0)
        spe_norm = spe.normalize(strategy="unity_area")
        # resample with spline
        spe_norm = resample_spline(spe_norm, _min, _max, _max - _min)
        # resample with histogram
        # spe_y_original.append(resample(spe_norm,_min,_max,_max-_min))
        spe_y_original.append(spe_norm.y)
        spe_c = setup_module.calmodel.apply_calibration_x(spe, spe_units="cm-1")
        spe_c_norm = spe_c.normalize(strategy="unity_area")
        spe_c_norm = resample_spline(spe_c_norm, _min, _max, _max - _min)
        spe_calibrated.append(spe_c_norm.y)
        _units = "cm^{-1}"
        _units = rf"$\mathrm{{[{_units}]}}$"
        if _plots:
            spe_norm.plot(ax=ax[index + 1], label=f"original {index}", color=crl[index][0])       
            spe_c_norm.plot(
                ax=ax[index + 1], label=f"calibrated {index}", color=crl[index][1]
            )
            ax[index + 1].set_xlabel(_units)
    cos_sim_matrix_original = cosine_similarity(spe_y_original)
    cos_sim_matrix = cosine_similarity(spe_calibrated)

    if _plots:
        plt.tight_layout()
        plt.savefig("test_calmodel_{}.png".format(name))
    print(name, np.mean(cos_sim_matrix_original), np.mean(cos_sim_matrix))
    # assert(np.mean(cos_sim_matrix_original) <= np.mean(cos_sim_matrix))
    assert np.mean(cos_sim_matrix_original) <= (np.mean(cos_sim_matrix) + 1e-4)


def test_xcalibration_pst(setup_module):
    assert setup_module.calmodel is not None
    compare_calibrated_spe(
        setup_module, [setup_module.spe_pst2, setup_module.spe_pst3], name="PST"
    )


def test_xcalibration_si(setup_module):
    assert setup_module.calmodel is not None
    compare_calibrated_spe(
        setup_module, [setup_module.spe_sil, setup_module.spe_sil2], name="Sil"
    )


def test_xcalibration_cal(setup_module):
    assert setup_module.calmodel is not None
    compare_calibrated_spe(
        setup_module, [setup_module.spe_nCal, setup_module.spe_nCal], name="nCal"
    )


def test_ycalibration(setup_module):
    certificates = CertificatesDict()
    certs = certificates.get_certificates(wavelength=int(setup_module.laser_wl))
    key = "NIST785_SRM2241"
    certificate = certs[key]
    ycal = YCalibrationComponent(
        setup_module.laser_wl, setup_module.spe_SRM2241, certificate=certificate
    )
    spe_to_correct = from_test_spe(
        sample=["PST"], provider=["FNMT"], OP=["03"], laser_wl=["785"]
    )
    # spe_to_correct = spe_to_correct.trim_axes(method='x-axis',boundaries=(100,2000))
    # remove the pedestal
    spe_to_correct.y = spe_to_correct.y - min(spe_to_correct.y)
    window_length = 5
    maxy = max(spe_to_correct.y)
    spe_to_correct1 = spe_to_correct.smoothing_RC1(
        method="savgol", window_length=window_length, polyorder=3
    )
    spe_to_correct1.y = maxy * spe_to_correct.y / max(spe_to_correct.y)
    spe_ycalibrated = ycal.process(spe_to_correct1)

    if _plots:
        fig, ax = plt.subplots(3, 1, figsize=(24, 8))
        certificate.plot(ax=ax[2], color="pink")
        spe_to_correct.plot(ax=ax[0], label="PST")
        spe_to_correct1.plot(ax=ax[0], label="smoothed")
        setup_module.spe_SRM2241.plot(ax=ax[0].twinx(), label=key)
        spe_ycalibrated.plot(ax=ax[1], label="y calibrated")
        plt.savefig("test_calmodel_{}.png".format(key))


def test_ycertificate():
    cert = YCalibrationCertificate(
        id="NIST785_SRM2241",
        description="optical glass",
        url="https://tsapps.nist.gov/srmext/certificates/2241.pdf",
        wavelength=785,
        params=(
            "A0 = 9.71937e-02, "
            "A1 = 2.28325e-04, "
            "A2 = -5.86762e-08, "
            "A3 = 2.16023e-10, "
            "A4 = -9.77171e-14, "
            "A5 = 1.15596e-17"
        ),
        equation="A0 + A1 * x + A2 * x**2 + A3 * x**3 + A4 * x**4 + A5 * x**5",
        temperature_c=(20, 25),
        raman_shift=(200, 3500),
    )
    if _plots:
        cert.plot()
        plt.savefig("test_calmodel_{}.png".format("ycert"))


def test_ycerts_dict():
    certificates = CertificatesDict()
    print(type(certificates.get_certificates(785)))
    assert certificates, "empty certificate"
    assert certificates.get_certificates(785), "empty certificate"
    cert = CertificatesDict.load(wavelength="785", key="NIST785_SRM2241")
    assert "NIST785_SRM2241" == cert.id
