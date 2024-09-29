#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pytest
import ramanchada2 as rc2
import ramanchada2.misc.constants as rc2const
from ramanchada2.protocols.calibration import (
    CalibrationModel,
    CertificatesDict,
    YCalibrationCertificate,
    YCalibrationComponent,
)
from sklearn.metrics.pairwise import cosine_similarity


class SetupModule:
    def __init__(self):
        self.laser_wl = 785
        self.spe_neon = rc2.spectrum.from_test_spe(
            sample=["Neon"], provider=["FNMT"], OP=["03"], laser_wl=["785"]
        )
        self.spe_pst2 = rc2.spectrum.from_test_spe(
            sample=["PST"], provider=["FNMT"], OP=["02"], laser_wl=["785"]
        )
        self.spe_pst3 = rc2.spectrum.from_test_spe(
            sample=["PST"], provider=["FNMT"], OP=["03"], laser_wl=["785"]
        )
        self.spe_sil = rc2.spectrum.from_test_spe(
            sample=["S0B"], provider=["FNMT"], OP=["03"], laser_wl=["785"]
        )
        self.spe_sil2 = rc2.spectrum.from_test_spe(
            sample=["S0B"], provider=["FNMT"], OP=["02"], laser_wl=["785"]
        )
        self.spe_nCal = rc2.spectrum.from_test_spe(
            sample=["nCAL"], provider=["FNMT"], OP=["03"], laser_wl=["785"]
        )

        self.spe_SRM2241 = rc2.spectrum.from_test_spe(
            sample=["NIST785_SRM2241"], provider=["FNMT"], OP=["03"], laser_wl=["785"]
        )
        # NIR785_EL0-9002

        self.spe_sil = self.spe_sil.trim_axes(
            method="x-axis", boundaries=(max(100, 520.45 - 200), 520.45 + 200)
        )
        self.spe_sil2 = self.spe_sil2.trim_axes(
            method="x-axis", boundaries=(max(100, 520.45 - 200), 520.45 + 200)
        )

        self.spe_neon = self.spe_neon.trim_axes(
            method="x-axis", boundaries=(100, max(self.spe_neon.x))
        )
        kwargs = {"niter": 40}
        self.spe_neon = self.spe_neon.subtract_baseline_rc1_snip(**kwargs)
        self.spe_sil = self.spe_sil.subtract_baseline_rc1_snip(**kwargs)
        self.spe_sil2 = self.spe_sil2.subtract_baseline_rc1_snip(**kwargs)

        # normalize min/max
        self.spe_neon = self.spe_neon.normalize()
        self.spe_sil = self.spe_sil.normalize()
        self.spe_sil2 = self.spe_sil2.normalize()

        try:
            neon_wl = rc2const.NEON_WL[785]
            self.calmodel = CalibrationModel.calibration_model_factory(
                785,
                self.spe_neon,
                self.spe_sil,
                neon_wl=neon_wl,
                find_kw={"wlen": 200, "width": 1},
                fit_peaks_kw={},
                should_fit=False,
            )
            assert len(self.calmodel.components) == 2
            # print(self.calmodel.components[1].profile, self.calmodel.components[1].peaks)
        except Exception as _err:
            self.calmodel = None
            print(_err)


@pytest.fixture(scope="module")
def setup_module():
    return SetupModule()


def test_laser_zeroing(setup_module):
    assert setup_module.calmodel is not None
    fig, ax = plt.subplots(1, 1, figsize=(12, 2))
    spe_sil_calib = setup_module.calmodel.apply_calibration_x(
        setup_module.spe_sil, spe_units="cm-1"
    )
    setup_module.spe_sil.plot(label="Si original", ax=ax)
    spe_sil_calib.plot(ax=ax, label="Si laser zeroed", fmt=":")
    # ax.set_xlim(520.45-50,520.45+50)

    _units = setup_module.calmodel.components[1].model_units
    if _units == "cm-1":
        _units = rf'$\mathrm{{[cm^{-1}]}}$'    
    assert _units == "nm"
    ax.set_xlabel(_units)
    # print(setup_module.calmodel.components[1])
    plt.tight_layout()
    plt.savefig("test_calmodel_{}.png".format("laser_zeroing"))


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
    fig, ax = plt.subplots(len(spectra) + 1, 1, figsize=(24, 8))
    setup_module.calmodel.plot(ax=ax[0])
    crl = [("blue", "red"), ("green", "gray")]
    spe_y_original = []
    _min = 200
    _max = 2000
    spe_calibrated = []
    for index, spe in enumerate(spectra):
        spe_norm = spe.normalize(strategy="unity_area")
        # resample with spline
        spe_norm = resample_spline(spe_norm, _min, _max, _max - _min)
        # resample with histogram
        # spe_y_original.append(resample(spe_norm,_min,_max,_max-_min))
        spe_norm.plot(ax=ax[index + 1], label=f"original {index}", color=crl[index][0])
        spe_y_original.append(spe_norm.y)
        spe_c = setup_module.calmodel.apply_calibration_x(spe, spe_units="cm-1")
        spe_c_norm = spe_c.normalize(strategy="unity_area")
        spe_c_norm = resample_spline(spe_c_norm, _min, _max, _max - _min)
        spe_c_norm.plot(
            ax=ax[index + 1], label=f"calibrated {index}", color=crl[index][1]
        )
        spe_calibrated.append(spe_c_norm.y)
        _units = 'cm^{-1}'
        _units = rf'$\mathrm{{[{_units}]}}$'    
        ax[index + 1].set_xlabel(_units)        
    cos_sim_matrix_original = cosine_similarity(spe_y_original)
    cos_sim_matrix = cosine_similarity(spe_calibrated)
    plt.tight_layout()
    plt.savefig("test_calmodel_{}.png".format(name))
    # print(name,np.mean(cos_sim_matrix_original),np.mean(cos_sim_matrix))
    # assert(np.mean(cos_sim_matrix_original) <= np.mean(cos_sim_matrix))
    assert np.mean(cos_sim_matrix_original) <= (np.mean(cos_sim_matrix) + 1e-5)


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
    fig, ax = plt.subplots(3, 1, figsize=(24, 8))
    certificates = CertificatesDict()
    certs = certificates.get_certificates(wavelength=int(setup_module.laser_wl))
    key = "NIST785_SRM2241"
    certificate = certs[key]
    certificate.plot(ax=ax[2], color="pink")
    ycal = YCalibrationComponent(
        setup_module.laser_wl, setup_module.spe_SRM2241, certificate=certificate
    )

    spe_to_correct = rc2.spectrum.from_test_spe(
        sample=["PST"], provider=["FNMT"], OP=["03"], laser_wl=["785"]
    )
    # spe_to_correct = spe_to_correct.trim_axes(method='x-axis',boundaries=(100,2000))
    # remove the pedestal
    spe_to_correct.y = spe_to_correct.y - min(spe_to_correct.y)
    spe_to_correct.plot(ax=ax[0], label="PST")

    window_length = 5
    maxy = max(spe_to_correct.y)
    spe_to_correct = spe_to_correct.smoothing_RC1(
        method="savgol", window_length=window_length, polyorder=3
    )
    spe_to_correct.y = maxy * spe_to_correct.y / max(spe_to_correct.y)
    spe_to_correct.plot(ax=ax[0], label="smoothed")
    setup_module.spe_SRM2241.plot(ax=ax[0].twinx(), label=key)
    spe_ycalibrated = ycal.process(spe_to_correct)
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
    cert.plot()
    plt.savefig("test_calmodel_{}.png".format("ycert"))


def test_ycerts_dict():
    certificates = CertificatesDict()
    print(type(certificates.get_certificates(785)))
    assert certificates, "empty certificate"
    assert certificates.get_certificates(785), "empty certificate"
    cert = CertificatesDict.load(wavelength="785", key="NIST785_SRM2241")
    assert "NIST785_SRM2241" == cert.id
