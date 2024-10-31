from .CalibrationComponent import CalibrationComponent
from ramanchada2.spectrum import Spectrum
import logging

logger = logging.getLogger(__name__)


class LazerZeroingComponent(CalibrationComponent):
    def __init__(
        self,
        laser_wl,
        spe,
        spe_units="nm",
        ref={520.45: 1},
        ref_units="cm-1",
        sample="Silicon",
    ):
        super(LazerZeroingComponent, self).__init__(
            laser_wl, spe, spe_units, ref, ref_units, sample
        )
        self.profile = "Pearson4"

    def derive_model(self, find_kw=None, fit_peaks_kw=None, should_fit=True, name=None):
        if find_kw is None:
            find_kw = {}
        if fit_peaks_kw is None:
            fit_peaks_kw = {}

        cand = self.spe.find_peak_multipeak(**find_kw)
        logger.debug(self.name, cand)
        self.fit_res = self.spe.fit_peak_multimodel(
            profile=self.profile, candidates=cand, **fit_peaks_kw
        )
        # df = self.fit_res.to_dataframe_peaks()
        df = self.fitres2df(self.spe)
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
            zero_peak_cm1 = self.zero_nm_to_shift_cm_1(zero_peak_nm, zero_peak_nm, list(self.ref.keys())[0])
            self.set_model(
                zero_peak_nm, "nm", df, "Laser zeroing using {} nm {} cm-1 ({}) ".
                format(zero_peak_nm, zero_peak_cm1, self.profile
                       )
            )
            logger.info(self.name, f"peak {self.profile} at {zero_peak_nm} nm")
        # laser_wl should be calculated  based on the peak position and set instead of the nominal

    def zero_nm_to_shift_cm_1(self, wl, zero_pos_nm, zero_ref_cm_1=520.45):
        return 1e7 * (1 / zero_pos_nm - 1 / wl) + zero_ref_cm_1

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
