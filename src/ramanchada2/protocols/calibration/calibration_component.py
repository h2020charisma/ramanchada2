import logging
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from ramanchada2.misc.plottable import Plottable
from ramanchada2.spectrum import Spectrum


logger = logging.getLogger(__name__)


class ProcessingModel:
    def __init__(self):
        pass


class CalibrationComponent(Plottable):
    nonmonotonic: Literal["ignore", "nan", "error", "drop"] = "error"

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
        self.fit_res = None

    def set_model(self, model, model_units, peaks, name=None):
        self.model = model
        self.model_units = model_units
        self.peaks = peaks
        self.name = "calibration component" if name is None else name

    def __str__(self):
        return (
            f"{self.name} spe ({self.spe_units}) reference ({self.ref_units}) "
            f"model ({self.model_units}) {self.model}"
        )

    def convert_units(self, old_spe, spe_unit="cm-1", newspe_unit="nm", laser_wl=None):
        if laser_wl is None:
            laser_wl = self.laser_wl
        logger.debug(
            "convert laser_wl {} {} --> {}".format(laser_wl, spe_unit, newspe_unit)
        )
        if spe_unit != newspe_unit:
            if spe_unit == "nm":
                new_spe = old_spe.abs_nm_to_shift_cm_1_filter(
                    laser_wave_length_nm=laser_wl
                )
            elif spe_unit == "cm-1":
                new_spe = old_spe.shift_cm_1_to_abs_nm_filter(
                    laser_wave_length_nm=laser_wl
                )
            elif spe_unit == "pixel":
                # pixel axes are not converted here; matching handles them via quantile mapping
                logger.warning(
                    "convert_units: pixel axis passed through unconverted (%s -> %s)",
                    spe_unit, newspe_unit
                )
                new_spe = old_spe.__copy__()
            else:
                raise ValueError(
                    f"Unsupported conversion {spe_unit} to {newspe_unit}"
                )
        else:
            new_spe = old_spe.__copy__()
        return new_spe

    def process(self, old_spe: Spectrum, spe_units="cm-1", convert_back=False):
        raise NotImplementedError(self)

    def derive_model(
        self, find_kw=None, fit_peaks_kw=None, should_fit=False, name=None
    ):
        raise NotImplementedError(self)

    def plot(self, ax=None, label=" ", **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots(3, 1, figsize=(12, 4))
        elif not isinstance(ax, (list, np.ndarray)):
            raise ValueError(
                "ax should be a list or array of Axes when creating multiple subplots."
            )

        self._plot(ax[0], label=label, **kwargs)
        ax[0].legend()
        return ax

    def _plot(self, ax, **kwargs):
        pass

    def __getstate__(self):
        # Return the state to be serialized, excluding transient_data
        state = self.__dict__.copy()
        del state["fit_res"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.fit_res = None

    def fitres2df(self, spe):
        df = pd.DataFrame(
            list(
                zip(
                    self.fit_res.centers,
                    self.fit_res.fwhm,
                    np.array(
                        [
                            v
                            for peak in self.fit_res
                            for k, v in peak.values.items()
                            if k.endswith("height")
                        ]
                    ),
                    np.array(
                        [
                            v
                            for peak in self.fit_res
                            for k, v in peak.values.items()
                            if k.endswith("amplitude")
                        ]
                    ),
                )
            ),
            columns=["center", "fwhm", "height", "amplitude"],
        )
        return df[(df["center"] >= min(spe.x)) & (df["center"] <= max(spe.x))]

    def pixels_to_wavenumber(self, pixel_array, reference_wn_dict, extension=30):
        """
        Scale pixel positions to wavenumber space based on reference peak range.

        Parameters
        ----------
        pixel_array : array-like
            Pixel positions to convert
        reference_wn_dict : dict
            Reference peaks as {wavenumber (cm⁻¹): intensity}
        extension : float, optional
            Extend the reference range by this amount (cm⁻¹) on each side

        Returns
        -------
        array-like
            Scaled positions in cm⁻¹

        Examples
        --------
        >>> neon_wn = {10**7 / wl: intensity for wl, intensity in neon_wl.items()}
        >>> scaled_x = pixels_to_wavenumber(spe_neon.x, neon_wn, extension=100)
        """
        # Get wavenumber range from reference
        wn_min = min(reference_wn_dict.keys()) - extension
        wn_max = max(reference_wn_dict.keys()) + 1

        # Get pixel range
        pix_min = min(pixel_array)
        pix_max = max(pixel_array)

        # Calculate scales
        scale_wn = wn_max - wn_min
        scale_pix = pix_max - pix_min

        # Linear scaling
        scaled = (pixel_array - pix_min) * scale_wn / scale_pix + wn_min
        logger.debug(f"pixels_to_wavenumber ({pix_min},{pix_max}) -> ({min(scaled)}, {max(scaled)})")
        return scaled
