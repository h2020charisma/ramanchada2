import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from sklearn.linear_model import LinearRegression

from ramanchada2.misc.plottable import Plottable
from ramanchada2.protocols.spectraframe import SpectraFrame
from ramanchada2.spectrum import Spectrum


class TwinningComponent(Plottable):
    """
    TwinningComponent is an implementation of CHARISMA Twinning protocol CWA18134 Sep 2024
    https://www.cencenelec.eu/media/CEN-CENELEC/CWAs/RI/2024/cwa18134-1.pdf
    It expects reference spectra and test spectra (to be twinned) as SpectraFrame objects.

    Attributes:
        grouping_cols (list): The SpectraFrame may contain replicates, which will be averaged by grouping by
            columns except replicates, e.g.
            ['sample', 'provider', 'laser_wl', 'laser_power_percent', 'laser_power_mW', 'time_ms'].

        reference (SpectraFrame): The averaged reference spectra.

        twinned (SpectraFrame): The averaged spectra to be twinned.

        boundaries (tuple): A tuple representing the boundaries for analysis (default: (50, 2000)).

        linreg_reference (tuple): Placeholder for storing the result of a linear regression on the reference spectra.
            Defaults to a tuple (None, None) which can later hold the regression slope and intercept.

        linreg_twinned (tuple): Placeholder for storing the result of a linear regression on the twinned spectra.
            Defaults to a tuple (None, None) which can later hold the regression slope and intercept.

        correction_factor (float): A scaling factor derived as ratio of slopes as defined in CWA18134.

        peak (float): The position of the peak (in nm) of interest for analysis, with a default value of 144 (TiO2).

    Methods:
        __init__(self, twinned: SpectraFrame, reference: SpectraFrame, boundaries=None, peak_at=144):
            Initializes a new TwinningComponent object by averaging the provided twinned and reference spectra
            based on predefined grouping columns. Optionally, boundaries for analysis and a peak position can be
            specified.

    Parameters:
        twinned (SpectraFrame): The SpectraFrame representing the twinned data.
        reference (SpectraFrame): The SpectraFrame representing the reference data.
        boundaries (tuple, optional): Optional boundary values to restrict the analysis region (default: (50, 2000)).
        peak_at (int, optional): The peak position to focus the analysis on (default: 144 for TiO2).
    """

    def __init__(
        self,
        twinned: SpectraFrame,
        reference: SpectraFrame,
        boundaries=None,
        reference_band_nm=144.0,
        grouping_cols=[
            "sample",
            "provider",
            "device",
            "laser_wl",
            "laser_power_percent",
            "laser_power_mW",
            "time_ms",
        ],
    ):
        self.grouping_cols = grouping_cols
        self.twinned = twinned.average(grouping_cols=self.grouping_cols)
        self.reference = reference.average(grouping_cols=self.grouping_cols)
        self.boundaries = (50, 2000) if boundaries is None else boundaries
        self.linreg_reference = (None, None)
        self.linreg_twinned = (None, None)
        self.correction_factor: float = 1.0
        self.reference_band_nm = reference_band_nm

    def normalize_by_laserpower_time(self, source="spectrum", target="spectrum"):
        for (index_refere4nce, row_reference), (index_twinned, row_twinned) in zip(
            self.reference.iterrows(), self.twinned.iterrows()
        ):
            laser_power_ratio = (
                row_reference["laser_power_mW"] / row_twinned["laser_power_mW"]
            )
            time_ratio = row_reference["time_ms"] / row_twinned["time_ms"]
            spe = row_twinned[source]
            self.twinned.at[index_twinned, target] = (
                spe * laser_power_ratio * time_ratio
            )
            self.twinned.at[index_twinned, "laser_power_ratio"] = laser_power_ratio
            self.twinned.at[index_twinned, "time_ratio"] = time_ratio

    def calc_peak_intensity(
        self,
        spe: Spectrum,
        boundaries=None,
        prominence_coeff=0.01,
        no_fit=False,
        peak_intensity="height",
    ):
        try:
            if boundaries is None:
                boundaries = (self.reference_band_nm - 50, self.reference_band_nm + 50)
            # TODO: Check if the MyPy type ignores below can be handled better.
            spe = spe.trim_axes(method="x-axis", boundaries=boundaries)  # type: ignore
            prominence = spe.y_noise_MAD() * prominence_coeff
            candidates = spe.find_peak_multipeak(prominence=prominence)  # type: ignore
            fit_res = spe.fit_peak_multimodel(  # type: ignore
                profile="Voigt", candidates=candidates, no_fit=no_fit
            )
            df = fit_res.to_dataframe_peaks()
            df["sorted"] = abs(
                df["center"] - self.reference_band_nm
            )  # closest peak to 144
            df_sorted = df.sort_values(by="sorted")

            # we get actual y value, not height or amplitude
            index_left = np.searchsorted(
                spe.x, df_sorted["center"].iloc[0], side="left", sorter=None
            )
            index_right = np.searchsorted(
                spe.x, df_sorted["center"].iloc[0], side="right", sorter=None
            )
            if index_right == index_left:
                peak_intensity = spe.y[index_left]
                peak_position = spe.x[index_left]
            else:
                peak_intensity = (spe.y[index_right] + spe.y[index_left]) / 2.0
                peak_position = (spe.x[index_right] + spe.x[index_left]) / 2

            # _label = "intensity = {:.3f} {} ={:.3f} amplitude={:.3f} center={:.1f}".format(
            #    intensity_val,peak_intensity,df_sorted.iloc[0][peak_intensity],
            #    df_sorted.iloc[0]["amplitude"],df_sorted.iloc[0]["center"])

            return peak_intensity, peak_position, fit_res
        except Exception as err:
            print(err)
            return None, None, None

    def laser_power_regression(
        self, df: SpectraFrame, boundaries=None, no_fit=False, source="spectrum"
    ):
        # fig, ax = plt.subplots(df.shape[0],1,figsize=(12,12))
        r = 0
        for index, row in df.iterrows():
            spe = row[source]
            if boundaries is None:
                boundaries = (self.reference_band_nm - 50, self.reference_band_nm + 50)
            peak_intensity, peak_position, fit_res = self.calc_peak_intensity(
                spe, boundaries=boundaries, no_fit=no_fit
            )
            df.at[index, "peak_intensity"] = peak_intensity
            df.at[index, "peak_position"] = peak_position
            # spe.trim_axes(method='x-axis',boundaries=boundaries).plot(ax=ax[r], fmt=':',label=row["laser_power_mW"])
            # if fit_res is not None:
            #    fit_res.plot(ax=ax[r])
            r = r + 1
        # plt.savefig("test_twinning_peaks_{}.png".format(title))
        # print(df[['laser_power_mW','peak_intensity']])
        return LinearRegression().fit(
            df[["laser_power_mW"]].values, df["peak_intensity"].values
        )

    def process(
        self, spe: SpectraFrame, source="spectrum", target="spectrum_harmonized"
    ):
        spe.multiply(self.correction_factor, source=source, target=target)
        spe.spe_area(
            source=target, target="area_harmonized", boundaries=self.boundaries
        )

    def derive_model(self):
        self.reference.trim(
            boundaries=self.boundaries, source="spectrum", target="spe_processed"
        )
        self.twinned.trim(
            boundaries=self.boundaries, source="spectrum", target="spe_processed"
        )

        self.normalize_by_laserpower_time(
            source="spe_processed", target="spe_processed"
        )

        self.reference.baseline_snip(source="spe_processed", target="spe_processed")
        self.twinned.baseline_snip(source="spe_processed", target="spe_processed")

        boundaries4area = self.boundaries
        self.reference.spe_area(
            target="area", boundaries=boundaries4area, source="spe_processed"
        )
        self.twinned.spe_area(
            target="area", boundaries=boundaries4area, source="spe_processed"
        )

        model_reference = self.laser_power_regression(
            self.reference, no_fit=False, source="spe_processed"
        )
        self.linreg_reference = (model_reference.intercept_, model_reference.coef_[0])

        model_twinned = self.laser_power_regression(
            self.twinned, no_fit=False, source="spe_processed"
        )
        self.linreg_twinned = (model_twinned.intercept_, model_twinned.coef_[0])

        self.correction_factor = model_reference.coef_[0] / model_twinned.coef_[0]
        self.twinned["correction_factor"] = self.correction_factor

    def plot(self, ax=None, label=" ", **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        self._plot(ax, label=label, **kwargs)

        return ax

    def _plot(self, ax, **kwargs):
        A = self.reference
        B = self.twinned
        regression_A = self.linreg_reference
        regression_B = self.linreg_twinned
        # fig, axes = plt.subplots(1,2, figsize=(10,4))
        ax[0].plot(
            A["laser_power_mW"], A["peak_intensity"], "o", label=A["device_id"].unique()
        )

        A_pred = A["laser_power_mW"] * regression_A[1] + regression_A[0]
        ax[0].plot(
            A["laser_power_mW"],
            A_pred,
            "-",
            label="{:.2e} * LP + {:.2e}".format(regression_A[1], regression_A[0]),
        )

        ax[0].plot(
            B["laser_power_mW"], B["peak_intensity"], "+", label=B["device_id"].unique()
        )

        # axes[0].plot(
        #     B["laser_power_mW"],
        #     B["peak_intensity"] * correction_factor,
        #     "+",
        #     label="{} corrected".format(B["device_id"].unique()),
        # )

        B_pred = B["laser_power_mW"] * regression_B[1] + regression_B[0]
        ax[0].plot(
            B["laser_power_mW"],
            B_pred,
            "-",
            label="{:.2e} * LP + {:.2e}".format(regression_B[1], regression_B[0]),
        )

        ax[0].set_ylabel("Peak intensity of the (fitted) peak @ 144cm-1")
        ax[0].set_xlabel("laser power, %")
        ax[0].legend()
        bar_width = 0.2  # Adjust this value to control the width of the groups
        bar_positions = np.arange(len(A["laser_power_percent"].values))
        ax[1].bar(
            bar_positions - bar_width,
            A["area"],
            width=bar_width,
            label=str(A["device_id"].unique()),
        )
        bar_positions = np.arange(len(B["laser_power_percent"].values))
        ax[1].bar(
            bar_positions,
            B["area"],
            width=bar_width,
            label=str(B["device_id"].unique()),
        )
        ax[1].bar(
            bar_positions + bar_width,
            B["area_harmonized"],
            width=bar_width,
            label="{} harmonized CF={:.2e}".format(
                B["device_id"].unique(), self.correction_factor
            ),
        )
        ax[1].set_ylabel("spectrum area")
        ax[1].set_xlabel("laser power, %")
        # Set the x-axis positions and labels
        plt.xticks(bar_positions, B["laser_power_percent"])
        ax[1].legend()
        plt.tight_layout()
