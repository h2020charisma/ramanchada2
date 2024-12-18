import matplotlib.pyplot as plt
import pytest
import ramanchada2 as rc2
from ramanchada2.auxiliary.spectra.datasets2 import filtered_df, prepend_prefix
from ramanchada2.protocols.spectraframe import SpectraFrame
from ramanchada2.protocols.twinning import TwinningComponent


def load_spectrum(row):
    return rc2.spectrum.Spectrum.from_local_file(row["filename"])


@pytest.fixture(scope="module")
def reference_with_replicates():
    df_ref = filtered_df(
        sample=["TiO2"], device=["BWtek"], provider=["FNMT-B"], laser_wl=["785"]
    )
    df_ref["filename"] = prepend_prefix(df_ref["filename"])
    df_ref["device_id"] = df_ref[["provider", "device", "laser_wl"]].agg(
        " ".join, axis=1
    )
    checkmetadata4twinning(df_ref, 25)
    df_ref["spectrum"] = df_ref.apply(load_spectrum, axis=1)
    return SpectraFrame.from_dataframe(df_ref, {"filename": "file_name"})


@pytest.fixture(scope="module")
def twinned_with_replicates():
    df_test = filtered_df(
        sample=["TiO2"], device=["BWtek"], provider=["ICV"], laser_wl=["785"]
    )
    df_test["filename"] = prepend_prefix(df_test["filename"])
    df_test["device_id"] = df_test[["provider", "device", "laser_wl"]].agg(
        " ".join, axis=1
    )
    checkmetadata4twinning(df_test, 25)
    df_test["spectrum"] = df_test.apply(load_spectrum, axis=1)
    return SpectraFrame.from_dataframe(df_test, {"filename": "file_name"})


def checkmetadata4twinning(df, expected_rows=25):
    for col in ["laser_power_mW", "laser_power_percent", "time_ms", "replicate"]:
        assert col in df.columns
        # these are strings in the test , we want numerics
        df[col] = df[col].astype(float)
    assert expected_rows == df.shape[0]


def plot_spectra(df: SpectraFrame, title="spectra", source="spectrum"):
    fig, ax = plt.subplots(1, 1, figsize=(12, 2))
    for index, row in df.iterrows():
        row[source].plot(ax=ax)
    plt.savefig("test_twinning_{}.png".format(title))


def test_twinning(
    reference_with_replicates: SpectraFrame, twinned_with_replicates: SpectraFrame
):

    grouping_cols = [
        col
        for col in reference_with_replicates.columns
        if col not in ["replicate", "file_name", "spectrum"]
    ]
    print(grouping_cols)
    cmp = TwinningComponent(
        twinned_with_replicates,
        reference_with_replicates,
        reference_band_nm=144.0,
        boundaries=(50, 3000),
        grouping_cols=grouping_cols,
    )

    cmp.derive_model()

    plot_spectra(cmp.reference, "reference", source="spectrum")
    plot_spectra(cmp.twinned, "twinned", source="spectrum")

    plot_spectra(cmp.reference, "reference_processed", source="spe_processed")
    plot_spectra(cmp.twinned, "twinned_processed", source="spe_processed")

    cmp.process(cmp.twinned, source="spe_processed", target="spectrum_harmonized")

    cmp.twinned.drop(
        columns=["spectrum", "spe_processed", "spectrum_harmonized"]
    ).to_csv("twinned.csv", index=False)
    cmp.reference.drop(columns=["spectrum", "spe_processed"]).to_csv(
        "reference.csv", index=False
    )

    plot_spectra(cmp.twinned, "spectrum_harmonized", source="spectrum_harmonized")

    print(cmp.twinned[["laser_power_mW", "area", "area_harmonized"]])
    cmp.plot()
    plt.savefig("test_twinning_evaluation.png")
