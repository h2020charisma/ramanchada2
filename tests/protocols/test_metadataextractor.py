import pytest
from ramanchada2.auxiliary.spectra.datasets2 import filtered_df, prepend_prefix
from ramanchada2.protocols.spectraframe import SpectraFrame


@pytest.fixture(scope="module")
def spectra2test():
    df_ref = filtered_df(
        sample=["TiO2"], device=["BWtek"], provider=["FNMT-B"], laser_wl=["785"]
    )
    df_ref.loc[:, "filename"] = prepend_prefix(df_ref["filename"])
    df_ref.loc[:, "id"] = df_ref[["provider", "device", "laser_wl"]].agg(
        " ".join, axis=1
    )
    return df_ref


def map_frame(myframe: SpectraFrame):
    print(myframe.columns)


def test_spectraframe(spectra2test):
    dynamic_column_mapping = {
        "spectrum": "spectrum",
        "filename": "file_name",
        "provider": "provider",
        "device": "device",
        "laser_wl": "laser_wl",
        "OP": "optical_path",
        "laser_power_mW": "laser_power_mW",
        "laser_power_percent": "laser_power_percent",
        "time_ms": "time_ms",
        "replicate": "replicate",
        "id": "device_id",
    }
    spe_frame = SpectraFrame.from_dataframe(spectra2test, dynamic_column_mapping)
    assert "optical_path" in spe_frame.columns
    assert "file_name" in spe_frame.columns
