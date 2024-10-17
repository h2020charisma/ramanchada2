from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from ramanchada2.protocols.metadata_helper import SpectrumMetadataExtractor
from ramanchada2.spectrum import Spectrum


class SpectraFrameSchema(BaseModel):
    file_name: Optional[str] = None
    sample: Optional[str] = None
    provider: Optional[str] = None
    device: Optional[str] = None
    device_id: Optional[str] = None
    laser_wl: int
    laser_power_mW: Optional[float] = None
    laser_power_percent: Optional[float] = None
    time_ms: Optional[float] = None
    replicate: Optional[int] = None
    optical_path: Optional[str] = None
    spectrum: Optional[Spectrum] = None

    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types


# Define a custom DataFrame class with dynamic column mapping and validation
class SpectraFrame(pd.DataFrame):

    @classmethod
    def validate_columns(cls, df: pd.DataFrame, column_mapping: Dict[str, str]):
        """
        Validate a DataFrame against the schema with dynamic column mapping.

        Args:
            df (pd.DataFrame): The DataFrame to validate.
            column_mapping (Dict[str, str]): A mapping from expected column names (in the schema)
                                             to actual column names in the DataFrame.
        """
        # Rename DataFrame columns according to the provided mapping
        df_mapped = df.rename(columns=column_mapping)
        # Validate each row against the schema
        errors = []
        for index, row in df_mapped.iterrows():
            try:
                # Convert row to dictionary and validate using Pydantic schema
                SpectraFrameSchema(**row.to_dict())
            except ValidationError as e:
                errors.append((index, e.errors()))

        if errors:
            for idx, err in errors:
                print(f"Row {idx} has errors: {err}")

            raise ValueError("DataFrame validation failed")
        else:
            print("DataFrame validation passed!")
        return df_mapped

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, column_mapping: Dict[str, str]):
        """
        Create an instance of SpectraFrame with dynamic column validation.

        Args:
            df (pd.DataFrame): The input DataFrame.
            column_mapping (Dict[str, str]): The dynamic mapping for column names.

        Returns:
            SpectraFrame: A validated SpectraFrame object.
        """
        if column_mapping is None:
            column_mapping = {}
        # Validate columns before creating the MyFrame
        df_mapped = cls.validate_columns(df, column_mapping)
        return cls(df_mapped)

    @classmethod
    def from_metadata(
        cls, spectra: List[Spectrum], metadata_extractor: SpectrumMetadataExtractor
    ):
        data = []
        for spectrum in spectra:
            metadata = metadata_extractor.extract(spectrum, None)
            data.append({"spectrum": spectrum, **metadata})
        return cls(pd.DataFrame(data))

    @classmethod
    def from_template(
        cls, template_file: str, metadata_extractor: SpectrumMetadataExtractor
    ):
        return

    def average(
        self,
        grouping_cols=[
            "sample",
            "provider",
            "device",
            "laser_wl",
            "laser_power_percent",
            "laser_power_mW",
            "time_ms",
        ],
        source="spectrum",
        target="spectrum",
    ):
        processed_rows = []

        for group_keys, group in self.groupby(grouping_cols):
            # Iterate over each row in the group
            spe_average = None
            for index, row in group.iterrows():
                if spe_average is None:
                    spe_average = row[source]
                else:
                    spe_average = spe_average + row[source]
            spe_average = spe_average / group.shape[0]

            processed_row = row.copy()[grouping_cols]  # Make a copy of the row
            processed_row[target] = spe_average
            processed_rows.append(processed_row)

        df = pd.DataFrame(processed_rows)
        df.sort_values(by="laser_power_percent")
        return SpectraFrame.from_dataframe(df, column_mapping={})

    # tbd make it more generic
    def trim(self, source="spectrum", target="spectrum", **kwargs):
        kwargs.setdefault("method", "x-axis")
        kwargs.setdefault("boundaries", (50, 4000))
        for index, row in self.iterrows():
            self.at[index, target] = row[source].trim_axes(**kwargs)
        return self

    def baseline_snip(self, source="spectrum", target="spectrum", **kwargs):
        kwargs.setdefault("niter", 40)
        for index, row in self.iterrows():
            self.at[index, target] = row[source].subtract_baseline_rc1_snip(**kwargs)
        return self

    def spe_area(self, boundaries=(50, 3000), source="spectrum", target="area"):
        for index, row in self.iterrows():
            spe = row[source]
            sc = spe.trim_axes(method="x-axis", boundaries=boundaries)
            self.at[index, target] = np.sum(sc.y * np.diff(sc.x_bin_boundaries))

    def multiply(self, multiplier: float, source="spectrum", target="spectrum"):
        for index, row in self.iterrows():
            self.at[index, target] = row[source] * multiplier
