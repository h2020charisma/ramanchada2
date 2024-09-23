from pydantic import BaseModel, ValidationError 
from typing import Dict, Optional
import pandas as pd


class SpectraFrameSchema(BaseModel):
    file_name: Optional[str] = None
    provider: Optional[str] = None
    device: Optional[str] = None
    device_id: Optional[str] = None
    laser_wl: int    
    laser_power_mW: Optional[float] = None
    laser_power_percent: Optional[float] = None
    time_ms: Optional[float] = None
    replicate: Optional[int] = None
    optical_path: Optional[str] = None
    spectrum: Optional[str] = None

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
        # Validate columns before creating the MyFrame
        return cls.validate_columns(df, column_mapping)
