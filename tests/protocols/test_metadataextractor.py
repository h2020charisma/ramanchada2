from ramanchada2.protocols.spectraframe import SpectraFrame
import pandas as pd
from ramanchada2.auxiliary.spectra.datasets2 import (filtered_df,get_filenames,
                                                     prepend_prefix)
import pytest
from ramanchada2.protocols.metadata_helper import TemplateMetadataExtractor,SpectrumMetadataExtractor
from ramanchada2.protocols.metadata_helper import FilenameMetadataExtractor,ChainedMetadataExtractor

@pytest.fixture(scope="module")
def spectra2test():
    df_ref = filtered_df(sample=['TiO2'],device=['BWtek'],provider=['FNMT-B'],laser_wl=['785'])    
    df_ref.loc[:, "filename"] = prepend_prefix(df_ref["filename"])
    df_ref.loc[:, 'id'] = df_ref[['provider', 'device', 'laser_wl']].agg(' '.join, axis=1)
    return df_ref

def map_frame(myframe: SpectraFrame):
    print(myframe.columns)


def test_spectraframe(spectra2test):
    dynamic_column_mapping = {
        'spectrum': 'spectrum',
        'filename': 'file_name',
        'provider': 'provider',
        'device': 'device' ,
        'laser_wl' : 'laser_wl',
        'OP': 'optical_path',
        'laser_power_mW': 'laser_power_mW',
        'laser_power_percent': 'laser_power_percent',
        'time_ms': 'time_ms',
        'replicate': 'replicate',
        'id' : 'device_id'
    }
    spe_frame = SpectraFrame.from_dataframe(spectra2test, dynamic_column_mapping)
    assert("optical_path" in spe_frame.columns)
    assert("file_name" in spe_frame.columns)
    

def test_tbd():

    # Initialize individual metadata extractors
    template_extractor = TemplateMetadataExtractor(template=['name'])
    filename_extractor = FilenameMetadataExtractor()
    spectrum_extractor = SpectrumMetadataExtractor()

    # Chain the metadata extractors
    chained_extractor = ChainedMetadataExtractor(template_extractor, filename_extractor, spectrum_extractor)

    # Create the data table
    data_table_creator = SpectraFrame.from_metadata(spectrum_extractor)
    data_table = data_table_creator.create_datatable()

    print(data_table)        