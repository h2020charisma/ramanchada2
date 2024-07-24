#!/usr/bin/env python3

import os

import pytest

test_data_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def crystal_simulation_out_file():
    with open(test_data_dir + '/data/simulations/crystal/Anatase_PBE_pob_TZVP_Raman_intens.out') as f:
        yield f


@pytest.fixture
def crystal_simulation_convolved_dat_file():
    with open(test_data_dir + '/data/simulations/crystal/calcite_CRYSTAL_PBE_convoluted.dat') as f:
        yield f


@pytest.fixture
def crystal_simulation_raw_dat_file():
    with open(test_data_dir + '/data/simulations/crystal/calcite_CRYSTAL_PBE_raw_data.dat') as f:
        yield f


@pytest.fixture
def vasp_simulation_dat_file():
    with open(test_data_dir + '/data/simulations/vasp/snCAL_vasp_raman_ALL.dat') as f:
        yield f


@pytest.fixture
def rruf_experimental_filename():
    return (
        test_data_dir +
        '/data/experimental/rruf/Anatase__R060277__Broad_Scan____0__unoriented__Raman_Data_RAW__21346.txt'
    )


@pytest.fixture
def opus_experimental_file():
    return test_data_dir + '/data/experimental/test_opus.0'
