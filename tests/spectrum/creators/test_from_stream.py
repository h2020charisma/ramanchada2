import os

import pytest

import ramanchada2 as rc2


def test_from_stream(experimental_filename):
    ft = os.path.splitext(experimental_filename)[1][1:]
    if ft in {'l6s', 'wxd'}:
        print('skip', ft)
        return
    print(ft)
    with open(experimental_filename, 'rb') as fp:
        spe = rc2.spectrum.from_stream(fp, filetype=ft, backend='rc1_parser')
        assert len(spe.x) > 10

    if ft not in {'wdf'}:
        with open(experimental_filename, 'rb') as fp:
            spe = rc2.spectrum.from_stream(fp, filetype=ft, backend='native')
            assert len(spe.x) > 10

        if ft in {'spc'}:
            with pytest.raises(UnicodeDecodeError):
                with open(experimental_filename, 'r') as fp:
                    rc2.spectrum.from_stream(fp, filetype=ft)
        else:
            with open(experimental_filename, 'r') as fp:
                rc2.spectrum.from_stream(fp, filetype=ft)
