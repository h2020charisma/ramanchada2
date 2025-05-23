#!/usr/bin/env python3

import datetime

import numpy as np
import pytest

from ramanchada2.misc.types.spectrum.metadata import SpeMetadataModel


def test_spectrum_metadata():
    test = {'_3': 3,
            '_3_0': 3.0,
            '_3_1': 3.1,
            's_3': '3',
            's_3_0': '3.0',
            's_3_1': '3.1',
            'datetime': '2022-01-02T10:11:12+02:00',
            'nparr': np.arange(20),
            'll': '''['kk', {"a": 1, 'b': 'dfdf'}, 1,2,3,'234', [1,2,3]]''',
            'dd': '''{"kk": {"a": 1, 'b': 'dfdf'}, 'i':1, 's':'234', 'l':[1,2,3]}'''
            }

    s = SpeMetadataModel.model_validate(test)
    assert len(s.root) == 10
    s._update({'zz': '2022-01-02T10:11:12+00:00'})
    assert len(s.root) == 11
    ser = s.serialize()

    assert s['zz']-s['datetime'] == datetime.timedelta(seconds=7200)
    assert ser['zz'] == '2022-01-02T10:11:12+00:00'
    s._del_key('zz')
    with pytest.raises(KeyError):
        s['zz']

    assert isinstance(s['ll'], list)
    assert isinstance(ser['ll'], str)
    assert isinstance(s['dd'], dict)
    assert isinstance(ser['dd'], str)

    assert isinstance(ser['nparr'], np.ndarray)
    assert (ser['nparr'] == np.arange(20)).all()

    assert isinstance(s['_3'], int)
    assert s['_3'] == 3
    assert s['_3_0'] == 3
    assert s['_3_1'] == 3.1
    assert s['s_3_0'] == 3
    assert s['s_3_1'] == 3.1

    s._flush()
    assert s.serialize() == {}
    assert len(s.root) == 0


def test_metadata2():
    assert (SpeMetadataModel.validate(['a0', 'b1', 'c2'])
            == SpeMetadataModel.validate({'0': 'a0', '1': 'b1', '2': 'c2'}))
    assert SpeMetadataModel.validate(None) == SpeMetadataModel.validate({})
    assert SpeMetadataModel.validate('') == SpeMetadataModel.validate({})

    meta0 = {'array1': np.array([1, 2, 3, 4, 5, 6]),
             'array2': np.array([1, 2, 3, 4, 5, 6.]),
             'list1': [1, 2, 3, '5', 'asdfasdf'],
             'dict1': {'1': 123,
                       '2': '234',
                       '': 2,
                       '3': None,
                       '4': [1, 2, 3, 4, 99],
                       '5': {'a': 'a', '5': 5, '[]': [], '{}': {}},
                       },
             b'asdf': 'asdf'
             }
    meta_ser = SpeMetadataModel(meta0).serialize()
    assert SpeMetadataModel(meta0) == SpeMetadataModel(meta_ser)
