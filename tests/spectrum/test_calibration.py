#!/usr/bin/env python3

import numpy as np
import pytest
import ramanchada2 as rc2


def test_scale_xaxis_fun():
    spe = rc2.spectrum.from_delta_lines(deltas={100: 100, 1000: 1000})
    with pytest.raises(ValueError):
        spe.scale_xaxis_fun(lambda x: (x-spe.x.mean())**2)

    spe1 = spe.scale_xaxis_fun(lambda x: x+100)
    assert np.allclose(spe.x, spe1.x-100)

    spe2 = spe.scale_xaxis_fun(lambda x: x**2+100)
    assert np.allclose(spe2.x, spe.x*spe.x+100)
