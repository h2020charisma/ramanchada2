import numpy as np

import ramanchada2 as rc2
from ramanchada2.spectrum.spikes.spikes import add_spike


def test_add_spike():
    x = np.linspace(-1, 1, 41)
    y = np.zeros_like(x)
    y1 = np.zeros_like(x)
    y1[-2] += -2
    assert np.allclose(add_spike(x, y, .95, -2), y1)

    x = np.linspace(-1, 1, 41)
    y = np.zeros_like(x)
    y1 = np.zeros_like(x)
    y1[0] += 2
    assert np.allclose(add_spike(x, y, -1, 2), y1)

    x = np.linspace(-1, 1, 41)
    y = np.zeros_like(x)
    y1 = np.zeros_like(x)
    y1[0] += 1
    y1[1] += 1
    assert np.allclose(add_spike(x, y, -.975, 2), y1)

    x = np.linspace(-1, 1, 41)
    y = np.zeros_like(x)
    y1 = np.zeros_like(x)
    y1[0] += 4
    y1[1] += 1
    assert np.allclose(add_spike(x, y, -.99, 5), y1)

    x = np.linspace(-1, 1, 41)
    y = np.zeros_like(x)
    y1 = np.zeros_like(x)
    y1[0] += -1
    y1[1] += -4
    assert np.allclose(add_spike(x, y, -.96, -5), y1)


def test_add_spike_spectrum():
    x = np.linspace(-1, 1, 41)
    spe = rc2.spectrum.from_theoretical_lines(x=x, shapes=[], params=[]).add_spike(-.96, -5)
    y1 = np.zeros_like(x)
    y1[0] += -1
    y1[1] += -4
    assert np.allclose(spe.y, y1)
