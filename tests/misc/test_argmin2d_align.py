import numpy as np

from ramanchada2.misc.utils.argmin2d import align


def test_align():
    x = np.array([-300, -200, -150, -100, 10, 20, 80, 120, 250, 270, 300, 330])
    y = -.00001*x**2 + .9999*x - 4
    x_idx = np.array([0,  1,  3,  5,  6,  7,  8,  9, 10])
    y_idx = np.array([1,  2,  4,  5,  6,  8,  9, 10, 11])

    def fn(x, a0, a1, a2):
        return a0*np.ones_like(x), a1*x, a2*x**2
    cal = align(y[y_idx], x[x_idx], p0=[0, 1, 0], func=fn)
    recal = np.sum(fn(y, *cal), axis=0)
    assert np.allclose(x, recal, rtol=1e-3)
