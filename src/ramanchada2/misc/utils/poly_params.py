import numpy as np


def linreg(x, y):
    idx = np.isfinite(x) & np.isfinite(y)
    x = x[idx]
    y = y[idx]
    s_xy = np.sum(x*y)
    s_x = np.sum(x)
    s_xx = np.sum(x**2)
    s_y = np.sum(y)
    s = len(x)
    d = s*s_xx-s_x**2
    a = (s*s_xy - s_x*s_y)/d
    b = (s_xx*s_y - s_x*s_xy)/d
    return a, b


def line_by_points(x, y):
    a = (y[1]-y[0])/(x[1]-x[0])
    b = y[0] - a*x[0]
    return a, b


def poly2_by_points(x, y):
    """
    Calculate poly-2 coefficients by 3 points.

    >>> x = [1, 2, 4]
    >>> y = [1, 0, 4]
    >>> a, b, c = poly2_by_points(x, y)
    >>> xi = np.linspace(0, 4, 30)
    >>> xt = np.array([1, 2, 3, 4])
    >>> yt = np.array([1, 0, 1, 4])
    >>> assert np.allclose(a*xt**2 + b*xt + c, yt)
    """
    a = np.array([np.square(x), x, np.ones_like(x)]).T
    a_1 = np.linalg.inv(a)
    return a_1@y
