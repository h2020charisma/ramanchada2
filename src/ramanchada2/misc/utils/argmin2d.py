from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt
from pydantic import PositiveInt, validate_call
from scipy import linalg


def argmin2d(A, median_limit: Optional[float] = None):
    if median_limit is None:
        median_limit = 10
    ymin_idx = np.argmin(A, axis=0)
    xmin_idx = np.argmin(A, axis=1)
    x_idx = np.unique(xmin_idx[xmin_idx[ymin_idx[xmin_idx]] == xmin_idx])
    y_idx = np.unique(ymin_idx[ymin_idx[xmin_idx[ymin_idx]] == ymin_idx])
    dist = A[y_idx, x_idx]
    filt = dist <= np.median(dist) * median_limit
    x_idx = x_idx[filt]
    y_idx = y_idx[filt]
    matches = np.stack([y_idx, x_idx]).T
    return matches


def find_closest_pairs_idx(x, y, **kw_args):
    outer_dif = np.abs(np.subtract.outer(x, y))
    return argmin2d(outer_dif, **kw_args).T


def find_closest_pairs(x, y, **kw_args):
    x_idx, y_idx = find_closest_pairs_idx(x, y, **kw_args)
    return x[x_idx], y[y_idx]


@validate_call(config=dict(arbitrary_types_allowed=True))
def align(x, y,
          p0: Union[List[float], npt.NDArray] = [0, 1, 0, 0],
          func=lambda x, a0, a1, a2, a3: (a0*np.ones_like(x), a1*x, a2*x**2/1, a3*(x/1000)**3),
          max_iter: PositiveInt = 1000,
          **kw_args):
    """
    Iteratively finds best match between x and y and evaluates the x scaling parameters.

    Finds best parameters *p that minimise L2 distance between scaled x and original y
    min((lambda(x, *p)-y)**2 | *p)

    Args:
        x (ArrayLike[float]): values that need to match the reference
        y (ArrayLike[float]): reference values
        p0 (Union[List[float], npt.NDArray], optional): initial values for the parameters `p`.
            Defaults to [0, 1, 0, 0].
        func (Callable, optional): Objective function to minimize. Returns list penalties
            calculated for each `p`. The total objective function is sum of the elements.
            Defaults to polynomial of 3-th degree.
        max_iter (PositiveInt, optional): max number of iterations. Defaults to 1000.

    Returns:
        ArrayLike[float]: array of parameters `p` that minimize the objective funciton
    """

    if isinstance(p0, list):
        p = np.array(p0)
    else:
        p = p0
    loss = np.inf
    cur_x = x
    for it in range(max_iter):
        cur_x = np.sum(func(x, *p), axis=0)
        x_idx, y_idx = find_closest_pairs_idx(cur_x, y, **kw_args)
        x_match, y_match = x[x_idx], y[y_idx]
        p_bak = p
        obj_mat = np.stack(func(x_match, *np.ones_like(p)), axis=1)
        p, *_ = linalg.lstsq(obj_mat, y_match, cond=1e-8)
        loss_bak = loss
        loss = np.sum((x_match-y_match)**2)/len(x_match)**2
        if np.allclose(p, p_bak):
            break
        if loss > loss_bak:
            pass
            return p_bak
    return p


@validate_call(config=dict(arbitrary_types_allowed=True))
def align_shift(x, y,
                p0: float = 0,
                max_iter: PositiveInt = 1000,
                **kw_args):
    loss = np.inf
    cur_x = x
    p = p0
    for it in range(max_iter):
        cur_x = x + p
        x_idx, y_idx = find_closest_pairs_idx(cur_x, y, **kw_args)
        x_match, y_match = x[x_idx], y[y_idx]
        p_bak = p
        p = np.mean(y_match-x_match)
        loss_bak = loss
        loss = np.sum((y_match-x_match)**2)
        if np.allclose(p, p_bak):
            break
        if loss > loss_bak:
            return p_bak
    return p
