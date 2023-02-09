import pydantic
import numpy as np
import numpy.typing as npt
from scipy import linalg
from typing import List, Union


def argmin2d(A):
    ymin_idx = np.argmin(A, axis=0)
    xmin_idx = np.argmin(A, axis=1)
    x_idx = np.unique(xmin_idx[xmin_idx[ymin_idx[xmin_idx]] == xmin_idx])
    y_idx = np.unique(ymin_idx[ymin_idx[xmin_idx[ymin_idx]] == ymin_idx])
    matches = np.stack([y_idx, x_idx]).T
    return matches


def find_closest_pairs_idx(x, y):
    outer_dif = np.abs(np.subtract.outer(x, y))
    return argmin2d(outer_dif).T


def find_closest_pairs(x, y):
    x_idx, y_idx = find_closest_pairs_idx(x, y)
    return x[x_idx], y[y_idx]


@pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
def align(x, y,
          p0: Union[List[float], npt.NDArray] = [0, 1, 0, 0],
          func=lambda x, a0, a1, a2, a3: (a0*np.ones_like(x), a1*x, a2*x**2/1, a3*(x/1000)**3),
          max_iter: pydantic.PositiveInt = 1000):
    """
    Iteratively finds best match between x and y and evaluates the x scaling parameters.
    min((lambda(x, *p)-y)**2 | *p)
    Finds best parameters *p that minimise L2 distance between scaled x and original y
    """
    if isinstance(p0, list):
        p = np.array(p0)
    else:
        p = p0
    loss = np.infty
    cur_x = x
    for it in range(max_iter):
        cur_x = np.sum(func(x, *p), axis=0)
        x_idx, y_idx = find_closest_pairs_idx(cur_x, y)
        x_match, y_match = x[x_idx], y[y_idx]
        obj_mat = np.stack(func(x_match, *np.ones_like(p)), axis=1)

        p_bak = p
        loss_bak = loss
        p, *_ = linalg.lstsq(obj_mat, y_match, cond=1e-8)
        loss = np.sum((x_match-y_match)**2)/len(x_match)**2
        if np.allclose(p, p_bak):
            break
        if loss > loss_bak:
            pass
            return p_bak
    return p
