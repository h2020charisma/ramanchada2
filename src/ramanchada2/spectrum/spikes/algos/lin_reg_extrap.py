from typing import Union

import numpy as np
from pydantic import validate_call


# lr_n3o1 equivalent
# Remove?
def metric(y):
    le = [(4*y[i-1] + y[i-2] - 2*y[i-3])/3 for i in range(3, len(y)-3)]
    ri = [(4*y[i+1] + y[i+2] - 2*y[i+3])/3 for i in range(3, len(y)-3)]
    le = np.pad(le, (3, 3), mode='edge')
    ri = np.pad(ri, (3, 3), mode='edge')
    max = np.max([le, ri], axis=0)
    min = np.min([le, ri], axis=0)
    metric = np.max([y-max, min-y], axis=0)
    metric[:2] = 0
    metric[-2:] = 0
    return metric


@validate_call()
def indices(s, threshold: Union[None, float] = None):
    if threshold is None:
        threshold = 100
    return np.where(metric(s) > threshold)[0]


def lr_extrap_n2_l1(y):
    return np.pad(
        2*y[1:-3]-y[:-4],
        (2, 2), mode='edge')


def lr_extrap_n2_l1_(y):
    return np.pad(
        [2*y[i-1]-y[i-2] for i in range(2, len(y)-2)],
        (2, 2), mode='edge')


def lr_extrap_n2_r1(y):
    return np.pad(
        2*y[3:-1]-y[4:],
        (2, 2), mode='edge')


def lr_extrap_n2_r1_(y):
    return np.pad(
        [2*y[i+1]-y[i+2] for i in range(2, len(y)-2)],
        (2, 2), mode='edge')


def lr_extrap_n2_l2(y):
    return np.pad(
        3*y[1: -5] - 2*y[0:-6],
        (3, 3), mode='edge')


def lr_extrap_n2_l2_(y):
    return np.pad(
        [3*y[i-2]-2*y[i-3] for i in range(3, len(y)-3)],
        (3, 3), mode='edge')


def lr_extrap_n2_r2(y):
    return np.pad(
        3*y[5: -1] - 2*y[6:],
        (3, 3), mode='edge')


def lr_extrap_n2_r2_(y):
    return np.pad(
        [3*y[i+2]-2*y[i+3] for i in range(3, len(y)-3)],
        (3, 3), mode='edge')


def lr_extrap_n3_l1(y):
    return np.pad(
        (4*y[2:-4] + y[1:-5] - 2*y[:-6])/3,
        (3, 3), mode='edge')


def lr_extrap_n3_l1_(y):
    return np.pad(
        [(4*y[i-1]+y[i-2]-2*y[i-3])/3 for i in range(3, len(y)-3)],
        (3, 3), mode='edge')


def lr_extrap_n3_r1(y):
    return np.pad(
        (4*y[4:-2]+y[5:-1]-2*y[6:])/3,
        (3, 3), mode='edge')


def lr_extrap_n3_r1_(y):
    return np.pad(
        [(4*y[i+1]+y[i+2]-2*y[i+3])/3 for i in range(3, len(y)-3)],
        (3, 3), mode='edge')


def lr_extrap_n3_l2(y):
    return np.pad(
        (11*y[2:-6]+2*y[1:-7]-7*y[:-8])/6,
        (4, 4), mode='edge')


def lr_extrap_n3_l2_(y):
    return np.pad(
        [(11*y[i-2]+2*y[i-3]-7*y[i-4])/6 for i in range(4, len(y)-4)],
        (4, 4), mode='edge')


def lr_extrap_n3_l2_orig_bad(y):
    return np.pad(
        [(11*y[i-1]+2*y[i-2]-7*y[i-3])/6 for i in range(4, len(y)-4)],
        (4, 4), mode='edge')


def lr_extrap_n3_r2(y):
    return np.pad(
        (11*y[6:-2]+2*y[7:-1]-7*y[8:])/6,
        (4, 4), mode='edge')


def lr_extrap_n3_r2_(y):
    return np.pad(
        [(11*y[i+2]+2*y[i+3]-7*y[i+4])/6 for i in range(4, len(y)-4)],
        (4, 4), mode='edge')


def lr_extrap_n3_r2_orig_bad(y):
    return np.pad(
        [(11*y[i+1]+2*y[i+2]-7*y[i+3])/6 for i in range(4, len(y)-4)],
        (4, 4), mode='edge')
