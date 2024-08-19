from typing import Literal, Union

import numpy as np
import pandas as pd
from pydantic import PositiveInt, validate_call
from scipy import sparse
from scipy.signal import wiener
from scipy.sparse.linalg import spsolve

from ramanchada2.misc.spectrum_deco import add_spectrum_filter
from ramanchada2.misc.types import PositiveOddInt

from ..spectrum import Spectrum


@validate_call
def baseline_als(y, lam: float = 1e5, p: float = 0.001, niter: PositiveInt = 100,
                 smooth: Union[PositiveOddInt, Literal[0]] = PositiveOddInt(7)):
    if smooth > 0:
        y = wiener(y, smooth)
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z


def baseline_snip(y0, niter: PositiveInt = 30):
    # y can't have negatives. fix by offset:
    y_offset = y0.min()
    y = y0 - y_offset
    # Spectrum must be row of a DataFrame
    raman_spectra = pd.DataFrame(y).T
    spectrum_points = len(raman_spectra.columns)
    raman_spectra_transformed = np.log(np.log(np.sqrt(raman_spectra + 1) + 1) + 1)
    working_spectra = np.zeros(raman_spectra.shape)
    for pp in np.arange(1, niter+1):
        r1 = raman_spectra_transformed.iloc[:, pp:spectrum_points - pp]
        r2 = (np.roll(raman_spectra_transformed, - pp, axis=1)[:, pp:spectrum_points - pp] +
              np.roll(raman_spectra_transformed, pp, axis=1)[:, pp:spectrum_points - pp])/2
        working_spectra = np.minimum(r1, r2)
        raman_spectra_transformed.iloc[:, pp:spectrum_points-pp] = working_spectra
    baseline = (np.exp(np.exp(raman_spectra_transformed)-1)-1)**2 - 1
    # Re-convert to np.array and apply inverse y offset to baseline
    return baseline.to_numpy()[0].T + y_offset


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def subtract_baseline_rc1_als(
        old_spe: Spectrum,
        new_spe: Spectrum,
        lam=1e5, p=0.001, niter=100, smooth=7
        ):
    new_spe.y = old_spe.y - baseline_als(old_spe.y, lam=lam, p=p, niter=niter, smooth=smooth)


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def subtract_baseline_rc1_snip(
        old_spe: Spectrum,
        new_spe: Spectrum,
        niter=30
        ):
    new_spe.y = old_spe.y - baseline_snip(old_spe.y, niter=niter)
