import numpy as np
from pydantic import validate_arguments

from ..spectrum import Spectrum
from ramanchada2.misc.spectrum_deco import add_spectrum_method


@add_spectrum_method
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def central_moments(spe: Spectrum, /,
                    left_idx=0, right_idx=-1, moments=[1, 2, 3, 4], normalize=False
                    ):
    mom = dict()
    x = spe.x[left_idx:right_idx]
    p = spe.y[left_idx:right_idx]
    p -= p.min()
    p /= p.sum()
    mom[1] = np.sum(x*p)
    mom[2] = np.sum((x - mom[1])**2 * p)
    for i in moments:
        if i <= 2:
            continue
        mom[i] = np.sum((x - mom[1])**i * p)
        if normalize and i > 2:
            mom[i] /= mom[2] ** i/2
    return [mom[i] for i in moments]
