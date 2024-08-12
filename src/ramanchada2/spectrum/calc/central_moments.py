import numpy as np
from pydantic import validate_call

from ramanchada2.misc.spectrum_deco import add_spectrum_method

from ..spectrum import Spectrum


@add_spectrum_method
@validate_call(config=dict(arbitrary_types_allowed=True))
def central_moments(spe: Spectrum, /,
                    boundaries=(-np.inf, np.inf), moments=[1, 2, 3, 4], normalize=False
                    ):
    mom = dict()
    filter_idx = (spe.x >= boundaries[0]) & (spe.x < boundaries[1])
    x = spe.x[filter_idx]
    p = spe.y[filter_idx]
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
