from typing import Union

from pydantic import PositiveInt, validate_call
from sklearn.mixture import BayesianGaussianMixture

from ramanchada2.misc.spectrum_deco import add_spectrum_method

from ..spectrum import Spectrum


@add_spectrum_method
@validate_call(config=dict(arbitrary_types_allowed=True))
def bayesian_gaussian_mixture(spe: Spectrum, /,
                              n_samples: PositiveInt = 5000,
                              n_components: PositiveInt = 50,
                              max_iter: PositiveInt = 100,
                              moving_minimum_window: Union[PositiveInt, None] = None,
                              random_state=None,
                              trim_range=None,
                              ):
    if moving_minimum_window is not None:
        spe = spe.subtract_moving_minimum(moving_minimum_window)  # type: ignore
    samp = spe.gen_samples(size=n_samples, trim_range=trim_range)
    X = [[i] for i in samp]
    bgm = BayesianGaussianMixture(n_components=n_components,
                                  random_state=random_state,
                                  max_iter=max_iter
                                  ).fit(X)
    return bgm
