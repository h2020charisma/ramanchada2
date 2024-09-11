from typing import Optional, Tuple, Union

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
                              trim_range: Optional[Tuple[float, float]] = None,
                              ) -> BayesianGaussianMixture:
    """
    Decompose the spectrum to Bayesian Gaussian Mixture

    Args:
        spe: internal use only
        n_samples: optional. Defaults to 5000.
            Resampled dataset size
        n_components: optional. Defaults to 50.
            Number of expected gaussian components
        max_iter: optional. Defaults to 100.
            Maximal number of iterations.
        moving_minimum_window: optional. Defaults to None.
            If None no moving minimum is subtracted, otherwise as specified.
        random_state: optional. Defaults to None.
            Random generator seed to be used.
        trim_range: optional. Defaults to None:
            If None ignore trimming, otherwise trim range is in x-axis values.

    Returns:
        BayesianGaussianMixture: Fitted Bayesian Gaussian Mixture
    """
    if moving_minimum_window is not None:
        spe = spe.subtract_moving_minimum(moving_minimum_window)  # type: ignore
    samp = spe.gen_samples(size=n_samples, trim_range=trim_range)
    X = [[i] for i in samp]
    bgm = BayesianGaussianMixture(n_components=n_components,
                                  random_state=random_state,
                                  max_iter=max_iter
                                  ).fit(X)
    return bgm
