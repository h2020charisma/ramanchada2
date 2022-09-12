#!/usr/bin/env python3

from pydantic import validate_arguments, PositiveFloat, PositiveInt
from sklearn.mixture import BayesianGaussianMixture

from ..spectrum import Spectrum
from ramanchada2.misc.spectrum_deco import add_spectrum_method
from ramanchada2.misc.types import PeakCandidatesGroupModel


@add_spectrum_method
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def bayesian_gaussian_mixture(spe: Spectrum, /,
                              n_samples: PositiveInt = 5000,
                              n_components: PositiveInt = 50,
                              max_iter: PositiveInt = 100,
                              moving_minimum_window: PositiveInt = None,
                              random_state=None,
                              trim_range=None,
                              ) -> PeakCandidatesGroupModel:
    if moving_minimum_window is not None:
        spe = spe.subtract_moving_minimum(moving_minimum_window)  # type: ignore
    samp = spe.gen_samples(size=n_samples, trim_range=trim_range)
    X = [[i] for i in samp]
    bgm = BayesianGaussianMixture(n_components=n_components,
                                  random_state=random_state,
                                  max_iter=max_iter
                                  ).fit(X)
    return bgm


@add_spectrum_method
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def find_peaks_bayesian_gaussian(spe: Spectrum, /,
                                 n_samples: PositiveInt = 5000,
                                 n_components: PositiveInt = 50,
                                 n_sigma_group: PositiveFloat = 5.,
                                 max_iter: PositiveInt = 100,
                                 moving_minimum_window: PositiveInt = None,
                                 random_state=None,
                                 trim_range=None,
                                 ) -> PeakCandidatesGroupModel:
    bgm = spe.bayesian_gaussian_mixture(n_samples=n_samples,  # type: ignore
                                        n_components=n_components,
                                        max_iter=max_iter,
                                        moving_minimum_window=moving_minimum_window,
                                        random_state=random_state,
                                        trim_range=trim_range)

    x = spe.x
    y = spe.y
    if trim_range is not None:
        trims = (x > trim_range[0]) & (x < trim_range[1])
        x = x[trims]
        y = y[trims]
    return PeakCandidatesGroupModel.from_find_peaks_bayesian_gaussian_mixture(
        bgm, x_arr=x, y_arr=y
        ).group_neighbours(n_sigma=n_sigma_group)
