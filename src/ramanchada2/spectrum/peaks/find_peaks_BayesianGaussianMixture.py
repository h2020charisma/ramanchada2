#!/usr/bin/env python3

from pydantic import validate_arguments, PositiveFloat, PositiveInt
from sklearn.mixture import BayesianGaussianMixture
import pandas as pd
import numpy as np


from ..spectrum import Spectrum
from ramanchada2.misc.spectrum_deco import (add_spectrum_method,
                                            add_spectrum_filter)
from ramanchada2.misc.types import PeakCandidatesGroupModel


@add_spectrum_method
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def find_peaks_bayesian_gaussian(spe: Spectrum, /,
                                 n_samples: PositiveInt = 5000,
                                 n_components: PositiveInt = 50,
                                 n_sigma_group: PositiveFloat = 5.,
                                 max_iter: PositiveInt = 100,
                                 moving_minimum_window: PositiveInt = None,
                                 random_state=None,
                                 trim_range=[-np.infty, np.infty],
                                 ) -> PeakCandidatesGroupModel:
    if moving_minimum_window is not None:
        spe = spe.subtract_moving_minimum(moving_minimum_window)  # type: ignore
    spe = spe.normalize()  # type: ignore
    #for auto segmenting use predict / predict_proba
    samp = spe.gen_samples(size=10000, trim_range=trim_range)
    X = [[i] for i in samp]
    bgm = BayesianGaussianMixture(n_components=n_components,
                                  random_state=random_state,
                                  max_iter=max_iter
                                  ).fit(X)

    return PeakCandidatesGroupModel.from_find_peaks_bayesian_gaussian_mixture(
        bgm, x_arr=spe.x, y_arr=spe.y
        ).group_neighbours(n_sigma=n_sigma_group)
