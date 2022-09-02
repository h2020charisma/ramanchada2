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
                                 max_iter: PositiveInt = None,
                                 random_state=None
                                 ) -> PeakCandidatesGroupModel:
    samp = spe.gen_samples(size=10000)
    X = [[i] for i in samp]
    bgm = BayesianGaussianMixture(n_components=n_components,
                                  random_state=random_state,
                                  max_iter=max_iter
                                  ).fit(X)
    res = [[mean[0], np.sqrt(cov[0][0]), weight]
           for mean, cov, weight in zip(bgm.means_, bgm.covariances_, bgm.weights_)]
    res = sorted(res, key=lambda x: x[2], reverse=True)
    pd.DataFrame(res, columns=("mean", "sigma", "weight"))

    return PeakCandidatesGroupModel.from_find_peaks_bayesian_gaussian_mixture(
        means, sigmas, weights, x_arr=spe.x, y_arr=spe.y
        ).group_neighbours(n_sigma=n_sigma_group)
