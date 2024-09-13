import numpy as np
from numpy.typing import NDArray
from pydantic import PositiveFloat, validate_call

from ramanchada2.misc.spectrum_deco import (add_spectrum_filter,
                                            add_spectrum_method)

from ..spectrum import Spectrum


@add_spectrum_method
@validate_call(config=dict(arbitrary_types_allowed=True))
def spike_indices(spe: Spectrum, /, n_sigma: PositiveFloat) -> NDArray:
    """
    Find spikes in spectrum

    Single-bin spikes are located using left and right successive
    differences. The threshold is based on the standart deviation
    of the metric which makes this algorithm less optimal.

    Args:
        spe: internal use only
        n_sigma: Threshold value should be `n_sigma` times the standart
          deviation of the metric.

    Returns: List of spike indices
    """
    yi = spe.y[1:-1]
    yi_1 = spe.y[:-2]
    yi1 = spe.y[2:]
    y_merit = np.abs(2*yi-yi_1-yi1) - np.abs(yi1-yi_1)
    spike_idx = y_merit > n_sigma * y_merit.std()
    spike_idx = np.concatenate(([False], spike_idx, [False]))
    return spike_idx


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def drop_spikes(old_spe: Spectrum,
                new_spe: Spectrum, /,
                n_sigma: PositiveFloat = 10):
    """
    Removes single-bin spikes.

    Remove x, y pairs recognised as spikes using left and right
    successive differences and standard-deviation-based threshold.

    Args:
        old_spe: internal use only
        new_spe: internal use only
        n_sigma: optional, default is `10`.
            Threshold is `n_sigma` times the standard deviation.

    Returns: modified Spectrum
    """
    use_idx = ~spike_indices(old_spe, n_sigma=n_sigma)
    new_spe.x = old_spe.x[use_idx]
    new_spe.y = old_spe.y[use_idx]


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def recover_spikes(old_spe: Spectrum,
                   new_spe: Spectrum, /,
                   n_sigma: PositiveFloat = 10):
    """
    Recover single-bin spikes.

    Recover x, y pairs recognised as spikes using left and right
    successive differences and standard-deviation-based threshold
    and linear interpolation.

    Args:
        old_spe: internal use only
        new_spe: internal use only
        n_sigma: optional, default is `10`.
            Threshold is `n_sigma` times the standard deviation.

    Returns: modified Spectrum
    """
    use_idx = ~spike_indices(old_spe, n_sigma=n_sigma)
    new_spe.y = np.interp(old_spe.x, old_spe.x[use_idx], old_spe.y[use_idx])


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def get_spikes(old_spe: Spectrum,
               new_spe: Spectrum, /,
               n_sigma: PositiveFloat = 10):
    """
    Get single-bin spikes only.

    Get x, y pairs recognised as spikes using left and right
    successive differences and standard-deviation-based threshold
    and linear interpolation.

    Args:
        old_spe: internal use only
        new_spe: internal use only
        n_sigma: optional, default is `10`.
            Threshold is `n_sigma` times the standard deviation.

    Returns: modified Spectrum
    """
    spike_idx = spike_indices(old_spe, n_sigma=n_sigma)
    new_spe.x = old_spe.x[spike_idx]
    new_spe.y = old_spe.y[spike_idx]
