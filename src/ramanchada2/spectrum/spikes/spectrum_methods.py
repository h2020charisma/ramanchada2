from pydantic import validate_call

from ramanchada2.misc.spectrum_deco import (add_spectrum_filter,
                                            add_spectrum_method)

from ..spectrum import Spectrum
from .spikes import add_spike as fn_add_spike
from .spikes import spikes_drop as fn_spikes_drop
from .spikes import spikes_fix_interp as fn_spikes_fix_interp
from .spikes import spikes_indices as fn_spikes_indices
from .spikes import spikes_metric as fn_spikes_metric
from .spikes import spikes_only as fn_spikes_only


@add_spectrum_method
@validate_call(config=dict(arbitrary_types_allowed=True))
def spikes_metric(spe: Spectrum, /,
                  method: str,
                  ):
    return fn_spikes_metric(spe.y, method=method)


@add_spectrum_method
@validate_call(config=dict(arbitrary_types_allowed=True))
def spikes_indices(spe: Spectrum, /,
                   method: str,
                   **kwargs):
    return fn_spikes_indices(spe.y, method=method, **kwargs)


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def spikes_drop(old_spe: Spectrum,
                new_spe: Spectrum, /,
                method: str,
                **kwargs):
    new_spe.x, new_spe.y = fn_spikes_drop(old_spe.x, old_spe.y, method=method, **kwargs)


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def spikes_only(old_spe: Spectrum,
                new_spe: Spectrum, /,
                method: str,  # type: ignore [valid-type]
                **kwargs):
    new_spe.x, new_spe.y = fn_spikes_only(old_spe.x, old_spe.y, method=method, **kwargs)


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def add_spike(old_spe: Spectrum,
              new_spe: Spectrum, /,
              location: float,
              amplitude: float,
              ):
    new_spe.y = fn_add_spike(old_spe.x, old_spe.y, location=location, amplitude=amplitude)


@add_spectrum_filter
@validate_call(config=dict(arbitrary_types_allowed=True))
def spikes_fix_interp(old_spe: Spectrum,
                      new_spe: Spectrum, /,
                      method: str,
                      kind='makima',
                      **kwargs):
    new_spe.y = fn_spikes_fix_interp(old_spe.x, old_spe.y, method=method, **kwargs)
