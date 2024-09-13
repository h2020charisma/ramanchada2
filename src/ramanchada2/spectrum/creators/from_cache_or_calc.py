import logging
from typing import Optional

from pydantic import validate_call

import ramanchada2.misc.types.spectrum as spe_t
from ramanchada2.misc.spectrum_deco import add_spectrum_constructor

from ..spectrum import Spectrum

logger = logging.getLogger(__name__)


@add_spectrum_constructor(set_applied_processing=False)
@validate_call
def from_cache_or_calc(required_steps: spe_t.SpeProcessingListModel,
                       cachefile: Optional[str] = None):
    """
    Load spectrum from cache or calculate if needed.

    The cache is a nested structure of spectra. All processings applied to
    a spectrum result to spectra of the initial one. If part of the requred
    processings are available, only the needed steps are calculated and added
    to the cache.

    Args:
        required_steps: List of required steps in the form
            [{'proc': str, 'args': List[Any], 'kwargs': Dict[str, Any]}, ...]
        cachefile: optional. Defaults to None.
            Filename of the cache. If None no cache is used
    """
    def recall():
        if len(required_steps):
            last_proc = required_steps.pop()
            if last_proc.is_constructor:
                spe = Spectrum.apply_creator(last_proc, cachefile_=cachefile)
            else:
                spe = recur(required_steps=required_steps)
                spe._cachefile = cachefile
                spe = spe.apply_processing(last_proc)
            return spe
        else:
            raise Exception('no starting point')

    def recur(required_steps: spe_t.SpeProcessingListModel):
        try:
            if cachefile:
                spe = get_cache()
            else:
                spe = recall()
        except Exception:
            spe = recall()
        spe._cachefile = cachefile
        return spe

    def get_cache():
        try:
            cache_path = required_steps.cache_path()
            if cache_path:
                cache_path = '/cache/'+cache_path+'/_data'
            else:
                cache_path = 'raw'
            spe = Spectrum.from_chada(cachefile, cache_path)
            spe._applied_processings.extend_left(required_steps.root)
            return spe
        except Exception as e:
            logger.info(repr(e))
            raise e

    return recur(required_steps)
