#!/usr/bin/env python3

from ..spectral_peak import SpectralPeak


class VoigtPeak(SpectralPeak):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
