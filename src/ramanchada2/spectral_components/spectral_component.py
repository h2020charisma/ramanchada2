#!/usr/bin/env python3

from __future__ import annotations

from ramanchada2.misc.plottable import Plottable
from ramanchada2.misc.base_class import BaseClass


class SpectralComponent(Plottable, BaseClass):
    def __init__(self, **kwargs):
        super(Plottable, self).__init__()
        super(BaseClass, self).__init__()
        self._origin = [(type(self).__name__, (), kwargs)]
