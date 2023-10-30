from __future__ import annotations
from abc import ABC, abstractmethod
from matplotlib.axes import Axes
import matplotlib.pyplot as plt


class Plottable(ABC):
    def plot(self, ax=None, label=' ', **kwargs) -> Axes:
        if ax is None:
            fig, ax = plt.subplots(1)
        self._plot(ax, label=label, **kwargs)
        ax.legend()
        return ax

    @abstractmethod
    def _plot(self, ax, **kwargs):
        pass
