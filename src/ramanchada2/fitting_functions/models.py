from lmfit import Model
from lmfit.models import guess_from_peak, update_param_vals
from pearsonivamplitudeparametrizationhpw import PearsonIVAmplitudeParametrizationHPW
from voigtareaparametrizationnu import VoigtAreaParametrizationNu
import numpy as np


class PearsonIVParametrizationHPWModel(Model):
    r"""A model based on a Pearson IV distribution.
    The model has five parameters: `height` (:math:`a`), `position`,
    `sigma` (:math:`\sigma`), `expon` (:math:`m`) and `skew` (:math:`\nu`).
    """

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(PearsonIVAmplitudeParametrizationHPW.GetYOfOneTerm, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('expon', value=1.5, min=0.5 + 1 / 1024.0, max=1000)
        self.set_param_hint('skew', value=0.0, min=-1000, max=1000)

    def guess(self, data, x, negative=False, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = guess_from_peak(self, data, x, negative)
        pars[f'{self.prefix}height'].set(value=pars[f'{self.prefix}amplitude'] / pars[f'{self.prefix}sigma'])
        pars[f'{self.prefix}expon'].set(value=1)
        pars[f'{self.prefix}skew'].set(value=0.0)
        return update_param_vals(pars, self.prefix, **kwargs)


class VoigtAreaParametrizationNuModel(Model):
    r"""A model based on a Voigt distribution function.
    The model has four Parameters: `amplitude`, `center`, `sigma`, and
    `nu`. The parameter `nu` is constrained to the range [0,1]. In addition,
    parameters `fwhm` are included as constraints to report
    full width at half maximum.

    For more information, see: https://en.wikipedia.org/wiki/Voigt_profile
    """

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(VoigtAreaParametrizationNu.GetYOfOneTerm, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('sigma', min=np.tiny)
        self.set_param_hint('nu', min=0, max=1)

        fexpr = ("2.0*{pre:s}sigma")
        self.set_param_hint('fwhm', expr=fexpr.format(pre=self.prefix))

    def guess(self, data, x, negative=False, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = guess_from_peak(self, data, x, negative)
        pars[f'{self.prefix}nu'].set(value=1.0)
        return update_param_vals(pars, self.prefix, **kwargs)
