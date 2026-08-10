import numpy as np
from lmfit import Model
from lmfit.models import guess_from_peak, update_param_vals

from .pearson4_hpw import (pearson4_hpw_area, pearson4_hpw_FWHM,
                           pearson4_hpw_FWHM_approx, pearson4_hpw_LeftWHM,
                           pearson4_hpw_RightWHM, perarson4_hpw_profile)
from .voigt_nu import (voigt_nu_FWHM, voigt_nu_FWHM_approx, voigt_nu_height,
                       voigt_nu_profile)


class VoigtNuModel(Model):
    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(voigt_nu_profile, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('w', min=1E-100)
        self.set_param_hint('nu', min=0.0, max=1.0)
        self.set_param_hint('sigma', expr='w * sqrt(nu) / sqrt(log(4))')
        self.set_param_hint('gamma', expr='sigma * (1-nu)')
        self.set_param_hint('fwhm', expr='voigt_nu_FWHM(w=sigma, nu=nu)')
        self.set_param_hint('fwhm_approx', expr='voigt_nu_FWHM_approx(w=sigma, nu=nu)')
        self.set_param_hint('fwhm_gamma_sigma', expr='1.0692*gamma+sqrt(0.8664*gamma**2+5.545083*sigma**2)')
        self.set_param_hint('height', expr='voigt_nu_height(amplitude=amplitude, sigma=sigma, nu=nu)')

    def guess(self, data, x, negative=False, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = self.make_params(amplitude=np.max(data), center=x[np.argmax(data)], nu=.5, w=10)
        return update_param_vals(pars, self.prefix, **kwargs)

    def make_params(self, *args, **kwargs):
        params = super().make_params(*args, **kwargs)
        params._asteval.symtable['voigt_nu_height'] = voigt_nu_height
        params._asteval.symtable['voigt_nu_FWHM'] = voigt_nu_FWHM
        params._asteval.symtable['voigt_nu_FWHM_approx'] = voigt_nu_FWHM_approx
        return params


class Pearson4HPWModel(Model):
    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(perarson4_hpw_profile, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('expon', value=1.5, min=0.5 + 1 / 1024.0, max=1000)
        self.set_param_hint('skew', value=0.0, min=-1000, max=1000)
        self.set_param_hint('amplitude', expr='pearson4_hpw_area(height, sigma, expon, skew)')
        self.set_param_hint('fwhm', expr='pearson4_hpw_FWHM(sigma, expon, skew)')
        self.set_param_hint('fwhm_approx', expr='pearson4_hpw_FWHM_approx(sigma, expon, skew)')

    def guess(self, data, x, negative=False, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = guess_from_peak(self, data, x, negative)
        return update_param_vals(pars, self.prefix, **kwargs)

    def make_params(self, *args, **kwargs):
        params = super().make_params(*args, **kwargs)
        params._asteval.symtable['pearson4_hpw_area'] = pearson4_hpw_area
        params._asteval.symtable['pearson4_hpw_FWHM_approx'] = pearson4_hpw_FWHM_approx
        params._asteval.symtable['pearson4_hpw_LeftWHM'] = pearson4_hpw_LeftWHM
        params._asteval.symtable['pearson4_hpw_RightWHM'] = pearson4_hpw_RightWHM
        params._asteval.symtable['pearson4_hpw_FWHM'] = pearson4_hpw_FWHM
        return params
