from lmfit import Model
from lmfit import Parameter
from lmfit.models import guess_from_peak, update_param_vals
from ramanchada2.fitting_functions.pearsonivamplitudeparametrizationhpw import (
    PearsonIVAmplitudeParametrizationHPW,
)
from ramanchada2.fitting_functions.voigtareaparametrizationnu import (
    VoigtAreaParametrizationNu,
)


class PearsonIVParametrizationHPWModel(Model):
    r"""A model based on a Pearson IV distribution.
    The model has five parameters: `height` (:math:`a`), `position`,
    `sigma` (:math:`\sigma`), `expon` (:math:`m`) and `skew` (:math:`\nu`).
    """

    def __init__(self, independent_vars=["x"], prefix="", nan_policy="raise", **kwargs):
        kwargs.update(
            {
                "prefix": prefix,
                "nan_policy": nan_policy,
                "independent_vars": independent_vars,
            }
        )
        super().__init__(PearsonIVAmplitudeParametrizationHPW.GetYOfOneTerm, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint("expon", value=1.5, min=0.5 + 1 / 1024.0, max=1000)
        self.set_param_hint("skew", value=0.0, min=-1000, max=1000)

    def guess(self, data, x, negative=False, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = guess_from_peak(self, data, x, negative)
        return update_param_vals(pars, self.prefix, **kwargs)

    def make_params(self, verbose=False, **kwargs):
        pars = super().make_params(verbose=verbose, **kwargs)
        pars[f"{self.prefix}height"].set(
            value=kwargs[f"{self.prefix}amplitude"] / kwargs[f"{self.prefix}sigma"]
        )
        pars[f"{self.prefix}expon"].set(value=1.0)
        pars[f"{self.prefix}skew"].set(value=0.0)
        return pars

    def fit(
        self,
        data,
        params=None,
        weights=None,
        method="leastsq",
        iter_cb=None,
        scale_covar=True,
        verbose=False,
        fit_kws=None,
        nan_policy=None,
        calc_covar=True,
        max_nfev=None,
        **kwargs,
    ):
        """overwrite fit in order to amend area and fwhm to the parameters"""
        result = super().fit(
            data,
            params=params,
            weights=weights,
            method=method,
            iter_cb=iter_cb,
            scale_covar=scale_covar,
            verbose=verbose,
            fit_kws=fit_kws,
            nan_policy=nan_policy,
            calc_covar=calc_covar,
            max_nfev=max_nfev,
            **kwargs,
        )
        pahf = PearsonIVAmplitudeParametrizationHPW.GetPositionAreaHeightFWHMFromPeakParameters(
            result.params[f"{self.prefix}height"],
            result.params[f"{self.prefix}center"],
            result.params[f"{self.prefix}sigma"],
            result.params[f"{self.prefix}expon"],
            result.params[f"{self.prefix}skew"],
            result.covar
        )
        p1 = Parameter(f"{self.prefix}amplitude", value=pahf.Area)
        p1.stderr = pahf.AreaStdDev
        p2 = Parameter(f"{self.prefix}fwhm", value=pahf.FWHM)
        p2.stderr = pahf.FWHMStdDev
        result.params.add_many(p1, p2)
        return result


class VoigtAreaParametrizationNuModel(Model):
    r"""A model based on a Voigt distribution function.
    The model has four Parameters: `amplitude`, `center`, `sigma`, and
    `nu`. The parameter `nu` is constrained to the range [0,1]. In addition,
    parameters `fwhm` are included as constraints to report
    full width at half maximum.

    For more information, see: https://en.wikipedia.org/wiki/Voigt_profile
    """

    def __init__(self, independent_vars=["x"], prefix="", nan_policy="raise", **kwargs):
        kwargs.update(
            {
                "prefix": prefix,
                "nan_policy": nan_policy,
                "independent_vars": independent_vars,
            }
        )
        super().__init__(VoigtAreaParametrizationNu.GetYOfOneTerm, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint("sigma", min=1E-100)
        self.set_param_hint("nu", min=0.0, max=1.0)
        self.set_param_hint("gamma", expr='sigma*(1-nu)')

    def guess(self, data, x, negative=False, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = guess_from_peak(self, data, x, negative)
        pars[f"{self.prefix}nu"].set(value=1.0)
        return update_param_vals(pars, self.prefix, **kwargs)

    def fit(
        self,
        data,
        params=None,
        weights=None,
        method="leastsq",
        iter_cb=None,
        scale_covar=True,
        verbose=False,
        fit_kws=None,
        nan_policy=None,
        calc_covar=True,
        max_nfev=None,
        **kwargs,
    ):
        """overwrite fit in order to amend area and fwhm to the parameters"""
        result = super().fit(
            data,
            params=params,
            weights=weights,
            method=method,
            iter_cb=iter_cb,
            scale_covar=scale_covar,
            verbose=verbose,
            fit_kws=fit_kws,
            nan_policy=nan_policy,
            calc_covar=calc_covar,
            max_nfev=max_nfev,
            **kwargs,
        )
        pahf = VoigtAreaParametrizationNu.GetPositionAreaHeightFWHMFromPeakParameters(
            result.params[f"{self.prefix}amplitude"],
            result.params[f"{self.prefix}center"],
            result.params[f"{self.prefix}sigma"],
            result.params[f"{self.prefix}nu"],
            result.covar
        )
        p1 = Parameter(f"{self.prefix}height", value=pahf.Height)
        p1.stderr = pahf.HeightStdDev
        p2 = Parameter(f"{self.prefix}fwhm", value=pahf.FWHM)
        p2.stderr = pahf.FWHMStdDev
        result.params.add_many(p1, p2)
        return result
