"""
"""
from jax import jit as jjit
from jax import numpy as jnp
from .flux import _calc_rest_mag_multifilter
from .wssp import _calc_weighted_ssp_from_diffstar_params
from .stars import DEFAULT_MAH_PARAMS
from .met import DEFAULT_MZR_PARAMS


MET_PARAMS = jnp.array(list(DEFAULT_MZR_PARAMS.values()))
MAH_K = DEFAULT_MAH_PARAMS["mah_k"]


@jjit
def _pred_mags_singlegal(
    params,
    t_obs,
    wave_filters,
    trans_filters,
    ssp_wave,
    ssp_templates,
    lgZsun_bin_mids,
    lg_ages,
):
    mah_params, ms_params, q_params = params[0:6], params[6:13], params[13:]
    args = (
        t_obs,
        lgZsun_bin_mids,
        lg_ages,
        ssp_templates,
        mah_params,
        ms_params,
        q_params,
        MET_PARAMS,
    )
    res = _calc_weighted_ssp_from_diffstar_params(*args)
    lgmet_weights, age_weights, lum_spec = res

    mags = _calc_rest_mag_multifilter(ssp_wave, lum_spec, wave_filters, trans_filters)
    return mags
