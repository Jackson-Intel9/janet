"""
"""
import numpy as np
from ..stars import DEFAULT_MS_PARAMS, DEFAULT_Q_PARAMS
from ..magpred import _pred_mags_singlegal, _pred_mags_galpop
from ..stars import DEFAULT_MAH_PARAMS as MAH_PARS


def test_calc_weighted_ssp_from_diffstar_params():

    t_obs = 11.0
    n_met, n_age, n_wave_spec = 22, 94, 1_000
    lg_ages = np.linspace(6, 10, n_age) - 9
    ssp_templates = np.zeros((n_met, n_age, n_wave_spec))
    lgZsun_bin_mids = np.linspace(-2, 0, n_met)

    mah_params = np.array(list(MAH_PARS.values()))
    ms_params = np.array(list(DEFAULT_MS_PARAMS.values()))
    q_params = np.array(list(DEFAULT_Q_PARAMS.values()))
    params = np.array((*mah_params, *ms_params, *q_params))

    n_wave_filter = 300
    wave_filter = np.linspace(0, 1, n_wave_filter)
    trans_filter = np.ones_like(wave_filter)

    ssp_wave = np.linspace(0, 1, n_wave_spec)

    n_filters = 6
    waves_filters = np.tile(wave_filter, n_filters).reshape((n_filters, -1))
    trans_filters = np.tile(trans_filter, n_filters).reshape((n_filters, -1))

    args = (
        params,
        t_obs,
        waves_filters,
        trans_filters,
        ssp_wave,
        ssp_templates,
        lgZsun_bin_mids,
        lg_ages,
    )
    mags_pred = _pred_mags_singlegal(*args)
    assert mags_pred.shape == (n_filters,)


def test_calc_weighted_ssp_from_diffstarpop():

    t_obs = 11.0
    n_met, n_age, n_wave_spec = 22, 94, 1_000
    lg_ages = np.linspace(6, 10, n_age) - 9
    ssp_templates = np.zeros((n_met, n_age, n_wave_spec))
    lgZsun_bin_mids = np.linspace(-2, 0, n_met)

    mah_params = np.array(list(MAH_PARS.values()))
    ms_params = np.array(list(DEFAULT_MS_PARAMS.values()))
    q_params = np.array(list(DEFAULT_Q_PARAMS.values()))
    params_singlegal = np.array((*mah_params, *ms_params, *q_params))

    n_gals = 1_000
    shape_pop = (n_gals, params_singlegal.size)
    params_galpop = np.tile(params_singlegal, n_gals).reshape(shape_pop)

    n_wave_filter = 300
    wave_filter = np.linspace(0, 1, n_wave_filter)
    trans_filter = np.ones_like(wave_filter)

    ssp_wave = np.linspace(0, 1, n_wave_spec)

    n_filters = 6
    waves_filters = np.tile(wave_filter, n_filters).reshape((n_filters, -1))
    trans_filters = np.tile(trans_filter, n_filters).reshape((n_filters, -1))

    args = (
        params_galpop,
        t_obs,
        waves_filters,
        trans_filters,
        ssp_wave,
        ssp_templates,
        lgZsun_bin_mids,
        lg_ages,
    )
    mags_pred = _pred_mags_galpop(*args)
    assert mags_pred.shape == (n_gals, n_filters)
