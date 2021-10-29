"""
"""
import numpy as np
from ..generate_tdata import _get_bounded_mah_early_late, MAH_BOUNDS
from ..generate_tdata import _get_bounded_qdrop_qrejuv, Q_BOUNDS
from ..generate_tdata import MS_BOUNDS
from ..generate_tdata import generate_varied_params
from jax import random as jran


def _check_q_params(lg_qt, lg_qs, lg_drop, lg_rejuv):
    assert Q_BOUNDS["lg_drop"][0] < lg_drop < lg_rejuv < Q_BOUNDS["lg_rejuv"][1]
    assert Q_BOUNDS["lg_qt"][0] < lg_qt < Q_BOUNDS["lg_qt"][1]
    assert Q_BOUNDS["lg_qs"][0] < lg_qs < Q_BOUNDS["lg_qs"][1]


def _check_mah_params(lgm0, lgtc, early, late):
    assert MAH_BOUNDS["mah_early"][1] > early > late > 0
    assert MAH_BOUNDS["mah_logm0"][0] < lgm0 < MAH_BOUNDS["mah_logm0"][1]
    assert MAH_BOUNDS["mah_logtc"][0] < lgtc < MAH_BOUNDS["mah_logtc"][1]


def _check_ms_params(lgmc, lgy, indx_lo, floor, tau):
    assert MS_BOUNDS["ms_lgmcrit"][0] < lgmc < MS_BOUNDS["ms_lgmcrit"][1]
    assert MS_BOUNDS["ms_lgy_at_mcrit"][0] < lgy < MS_BOUNDS["ms_lgy_at_mcrit"][1]
    assert MS_BOUNDS["ms_indx_lo"][0] < indx_lo < MS_BOUNDS["ms_indx_lo"][1]
    assert MS_BOUNDS["ms_floor_low"][0] < floor < MS_BOUNDS["ms_floor_low"][1]
    assert MS_BOUNDS["ms_tau_dep"][0] < tau < MS_BOUNDS["ms_tau_dep"][1]


def test_get_bounded_mah_early_late():
    ran_key = jran.PRNGKey(0)
    for itry in range(100):
        ran_key, ikey = jran.split(ran_key)
        u = jran.uniform(ikey, minval=0, maxval=1, shape=(2,))
        early, late = _get_bounded_mah_early_late(*u)
        assert MAH_BOUNDS["mah_early"][1] > early > late > 0


def test_get_bounded_qdrop_qrejuv():
    ran_key = jran.PRNGKey(0)
    for itry in range(100):
        ran_key, ikey = jran.split(ran_key)
        u = jran.uniform(ikey, minval=0, maxval=1, shape=(2,))
        lg_drop, lg_rejuv = _get_bounded_qdrop_qrejuv(*u)
        assert Q_BOUNDS["lg_drop"][0] < lg_drop < lg_rejuv < Q_BOUNDS["lg_rejuv"][1]


def test_generate_random_params():
    ran_key = jran.PRNGKey(0)
    for itry in range(100):
        ran_key, ikey = jran.split(ran_key)
        frac_params = jran.uniform(ikey, minval=0, maxval=1, shape=(13,))
        mah_params, ms_params, q_params = generate_varied_params(*frac_params)

        lgm0, lgtc, early, late = mah_params
        _check_mah_params(lgm0, lgtc, early, late)

        ms_lgmc, ms_lgy, ms_indx_lo, ms_floor, ms_tau_dep = ms_params
        _check_ms_params(*ms_params)

        lg_qt, lg_qs, lg_drop, lg_rejuv = q_params
        _check_q_params(lg_qt, lg_qs, lg_drop, lg_rejuv)


def test_generate_minimum_params():
    mah_params, ms_params, q_params = generate_varied_params(*[0] * 13)
    lgm0, lgtc, early, late = mah_params
    assert np.allclose(lgm0, MAH_BOUNDS["mah_logm0"][0])
    assert np.allclose(lgtc, MAH_BOUNDS["mah_logtc"][0])
    assert np.allclose(late, MAH_BOUNDS["mah_late"][0])
    assert np.allclose(late, early)

    for key, val in zip(MS_BOUNDS.keys(), ms_params):
        assert np.allclose(MS_BOUNDS[key][0], val)

    lg_qt, lg_qs, lg_drop, lg_rejuv = q_params
    assert np.allclose(lg_qt, Q_BOUNDS["lg_qt"][0])
    assert np.allclose(lg_qs, Q_BOUNDS["lg_qs"][0])
    assert np.allclose(lg_drop, Q_BOUNDS["lg_drop"][0])
    assert np.allclose(lg_qs, lg_rejuv)


def test_generate_maximum_params():
    mah_params, ms_params, q_params = generate_varied_params(*[1] * 13)
    lgm0, lgtc, early, late = mah_params
    assert np.allclose(lgm0, MAH_BOUNDS["mah_logm0"][1])
    assert np.allclose(lgtc, MAH_BOUNDS["mah_logtc"][1])
    assert np.allclose(early, MAH_BOUNDS["mah_early"][1])
    assert np.allclose(late, MAH_BOUNDS["mah_late"][1])

    for key, val in zip(MS_BOUNDS.keys(), ms_params):
        assert np.allclose(MS_BOUNDS[key][1], val)

    lg_qt, lg_qs, lg_drop, lg_rejuv = q_params
    assert np.allclose(lg_qt, Q_BOUNDS["lg_qt"][1])
    assert np.allclose(lg_qs, Q_BOUNDS["lg_qs"][1])
    assert np.allclose(lg_drop, Q_BOUNDS["lg_drop"][1])
    assert np.allclose(lg_drop, lg_rejuv)
