"""
"""
import numpy as np
from collections import OrderedDict
from jax import jit as jjit
from jax import numpy as jnp
from .stars import LGT0, DEFAULT_MAH_PARAMS, DEFAULT_MS_PARAMS
from .stars import diffstar_sfh
from .magpred import _pred_mags_singlegal
from .sampling_helpers import latin_hypercube
from .network_helpers import _unit_scale_traindata

T_OBS_MIN = 1.5
MAGS_MIN, MAGS_MAX = -5, 25

_X0, _K = 0, 1

MS_BOUNDS = OrderedDict(
    ms_lgmcrit=(9.0, 13.5),
    ms_lgy_at_mcrit=(-3.0, 0.5),
    ms_indx_lo=(0.0, 5.0),
    ms_floor_low=(0.5, 3.0),
    ms_tau_dep=(0.0, 10.0),
)
Q_BOUNDS = OrderedDict(
    lg_qt=(0.5, 1.5), lg_qs=(-2.0, -0.01), lg_drop=(-2, 0.0), lg_rejuv=(-2, 0.0)
)

MAH_BOUNDS = OrderedDict(
    mah_logm0=(10, 15.0), mah_logtc=(0, 1.25), mah_early=(0.5, 2.0), mah_late=(0.0, 1)
)


def get_diffstar_photometry_tdata_batch(fsps_data, n_batch):
    gen = diffstar_param_generator()
    x_collector = []
    y_collector = []
    while True:
        for i in range(n_batch):
            lhs, mah, ms, q, mah_params, ms_params, q_params = next(gen)
            t_obs = np.random.uniform(T_OBS_MIN, 10 ** LGT0)
            params = (*mah_params, *ms_params, *q_params)
            mags = _pred_mags_singlegal(params, t_obs, *fsps_data)
            y = _unit_scale_traindata(mags, MAGS_MIN, MAGS_MAX).flatten()

            x_collector.append(lhs)
            y_collector.append(y)
        yield jnp.array(x_collector), jnp.array(y_collector)


def diffstar_param_generator():
    xmins = np.zeros(13)
    xmaxs = np.ones(13)
    tarr = np.linspace(0.1, 10 ** LGT0, 50)
    while True:
        lhs = latin_hypercube(xmins, xmaxs, 13, 1).flatten()
        mah, ms, q = generate_varied_params(*lhs)
        res = get_diffstar_sfh_inputs_from_varied_params(mah, ms, q)
        mah_params, ms_params, q_params = res
        sfh = diffstar_sfh(tarr, mah_params, ms_params, q_params)
        if np.all(np.isfinite(sfh)):
            yield lhs, mah, ms, q, mah_params, ms_params, q_params


def generate_varied_params(*frac_params):
    f_m0, f_lgtc, f_early, f_late = frac_params[:4]
    lgm0 = _get_bounded_param(f_m0, *MAH_BOUNDS["mah_logm0"])
    lgtc = _get_bounded_param(f_lgtc, *MAH_BOUNDS["mah_logtc"])
    early, late = _get_bounded_mah_early_late(f_early, f_late)
    mah_params = lgm0, lgtc, early, late

    f_ms_lgmc, f_ms_lgy, f_ms_indx_lo, f_ms_floor, f_ms_tau_dep = frac_params[4:9]
    ms_lgmc = _get_bounded_param(f_ms_lgmc, *MS_BOUNDS["ms_lgmcrit"])
    ms_lgy = _get_bounded_param(f_ms_lgy, *MS_BOUNDS["ms_lgy_at_mcrit"])
    ms_floor = _get_bounded_param(f_ms_floor, *MS_BOUNDS["ms_floor_low"])
    ms_indx_lo = _get_bounded_param(f_ms_indx_lo, *MS_BOUNDS["ms_indx_lo"])
    ms_tau_dep = _get_bounded_param(f_ms_tau_dep, *MS_BOUNDS["ms_tau_dep"])
    ms_params = ms_lgmc, ms_lgy, ms_indx_lo, ms_floor, ms_tau_dep

    f_lg_qt, f_lg_qs, f_lg_qdrop, f_lg_rejuv = frac_params[9:]
    lg_qt = _get_bounded_param(f_lg_qt, *Q_BOUNDS["lg_qt"])
    lg_drop, lg_rejuv = _get_bounded_qdrop_qrejuv(f_lg_qdrop, f_lg_rejuv)
    lg_qs = _get_bounded_param(f_lg_qs, *Q_BOUNDS["lg_qs"])
    q_params = lg_qt, lg_qs, lg_drop, lg_rejuv
    return mah_params, ms_params, q_params


def get_diffstar_sfh_inputs_from_varied_params(mah, ms, q):
    lgm0, lgtc, early, late = mah
    mah_params = LGT0, lgm0, lgtc, DEFAULT_MAH_PARAMS["mah_k"], early, late

    lgmc, lgy, indx_lo, floor, tau_dep = ms
    ms_params = (
        lgmc,
        lgy,
        DEFAULT_MS_PARAMS["ms_indx_k"],
        indx_lo,
        DEFAULT_MS_PARAMS["ms_indx_hi"],
        floor,
        tau_dep,
    )
    return mah_params, ms_params, q


def generate_diffstar_sfh_inputs(*frac_params):
    _mah_params, _ms_params, _q_params = generate_varied_params(*frac_params)
    mah_params, ms_params, q_params = get_diffstar_sfh_inputs_from_varied_params(
        _mah_params, _ms_params, _q_params
    )
    return mah_params, ms_params, q_params


@jjit
def _sigmoid(x, logtc, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1.0 + jnp.exp(-k * (x - logtc)))


@jjit
def _get_bounded_mah_early_late(f_early, f_late):
    late = _get_bounded_param(f_late, *MAH_BOUNDS["mah_late"])
    early = _get_bounded_param(f_early, late, MAH_BOUNDS["mah_early"][1])
    return early, late


@jjit
def _get_bounded_qdrop_qrejuv(f_drop, f_rejuv):
    lg_drop = _get_bounded_param(f_drop, *Q_BOUNDS["lg_drop"])
    lg_rejuv = _get_bounded_param(f_rejuv, lg_drop, Q_BOUNDS["lg_rejuv"][1])
    return lg_drop, lg_rejuv


@jjit
def _get_bounded_param(f, pmin, pmax):
    dp = pmax - pmin
    return pmin + f * dp


@jjit
def _inverse_sigmoid(y, x0, k, ylo, yhi):
    lnarg = (yhi - ylo) / (y - ylo) - 1
    return x0 - jnp.log(lnarg) / k
