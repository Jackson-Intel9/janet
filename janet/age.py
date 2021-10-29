"""
"""
from jax import numpy as jnp
from jax import jit as jjit
from jax import ops as jops
from .utils import _jax_get_dt_array
from .stars import diffstar_sfh, TODAY


T_BIRTH_MIN = 0.001
N_T_LGSM_INTEGRATION = 100


@jjit
def _get_linspace_time_tables():
    t_table = jnp.linspace(T_BIRTH_MIN, TODAY, N_T_LGSM_INTEGRATION)
    lgt_table = jnp.log10(t_table)
    dt_table = _jax_get_dt_array(t_table)
    return t_table, lgt_table, dt_table


@jjit
def _get_sfh_tables(mah_params, ms_params, q_params):
    t_table, lgt_table, dt_table = _get_linspace_time_tables()
    sfh_table = diffstar_sfh(t_table, mah_params, ms_params, q_params)
    logsm_table = jnp.log10(jnp.cumsum(sfh_table * dt_table)) + 9
    return t_table, lgt_table, dt_table, sfh_table, logsm_table


@jjit
def _get_lgt_birth(t_obs, lg_ages):
    t_birth = t_obs - 10 ** lg_ages
    t_birth = jnp.where(t_birth < T_BIRTH_MIN, T_BIRTH_MIN, t_birth)
    lgt_birth = jnp.log10(t_birth)
    return lgt_birth


@jjit
def _get_age_weights_from_tables(lgt_birth_bin_edges, lgt_table, logsm_table):
    logsm_at_t_birth_bin_edges = jnp.interp(lgt_birth_bin_edges, lgt_table, logsm_table)
    delta_mstar_at_t_birth = -jnp.diff(10 ** logsm_at_t_birth_bin_edges)
    age_weights = delta_mstar_at_t_birth / delta_mstar_at_t_birth.sum()
    return age_weights


@jjit
def _get_lg_age_bin_edges(lg_ages):
    dlgtau = _jax_get_dt_array(lg_ages)

    lg_age_bin_edges = jnp.zeros(dlgtau.size + 1)
    lg_age_bin_edges = jops.index_update(
        lg_age_bin_edges, jops.index[:-1], lg_ages - dlgtau / 2
    )

    dlowest = (lg_ages[1] - lg_ages[0]) / 2
    lowest = lg_ages[0] - dlowest
    highest = lg_ages[-1] + dlgtau[-1] / 2
    lg_age_bin_edges = jops.index_update(lg_age_bin_edges, jops.index[0], lowest)
    lg_age_bin_edges = jops.index_update(lg_age_bin_edges, jops.index[-1], highest)

    return lg_age_bin_edges
