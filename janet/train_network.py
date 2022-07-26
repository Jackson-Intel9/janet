"""
"""
from jax import random as jran
from jax import numpy as jnp
from jax import jit as jjit
from jax.example_libraries import optimizers as jax_opt
from .retrieve_fake_fsps_data import load_fake_sps_data
from .generate_tdata import get_diffstar_photometry_tdata_batch
from .network_helpers import get_network, train_network


def train_photometry_network(
    n_batch, n_steps, dense_sizes=(32, 16, 8, 4), seed=0, sps_data=None, step_size=0.001
):
    if sps_data is None:
        fake_fsps_data = load_fake_sps_data()

    loss_data_generator = get_diffstar_photometry_tdata_batch(fake_fsps_data, n_batch)
    loss_data_init = next(loss_data_generator)
    X, Y = loss_data_init
    dim_in = X.shape[1]
    dim_out = Y.shape[1]

    ran_key = jran.PRNGKey(seed)
    net_init, net_pred = get_network("Selu", dim_in, dim_out, dense_sizes)
    net_params_init = net_init(ran_key)

    @jjit
    def _net_loss(p, data):
        x, target = data
        pred = net_pred(p, x)
        return _mse(pred, target)

    @jjit
    def _mse(y, x):
        d = y - x
        return jnp.mean(d * d)

    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state_init = opt_init(net_params_init)

    loss_data_generator = get_diffstar_photometry_tdata_batch(fake_fsps_data, n_batch)

    res = train_network(
        n_steps, _net_loss, get_params, opt_state_init, opt_update, loss_data_generator
    )
    state, loss_history, p_best, loss_min = res
    return state, loss_history, p_best, loss_min
