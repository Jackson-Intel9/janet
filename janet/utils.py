"""
"""
from jax import jit as jjit
from jax import numpy as jnp
from jax import ops as jops
from jax import vmap


@jjit
def _jax_get_dt_array(t):
    dt = jnp.zeros_like(t)
    tmids = 0.5 * (t[:-1] + t[1:])
    dtmids = jnp.diff(tmids)
    dt = jops.index_update(dt, jops.index[1:-1], dtmids)

    t_lo = t[0] - (t[1] - t[0]) / 2
    t_hi = t[-1] + dtmids[-1] / 2

    dt = jops.index_update(dt, jops.index[0], tmids[0] - t_lo)
    dt = jops.index_update(dt, jops.index[-1], t_hi - tmids[-1])
    return dt


@jjit
def _get_bin_edges(bin_mids, lowest_bin_edge, highest_bin_edge):
    dbins = _jax_get_dt_array(bin_mids)

    bin_edges = jnp.zeros(dbins.size + 1)
    bin_edges = jops.index_update(bin_edges, jops.index[:-1], bin_mids - dbins / 2)

    bin_edges = jops.index_update(bin_edges, jops.index[0], lowest_bin_edge)
    bin_edges = jops.index_update(bin_edges, jops.index[-1], highest_bin_edge)

    return bin_edges


@jjit
def _tw_cuml_kern(x, m, h):
    """Triweight kernel version of an err function."""
    z = (x - m) / h
    val = (
        -5 * z ** 7 / 69984
        + 7 * z ** 5 / 2592
        - 35 * z ** 3 / 864
        + 35 * z / 96
        + 1 / 2
    )
    val = jnp.where(z < -3, 0, val)
    val = jnp.where(z > 3, 1, val)
    return val


@jjit
def _tw_sigmoid(x, x0, tw_h, ymin, ymax):
    height_diff = ymax - ymin
    body = _tw_cuml_kern(x, x0, tw_h)
    return ymin + height_diff * body


@jjit
def _get_tw_h_from_sigmoid_k(k):
    return 1 / (0.614 * k)


@jjit
def _triweighted_histogram_kernel(x, sig, lo, hi):
    """Triweight kernel integrated across the boundaries of a single bin."""
    a = _tw_cuml_kern(x, lo, sig)
    b = _tw_cuml_kern(x, hi, sig)
    return a - b


_a = [None, None, 0, 0]
_triweighted_histogram_vmap = jjit(vmap(_triweighted_histogram_kernel, in_axes=_a))


@jjit
def triweighted_histogram(x, sig, xbins):
    return _triweighted_histogram_vmap(x, sig, xbins[:-1], xbins[1:])
