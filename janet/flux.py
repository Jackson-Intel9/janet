"""
"""
from jax import numpy as jnp
from jax import jit as jjit
from jax import vmap as jvmap
import jax
AB0 = 1.13492e-13  # 3631 Jansky placed at 10 pc in units of Lsun/Hz


@jjit
def _calc_obs_mag(
    wave_spec_rest, lum_spec, wave_filter, trans_filter, z, z_table, dmod_table
):
    flux_source = _obs_flux_ssp(wave_spec_rest, lum_spec, wave_filter, trans_filter, z)
    flux_ab0 = _flux_ab0_at_10pc(wave_filter, trans_filter)
    mag_no_dimming = -2.5 * jnp.log10(flux_source / flux_ab0)
    obs_mag = _cosmological_dimming(mag_no_dimming, z, z_table, dmod_table)
    return obs_mag


@jjit
def _cosmological_dimming(mag_no_dimming, z, z_table, distance_modulus_table):
    distance_modulus = jnp.interp(z, z_table, distance_modulus_table)
    return mag_no_dimming + distance_modulus - 2.5 * jnp.log10(1 + z)


@jjit
def _calc_obs_mag_no_dimming(wave_spec_rest, lum_spec, wave_filter, trans_filter, z):
    flux_source = _obs_flux_ssp(wave_spec_rest, lum_spec, wave_filter, trans_filter, z)
    flux_ab0 = _flux_ab0_at_10pc(wave_filter, trans_filter)
    return -2.5 * jnp.log10(flux_source / flux_ab0)


@jjit
def _calc_rest_mag(wave_spec_rest, lum_spec, wave_filter, trans_filter):
    flux_source = _rest_flux_ssp(wave_spec_rest, lum_spec, wave_filter, trans_filter)
    flux_ab0 = _flux_ab0_at_10pc(wave_filter, trans_filter)
    return -2.5 * jnp.log10(flux_source / flux_ab0)


@jjit
def _obs_flux_ssp(wave_spec_rest, lum_spec, wave_filter, trans_filter, z):
    lum_zshift_phot = jnp.interp(
        wave_filter, wave_spec_rest * (1 + z), lum_spec, left=0, right=0
    )
    integrand = trans_filter * lum_zshift_phot / wave_filter
    lum_filter = jax.scipy.integrate.trapezoid(integrand, x=wave_filter)
    return lum_filter


@jjit
def _rest_flux_ssp(wave_spec_rest, lum_spec, wave_filter, trans_filter):
    lum_phot = jnp.interp(wave_filter, wave_spec_rest, lum_spec, left=0, right=0)
    integrand = trans_filter * lum_phot / wave_filter
    lum_filter = jax.scipy.integrate.trapezoid(integrand, x=wave_filter)
    return lum_filter


@jjit
def _flux_ab0_at_10pc(wave_filter, trans_filter):
    integrand = trans_filter * AB0 / wave_filter
    lum_ab0_filter = jax.scipy.integrate.trapezoid(integrand, x=wave_filter)
    return lum_ab0_filter


_a = [None, None, 0, 0]
_calc_rest_mag_multifilter = jjit(jvmap(_calc_rest_mag, in_axes=_a))
