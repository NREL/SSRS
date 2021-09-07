from datetime import datetime
from typing import Tuple
import numpy as np
from scipy.interpolate import griddata


min_updraft = 0.

def orographic_updraft_function(
        wspeed: np.ndarray,
        wdirn: np.ndarray,
        terrain_slope: np.ndarray,
        terrain_aspect: np.ndarray
) -> np.ndarray:
    """ Returns orographic updraft using provided atmospheric entities """

    return np.maximum(min_updraft, np.multiply(wspeed, np.multiply(np.sin(
        terrain_slope), np.maximum(1e-5, np.cos(terrain_aspect - wdirn)))))


def compute_orographic_updraft(
        tr_res: float,
        tr_extent: Tuple[float, float, float, float],
        tr_slope: np.ndarray,
        tr_aspect: np.ndarray,
        wtk_indices: np.ndarray,
        wtk_xgrid: np.ndarray,
        wtk_ygrid: np.ndarray,
        wtk_wspeed: np.ndarray,
        wtk_wdirn_deg: np.ndarray,
        interp_type: str,
        debug: int = 0
):
    """Interpolates WTK data onto the given terrain resolution,
    and then computes orographic updraft"""

    # print('Orographic: {0:s}'.format(timestamp.strftime('%I %p, %x')),
    #      end=" ", flush=True)
    wtk_wdirn = wtk_wdirn_deg * np.pi / 180.
    wtk_easterly = np.multiply(wtk_wspeed, np.sin(wtk_wdirn))
    wtk_northerly = np.multiply(wtk_wspeed, np.cos(wtk_wdirn))

    # interpolate individual wind speed components
    xmin, xmax, ymin, ymax = tr_extent
    xsize = int((xmax - xmin) * 1000. // tr_res)
    ysize = int((ymax - ymin) * 1000. // tr_res)
    tr_xgrid = np.linspace(xmin, xmax, xsize)
    tr_ygrid = np.linspace(ymin, ymax, ysize)

    tr_xmesh, tr_ymesh = np.meshgrid(tr_xgrid, tr_ygrid)
    points = np.array([wtk_xgrid, wtk_ygrid]).T
    interp_easterly = griddata(points, wtk_easterly, (tr_xmesh, tr_ymesh),
                               method=interp_type)
    interp_northerly = griddata(points, wtk_northerly, (tr_xmesh, tr_ymesh),
                                method=interp_type)

    # Convert back to wspeed and wdirn
    interp_wspeed = np.sqrt(np.square(interp_easterly) +
                            np.square(interp_northerly))
    interp_wdirn = np.mod(np.arctan2(
        interp_easterly, interp_northerly) + 2. * np.pi, 2. * np.pi)

    # compute updraft speed
    orographs = orographic_updraft_function(interp_wspeed, interp_wdirn,
                                            tr_slope, tr_aspect)
    if debug:
        interp_data = np.array([interp_wspeed, interp_wdirn * 180 / np.pi])
        return orographs, interp_data
    else:
        return orographs.astype(np.float32)


def deardoff_velocity_function(
        pot_temperature: np.ndarray,
        blayer_height: np.ndarray,
        surface_heat_flux: np.ndarray
) -> np.ndarray:
    """ returns deardoff velocity (convective velocity scale) """
    fac = 9.8 / 1216.  # to produce kinematic entity
    pot_temp_kelvin = np.add(pot_temperature, 273.15)
    pos_heat_flux = surface_heat_flux.clip(min=0.)
    mod_blheight = blayer_height.clip(min=100.)
    return np.maximum(min_updraft, np.power(fac * np.divide(
        np.multiply(mod_blheight, pos_heat_flux), pot_temp_kelvin), 1. / 3.))


def compute_potential_temperature(
        pressure: np.ndarray,
        temperature: np.ndarray,
) -> np.ndarray:
    """ returns potential temperature in degree celsius"""
    temp_k = np.add(temperature, 273.15)
    ref_pressure = 1e5
    temp_r = np.divide(ref_pressure, pressure)
    return np.multiply(temp_k, np.power(temp_r, 0.2857)) - 273.15


def compute_thermal_updraft(
        z: np.ndarray,
        deardoff_vel: np.ndarray,
        blayer_height: np.ndarray
) -> np.ndarray:
    """ returns thermal updraft at any height z"""
    zbyzi = np.divide(z, blayer_height).clip(min=0., max=1.)
    emat = 0.85 * np.multiply(np.power(zbyzi, 1 / 3), np.subtract(1.3, zbyzi))
    return np.maximum(min_updraft, np.multiply(deardoff_vel, emat))


def compute_deardoff_velocity(
        tr_res: float,
        tr_extent: Tuple[float, float, float, float],
        wtk_indices: np.ndarray,
        wtk_xgrid: np.ndarray,
        wtk_ygrid: np.ndarray,
        wtk_pot_temp: np.ndarray,
        wtk_blheight: np.ndarray,
        wtk_sflux: np.ndarray,
        interp_type: str,
        debug: int = 0
):
    """Interpolates WTK data and then computes deardoff velocity"""

    # interpolate onto the terrain grid
    tr_xgrid = np.arange(tr_extent[0], tr_extent[1], tr_res)
    tr_ygrid = np.arange(tr_extent[2], tr_extent[3], tr_res)
    tr_xmesh, tr_ymesh = np.meshgrid(tr_xgrid, tr_ygrid)
    points = np.array([wtk_xgrid, wtk_ygrid]).T
    interp_pot_temp = griddata(points, wtk_pot_temp, (tr_xmesh, tr_ymesh),
                               method=interp_type)
    interp_blheight = griddata(points, wtk_blheight, (tr_xmesh, tr_ymesh),
                               method=interp_type)
    interp_sflux = griddata(points, wtk_sflux, (tr_xmesh, tr_ymesh),
                            method=interp_type)

    # compute updraft speed
    deardoff = deardoff_velocity_function(interp_pot_temp,
                                          interp_blheight,
                                          interp_sflux)

    # finalize
    if debug:
        interp_data = np.array(
            [interp_pot_temp, interp_blheight, interp_sflux])
        return deardoff, interp_data
    else:
        return deardoff, interp_blheight


def weibull_percentage_above_threshold(
        updraft_median: np.ndarray,
        weibull_k: float,
        updraft_threshold: float
) -> np.ndarray:
    """return the percentage of updraft being more than the threshold based 
    on Weibull disstribution with given median """

    lamda = np.divide(updraft_median, np.log(2) ** (1 / weibull_k))
    lamda = lamda.clip(min=1e-5)
    exponent = np.power(np.divide(updraft_threshold, lamda), weibull_k)
    perc = np.exp(-exponent).clip(min=min_updraft)
    return perc
