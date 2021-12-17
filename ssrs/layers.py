""" Module for computing topographical and atmospheric layers """

import numpy as np
# import richdem as rd


def compute_orographic_updraft(
    wspeed: np.ndarray,
    wdirn: np.ndarray,
    slope: np.ndarray,
    aspect: np.ndarray,
    min_updraft_val: float = 1e-5
) -> np.ndarray:
    """ Returns orographic updraft using wind speed, wind direction, slope
    and aspect """
    aspect_diff = np.maximum(0., np.cos((aspect - wdirn) * np.pi / 180.))
    return np.maximum(min_updraft_val, np.multiply(wspeed, np.multiply(np.sin(
        slope * np.pi / 180.), aspect_diff)))


def deardoff_velocity_function(
        pot_temperature: np.ndarray,
        blayer_height: np.ndarray,
        surface_heat_flux: np.ndarray,
        min_updraft_val: float = 1e-5
) -> np.ndarray:
    """ returns deardoff velocity (convective velocity scale) """
    fac = 9.8 / 1216.  # to produce kinematic entity
    pot_temp_kelvin = np.add(pot_temperature, 273.15)
    pos_heat_flux = surface_heat_flux.clip(min=0.)
    mod_blheight = blayer_height.clip(min=100.)
    return np.maximum(min_updraft_val, np.power(fac * np.divide(
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
        zmat: np.ndarray,
        deardoff_vel: np.ndarray,
        blayer_height: np.ndarray,
        min_updraft_val=1e-5
) -> np.ndarray:
    """ returns thermal updraft at any height z"""
    zbyzi = np.divide(zmat, blayer_height).clip(min=0., max=1.)
    emat = 0.85 * np.multiply(np.power(zbyzi, 1 / 3), np.subtract(1.3, zbyzi))
    return np.maximum(min_updraft_val, np.multiply(deardoff_vel, emat))


def compute_slope_degrees(z_mat: np.ndarray, res: float):
    """ Calculate local terrain slope using 3x3 stencil

    Parameters:
    ----------
    z_mat : numpy array
        Contains elevation data in meters
    res: float
        Resolution in meters, assumed to be same in both directions

    Returns:
    --------
    numpy array containing slope in degrees
    """

    slope = np.empty_like(z_mat)
    slope[:, :] = np.nan
    z_1 = z_mat[:-2, 2:]  # upper left
    z_2 = z_mat[1:-1, 2:]  # upper middle
    z_3 = z_mat[2:, 2:]  # upper right
    z_4 = z_mat[:-2, 1:-1]  # center left
   # z5 = z[ 1:-1, 1:-1] # center
    z_6 = z_mat[2:, 1:-1]  # center right
    z_7 = z_mat[:-2, :-2]  # lower left
    z_8 = z_mat[1:-1, :-2]  # lower middle
    z_9 = z_mat[2:, :-2]  # lower right
    dz_dx = ((z_3 + 2 * z_6 + z_9) - (z_1 + 2 * z_4 + z_7)) / (8 * res)
    dz_dy = ((z_1 + 2 * z_2 + z_3) - (z_7 + 2 * z_8 + z_9)) / (8 * res)
    rise_run = np.sqrt(dz_dx**2 + dz_dy**2)
    slope[1:-1, 1:-1] = np.degrees(np.arctan(rise_run))
    return np.nan_to_num(slope)


def compute_aspect_degrees(z_mat: np.ndarray, res: float):
    """ Calculate local terrain aspect using 3x3 stencil

    Parameters:
    ----------
    z : numpy array
        Contains elevation data in meters
    res: float
        Resolution in meters, assumed to be same in both directions

    Returns:
    --------
    numpy array containing aspect in degrees
    """

    aspect = np.empty_like(z_mat)
    aspect[:, :] = np.nan
    z_1 = z_mat[:-2, 2:]  # upper left
    z_2 = z_mat[1:-1, 2:]  # upper middle
    z_3 = z_mat[2:, 2:]  # upper right
    z_4 = z_mat[:-2, 1:-1]  # center left
   # z5 = z[ 1:-1, 1:-1] # center
    z_6 = z_mat[2:, 1:-1]  # center right
    z_7 = z_mat[:-2, :-2]  # lower left
    z_8 = z_mat[1:-1, :-2]  # lower middle
    z_9 = z_mat[2:, :-2]  # lower right
    dz_dx = ((z_3 + 2 * z_6 + z_9) - (z_1 + 2 * z_4 + z_7)) / (8 * res)
    dz_dy = ((z_1 + 2 * z_2 + z_3) - (z_7 + 2 * z_8 + z_9)) / (8 * res)
    dz_dx[dz_dx == 0.] = 1e-10
    angle = np.degrees(np.arctan(np.divide(dz_dy, dz_dx)))
    angle_mod = 90. * np.divide(dz_dx, np.absolute(dz_dx))
    aspect[1:-1, 1:-1] = 180. - angle + angle_mod
    return np.nan_to_num(aspect)


# def compute_slope_richdem_degrees(z_mat: np.ndarray, res: float) -> np.ndarray:
#     """ Compute slope using richdem package

#     Parameters:
#     -----------
#     z: numpy array
#         Contains elevation data for the concerned region in meters
#     res: float
#         Resolution in meters, assumed to be same in both directions

#     Returns:
#     --------
#     numpy array containing slope in degrees
#     """

#     z_rd = rd.rdarray(z_mat, no_data=-9999)
#     out = rd.TerrainAttribute(z_rd, attrib='slope_degrees', zscale=1 / res)
#     return out


# def compute_aspect_richdem_degrees(z_mat: np.ndarray, res: float) -> np.ndarray:
#     """ Compute aspect using richdem package

#     Parameters:
#     -----------
#     z: numpy array
#         Contains elevation data for the concerned region in meters
#     res: float
#         Resolution in meters, assumed to be same in both directions

#     Returns:
#     --------
#     numpy array containing aspect in degrees
#     """

#     z_rd = rd.rdarray(z_mat, no_data=-9999)
#     out = rd.TerrainAttribute(z_rd, attrib='aspect', zscale=1 / res)
#     return out
