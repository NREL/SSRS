""" Module for computing topographical and atmospheric layers """

import numpy as np
import richdem as rd
from datetime import datetime
from typing import Tuple
import random
from scipy import ndimage


def compute_orographic_updraft(
    wspeed: np.ndarray,
    wdirn: np.ndarray,
    slope: np.ndarray,
    aspect: np.ndarray,
    min_updraft_val: float = 0.
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


def compute_slope_richdem_degrees(z_mat: np.ndarray, res: float) -> np.ndarray:
    """ Compute slope using richdem package

    Parameters:
    -----------
    z: numpy array
        Contains elevation data for the concerned region in meters
    res: float
        Resolution in meters, assumed to be same in both directions

    Returns:
    --------
    numpy array containing slope in degrees
    """

    z_rd = rd.rdarray(z_mat, no_data=-9999)
    out = rd.TerrainAttribute(z_rd, attrib='slope_degrees', zscale=1 / res)
    return out


def compute_aspect_richdem_degrees(z_mat: np.ndarray, res: float) -> np.ndarray:
    """ Compute aspect using richdem package

    Parameters:
    -----------
    z: numpy array
        Contains elevation data for the concerned region in meters
    res: float
        Resolution in meters, assumed to be same in both directions

    Returns:
    --------
    numpy array containing aspect in degrees
    """

    z_rd = rd.rdarray(z_mat, no_data=-9999)
    out = rd.TerrainAttribute(z_rd, attrib='aspect', zscale=1 / res)
    return out


def get_above_threshold_speed_scalar(in_val, val):
    """ Converts updraft using threshold speed """
    if in_val > 1e-02:
        if in_val > val:
            fval = in_val
        else:
            fval = val * (np.exp((in_val / val)**5) - 1) / (np.exp(1) - 1)
    else:
        fval = 0.
    return fval


def get_above_threshold_speed(in_array: np.ndarray, threshold: float):
    """ vectorized form """
    return np.vectorize(get_above_threshold_speed_scalar)(in_array, threshold)


def compute_random_thermals(
    aspect: np.ndarray,  # terrain aspect, used for weighting
    thermal_intensity_scale: float  # describe strength of field
) -> np.ndarray:
    """ Returns field of smoothed random thermals from lognornal dist"""
    ysize, xsize = aspect.shape
    wt_init = np.zeros([ysize, xsize])
    border_x = int(0.1 * xsize)
    border_y = int(0.1 * ysize)
    # border with no thermals used to reduce problems of circling out of the domain
    for i in range(border_y, ysize - border_y):
        for j in range(border_x, xsize - border_x):
            wtfactor = 1000 + (abs(aspect[i, j] - 180.) / 180.) * \
                2000.  # weight prob using aspect: asp near 180 has highest prob of a thermal
            num1 = np.random.randint(1, int(wtfactor))
            if num1 == 5:
                wt_init[i, j] = np.random.lognormal(
                    thermal_intensity_scale + 3, 0.5)
            else:
                wt_init[i, j] = 0.0
    # est const = 2500 based on G Young 1.5 rule with 30 m grid
    # num1=np.random.randint(1,2000)

    # smooth the result to form Gaussian thermals
    wt = ndimage.gaussian_filter(wt_init, sigma=4, mode='constant')

    return wt


# def compute_thermals(
#     grid_size: Tuple[int, int],
#     res: float,
#     z: float,
#     zi: float,
#     wstar: float,
#     datetime
# ):
#     ny, nx = grid_size
#     if wstar == None:
#         wstar = 1e-8

#     # DEFINE UPDRAFT PARAMETERS
#     wgain = 1      # multiplier on vertical velocity
#     rgain = 1      # multiplier on radius
#     placement = 'random'  # line or random placement. line is for testing

#     # Get time-of-the-day and time-of-the-year gains
#     diurnalgain, seasonalgain = computeDatetimeGain(datetime)

#     # DEFINE AREA
#     xmin = 0
#     xmax = nx * res
#     ymin = 0
#     ymax = ny * res
#     print(xmin, xmax, ymin, ymax)
#     #res = 10
#     X = xmax - xmin    # length of test area, m
#     Y = ymax - ymin    # width of test area, m

#     # CALCULATE OUTER RADIUS
#     zzi = z / zi
#     r2 = (.102 * zzi**(1 / 3)) * (1 - (.25 * zzi)) * \
#         zi  # Equation 12 from updraft paper

#     # CALCULATE NUMBER OF UPDRAFTS IN GIVEN AREA
#     N = np.int(np.round(.6 * Y * X / (zi * r2)))

#     # SET PERTURBATION GAINS FOR EACH UPDRAFT
#     wgain = np.repeat(1, N)  # multiplier on vertical velocity
#     rgain = np.repeat(1, N)  # multiplier on radius
#     enableDiurnalWeight = True
#     enableSeasonalWeight = True
#     if enableDiurnalWeight:
#         wgain = [random.uniform(0.7 * diurnalgain, 1.3 * diurnalgain)
#                  for i in range(N)]
#     if enableSeasonalWeight:
#         rgain = [random.uniform(0.8 * seasonalgain, 1.2 * seasonalgain)
#                  for i in range(N)]

#     # PLACE UPDRAFTS
#     if placement == 'line':
#         xt = np.zeros(N)
#         yt = np.zeros(N)
#         for kn in np.arange(N):  # for each updraft
#             xt[kn] = (kn + 1) * X / (N + 1)
#             yt[kn] = (kn + 1) * Y / (N + 1)
#     elif placement == 'random':
#         xt = [random.randrange(xmin, xmax) for i in range(N)]
#         yt = [random.randrange(xmin, xmax) for i in range(N)]
#     else:
#         raise ValueError('Option not valid')

#     # DEFINE GRID OF TEST LOCATIONS
#     xc = np.arange(xmin, xmax, res)
#     yc = np.arange(ymin, ymax, res)
#     xx, yy = np.meshgrid(xc, yc, indexing='ij')
#     zz = np.ones(np.shape(xx)) * z  # create matrix of z values
#     w = np.zeros(np.shape(xx))  # create the empty w field

#     wpeak = np.zeros(np.shape(xx))  # create the empty temp fields
#     wl = np.zeros(np.shape(xx))  # create the empty temp fields
#     wd = np.zeros(np.shape(xx))  # create the empty temp fields
#     we = np.zeros(np.shape(xx))  # create the empty temp fields
#     ws = np.zeros(np.shape(xx))  # create the empty temp fields
#     r2 = np.zeros(np.shape(xx))  # create the empty temp fields
#     print(xc, yc)
#     for i in np.arange(len(xc)):
#         for j in np.arange(len(yc)):
#             # CALL UPDRAFT FUNCTION
#             print(i, j)
#             w[i, j], r2[i, j], wpeak[i, j], wl[i, j], wd[i, j], we[i, j], ws[i, j] = generateupdraft(
#                 xx[i, j], yy[i, j], zz[i, j], xt, yt, wstar, wgain, rgain, zi, A=X * Y, sflag=0)

#     return xx, yy, w


# def generateupdraft(x, y, z, xt, yt, wstar, wgain, rgain, zi, A, sflag=1):
#     '''
#         Input: x = Aircraft x position (m)
#                y = Aircraft y position (m)
#                z = Aircraft height above ground (m)
#                xt = Vector of updraft x positions (m)
#                yt = Vector of updraft y positions (m)
#                wstar = updraft strength scale factor,(m/s)
#                wgain = Vector of perturbations from wstar (multiplier)
#                rgain = Vector of updraft radius perturbations from average (multiplier)
#                zi = updraft height (m)
#                A = Area of test space
#                sflag = 0=no sink outside of thermals, 1=sink

#         Output: w = updraft vertical velocity (m/s)
#                 r2 = outer updraft radius, m
#                 wpeak = updraft velocity at center of thermal, m/s

#         After Allen (2006)
#     '''

#     # DEFINE UPDRAFT SHAPE FACTORS
#     r1r2shape = np.array([0.14, 0.25, 0.36, 0.47, 0.58, 0.69, 0.80])
#     Kshape = np.array([[1.5352, 2.5826, -0.0113, -0.1950, 0.0008],
#                        [1.5265, 3.6054, -0.0176, -0.1265, 0.0005],
#                        [1.4866, 4.8356, -0.0320, -0.0818, 0.0001],
#                        [1.2042, 7.7904, 0.0848, -0.0445, 0.0001],
#                        [0.8816, 13.9720, 0.3404, -0.0216, 0.0001],
#                        [0.7067, 23.9940, 0.5689, -0.0099, 0.0002],
#                        [0.6189, 42.7965, 0.7157, -0.0033, 0.0001]])

#     # CALCULATE DISTANCE TO EACH UPDRAFT
#     N = len(xt)
#     dist = ((x - xt)**2 + (y - yt)**2)**0.5

#     # CALCULATE AVERAGE UPDRAFT SIZE
#     zzi = z / zi
#     rbar = (.102 * zzi**(1 / 3)) * (1 - (.25 * zzi)) * zi  # eq 12

#     # CALCULATE AVERAGE UPDRAFT STRENGTH
#     wtbar = (zzi**(1 / 3)) * (1 - 1.1 * zzi) * wstar  # eq 11

#     # USE NEAREST UPDRAFT
#     upused = np.argmax(dist == np.min(dist))

#     # CALCULATE INNER AND OUTER RADIUS OF ROTATED TRAPEZOID UPDRAFT
#     r2 = max(10, rbar * rgain[upused])
#     if r2 < 600:
#         r1r2 = 0.0011 * r2 + 0.14
#     else:
#         r1r2 = 0.8
#     r1 = r1r2 * r2

#     # MULTIPLY AVERAGE UPDRAFT STRENGTH BY WGAIN FOR THIS UPDRAFT
#     wbar = wtbar * wgain[upused]  # add random perturbation

#     # CALCULATE STRENGTH AT CENTER OF ROTATED TRAPEZOID UPDRAFT
#     wpeak = (3 * wbar * ((r2**3) - (r2**2) * r1)) / \
#         ((r2**3) - (r1**3))  # eq 15

#     # CALCULATE UPDRAFT VELOCITY, eq 16
#     r = dist[upused]
#     rr2 = r / r2  # r/r2
#     if z < zi:  # if you are below the BL height
#         if r1r2 < .5 * (r1r2shape[0] + r1r2shape[1]):  # pick shape
#             k1 = Kshape[0, 0]
#             k2 = Kshape[0, 1]
#             k3 = Kshape[0, 2]
#             k4 = Kshape[0, 3]
#         elif r1r2 < .5 * (r1r2shape[1] + r1r2shape[2]):
#             k1 = Kshape[1, 0]
#             k2 = Kshape[1, 1]
#             k3 = Kshape[1, 2]
#             k4 = Kshape[1, 3]
#         elif r1r2 < .5 * (r1r2shape[2] + r1r2shape[3]):
#             k1 = Kshape[2, 0]
#             k2 = Kshape[2, 1]
#             k3 = Kshape[2, 2]
#             k4 = Kshape[2, 3]
#         elif r1r2 < .5 * (r1r2shape[3] + r1r2shape[4]):
#             k1 = Kshape[3, 0]
#             k2 = Kshape[3, 1]
#             k3 = Kshape[3, 2]
#             k4 = Kshape[3, 3]
#         elif r1r2 < .5 * (r1r2shape[4] + r1r2shape[5]):
#             k1 = Kshape[4, 0]
#             k2 = Kshape[4, 1]
#             k3 = Kshape[4, 2]
#             k4 = Kshape[4, 3]
#         elif r1r2 < .5 * (r1r2shape[5] + r1r2shape[6]):
#             k1 = Kshape[5, 0]
#             k2 = Kshape[5, 1]
#             k3 = Kshape[5, 2]
#             k4 = Kshape[5, 3]
#         else:
#             k1 = Kshape[6, 0]
#             k2 = Kshape[6, 1]
#             k3 = Kshape[6, 2]
#             k4 = Kshape[6, 3]
#         # inn=rr2;

#         # CALCULATE SMOOTH VERTICAL VELOCITY DISTRIBUTION (first part of eq 16)
#         ws = max((1 / (1 + (k1 * abs(rr2 + k3))**k2)) +
#                  k4 * rr2, 0)  # no neg updrafts
#     else:
#         ws = 0

#     # CALCULATE DOWNDRAFT VELOCITY AT THE EDGE OF THE UPDRAFT
#     if dist[upused] > r1 and rr2 < 2:
#         wl = (np.pi / 6) * np.sin(np.pi * rr2)
#     else:
#         wl = 0

#     if zzi > .5 and zzi <= .9:
#         wd = min(2.5 * wl * (zzi - 0.5), 0)
#     else:
#         wd = 0

#     w = wpeak * ws + wd * wbar  # scale updraft to actual velocity, eq 16

#     # CALCULATE ENVIRONMENT SINK VELOCITY
#     Aupdraft = N * np.pi * rbar**2  # total area taken by updrafts
#     assert Aupdraft < A, ValueError('Area of test space is too small')
#     if sflag:
#         swd = 2.5 * (zzi - 0.5)
#         we = -(wtbar * Aupdraft * (-swd)) / (A - Aupdraft)
#         we = [we] if isinstance(we, (int, float)) else we
#         we = [min(wei, 0) for wei in we]  # don't allow positive sink
#         if len(we) == 1:
#             we = we[0]
#     else:
#         we = 0

#     # STRETCH UPDRAFT TO BLEND WITH SINK AT EDGE
#     if dist[upused] > r1:  # if you are outside the core stretch
#         w = w * (1 - we / wpeak) + we

#     return w, r2, wpeak, wl, wd, we, ws


# def computeDatetimeGain(datetime):

#     tofday_hour = datetime.hour
#     tofyear_mon = datetime.month

#     # Compute the diurnal cycle weights

#     tsunrise = 6       # time of sunrise, in hours. Tipically 6AM. Given in military time
#     tsunset = 18       # time of sunset, in hours. Tipically 6PM. Given in military time
#     maxfactor = 1.2  # factor by which the quantities will be multiplied by at the solar zenith
#     # factor by which the quantities will be multiplied by at night (could be negative, indicating a stable boundary layer)
#     minfactor = 0

#     tday = np.linspace(tsunrise, tsunset, 100)
#     period = tsunrise - tsunset
#     phase = period / 2 + tsunrise
#     amp = (maxfactor - minfactor) / 2
#     offset = (maxfactor + minfactor) / 2
#     tofday_weight = amp * \
#         np.cos((2 * np.pi * (1 / period) * (tday - phase))) + offset

#     # Add bounds of simulation times
#     tday = np.concatenate(([0], tday, [24]))
#     tofday_weight = np.concatenate(([minfactor], tofday_weight, [minfactor]))

#     # Compute the seasonal cycle weights

#     # month in which the summer begins. Left for generality. 1=Jan; 12=Dec.
#     moSummerStart = 4
#     # month in which the summer ends. Left for generality. 1=Jan; 12=Dec.
#     moSummerEnd = 9
#     maxfactor = 1.1  # factor by which the quantities will be multiplied by middle of Summer
#     minfactor = 0.5  # factor by which the quantities will be multiplied by at other seasons

#     tyear = np.linspace(moSummerStart, moSummerEnd, 100)
#     period = moSummerStart - moSummerEnd
#     phase = period / 2 + moSummerStart
#     amp = (maxfactor - minfactor) / 2
#     offset = (maxfactor + minfactor) / 2
#     tofyear_weight = amp * \
#         np.cos((2 * np.pi * (1 / period) * (tyear - phase))) + offset

#     # Add bounds of simulation times
#     tyear = np.concatenate(([0], tyear, [12]))
#     tofyear_weight = np.concatenate(([minfactor], tofyear_weight, [minfactor]))

#     diurnalgain = np.interp(tofday_hour, tday, tofday_weight)
#     seasonalgain = np.interp(tofyear_mon, tyear, tofyear_weight)

#     return diurnalgain, seasonalgain


# # def plotThermal(xx, yy, wthermal):
# #     import matplotlib.colors as colors

# #     # PLOT FIELD AND CROSS SECTION
# #     fig, axs = plt.subplots(ncols=2, figsize=(15.4,5))

# #     norm=colors.TwoSlopeNorm(vcenter=0, vmin=-0.6, vmax=2.8)
# #     cm = axs[0].pcolormesh(xx,yy,wthermal, shading='auto', norm=norm, cmap='RdBu')
# #     cb = fig.colorbar(cm, ax=axs[0], label='w thermal [m/s]')
# #     axs[0].set_aspect('equal')
# #     axs[0].plot(xx[:,50],yy[:,50],'k-')

# #     axs[1].plot(xx[:,50],wthermal[:,50],'k-')

# #     axs[1].grid()
# #     axs[1].set_xlabel('y position, [m]')
# #     axs[1].set_ylabel('w [m/s]')
# #     plt.show()
