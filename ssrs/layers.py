""" Module for computing topographical and atmospheric layers """

import numpy as np
import richdem as rd
from datetime import datetime
from typing import Tuple
import random
from scipy import ndimage
from .hrrr import HRRR
from .config import Config
import pathos.multiprocessing as mp


def calcOrographicUpdraft(
    elev: np.ndarray,      # high-res 
    wspeed: np.ndarray,    # high-res
    wdirn: np.ndarray,     # high-res
    slope: np.ndarray,     # high-res
    aspect: np.ndarray,    # high-res
    res_terrain: float,    # high-res terrain data resolution
    res: float,            # low-res analysis resolution
    sx: np.ndarray = None, # low-res
    h: float = 80.,
    min_updraft_val: float = 0.
) -> np.ndarray:
    """
    Returns orographic updraft using wind speed, wind direction, slope
    and aspect.

    Returns low-res coarsened field
    """
    if sx is None:
        return calcOrographicUpdraft_original(wspeed, wdirn, slope, aspect,
                                              res_terrain, res, min_updraft_val)
    else:
        return calcOrographicUpdraft_improved(wspeed, wdirn, slope, aspect,
                                              elev, res_terrain, res, sx, h, min_updraft_val)


def calcOrographicUpdraft_original(
    wspeed: np.ndarray,
    wdirn: np.ndarray,
    slope: np.ndarray,
    aspect: np.ndarray,
    res_terrain: float, # high-res terrain data resolution
    res: float,         # low-res analysis resolution
    min_updraft_val: float = 0.
) -> np.ndarray:
    """ Return dimensional orographic updraft using Brandes and Ombalski model
    
        Receives high-res array, returns coarsened updraft field
    """

    sinterm = np.sin(np.deg2rad(slope))
    costerm = np.cos(np.deg2rad(aspect-wdirn))
    w0 = wspeed * sinterm * costerm

    w0_abovemin = np.maximum(min_updraft_val, w0)
    w0_abovemin_coarse = highRes2lowRes(w0_abovemin, res_terrain, res)
    
    return w0_abovemin_coarse


def calcOrographicUpdraft_improved(
    wspeed: np.ndarray,
    wdirn: np.ndarray,
    slope: np.ndarray,
    aspect: np.ndarray,
    elev: np.ndarray,
    res_terrain: float, # high-res terrain data resolution
    res: float,         # low-res analysis resolution
    sx: np.ndarray = None, # low-res
    h : float = 80.,
    min_updraft_val: float = 0.
) -> np.ndarray:
    """ Return dimensional orographic updraft using our improved model"""

    # Get coarse orographic using BO04's model
    w0prime = calcOrographicUpdraft_original(wspeed, wdirn, slope, aspect,
                                             res_terrain, res, min_updraft_val)

    # Compute height adjustment
    print('Computing adjusting factors from improved model (1/3)..', end='\r')
    a=0.00004;  b=0.0028;  c=0.8;  d=0.35;  e=0.095;  f=-0.09
    slope_lowres = highRes2lowRes(slope, res_terrain, res)
    factor_height = ( a*h**2 + b*h + c ) * d**(-np.cos(np.deg2rad(slope_lowres)) + e) + f  # low-res
    # Compute Sx adjustment
    print('Computing adjusting factors from improved model (2/3)..', end='\r')
    factor_sx = 1 + np.tan(np.deg2rad(sx))
    # Compute terrain complexity adjustment
    print('Computing adjusting factors from improved model (3/3)..', end='\r')
    filterSize_in_m = 500
    filterSize = int(np.floor(filterSize_in_m/res_terrain))
    elev_lowres = highRes2lowRes(elev, res_terrain, res)
    local_zmean = ndimage.generic_filter(elev_lowres, np.mean, footprint=np.ones((filterSize,filterSize)) )
    local_zmin  = ndimage.generic_filter(elev_lowres, np.min,  footprint=np.ones((filterSize,filterSize)) )
    local_zmax  = ndimage.generic_filter(elev_lowres, np.max,  footprint=np.ones((filterSize,filterSize)) )
    tc = (local_zmean - local_zmin) / (local_zmax - local_zmin)
    factor_tc = 1 + tc*(h/40)  # low-res
    
    # Combine all factors
    print('Computing adjusting factors from improved model..       ')
    F = factor_tc * factor_sx / factor_height

    # Compute improved value and remove NaNs from last column and row (product of Sx calculation)
    wo_imp = F*w0prime
    wo_imp[np.isnan(wo_imp)] = 0.0

    return wo_imp


def highRes2lowRes(field, res_h, res_l, sigma_in_m=30):
    """
    Upsamples a high-resolution field to a lower resolution

    It works by first creating a gaussian filter with std of 30 m on the high-res
    field, keeping the same resolution. Then, the data is resampled at the target
    low resolution (coarsened) using scipy.ndimage.zoom.

    The value of the filter can be changed using the sigma_in_m varible.
    """

    ratio = res_h/res_l
    if ratio == 1:
        # same resolution, nothing to do here
        return field

    sigma = sigma_in_m/res_h
    if sigma<=1:
        print('    ! Low resolution terrain data. Consider ',
              'increasing the resolution (`resolution_terrain`).')

    filtered = ndimage.gaussian_filter(field, sigma=sigma)
    field_coarse = ndimage.zoom(filtered,  (ratio, ratio))

    return field_coarse


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


def calcSlopeDegrees(z_mat: np.ndarray, res: float):
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


def calcAspectDegrees(z_mat: np.ndarray, res: float):
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
    # change reference
    aspect = (-aspect+90)%360
    return np.nan_to_num(aspect)

def blurQuantity(quant: np.ndarray, res: float, h: float):
    '''
    Calculate a blurred version of a quantity quant based
    on the height h
    '''

    sigma_in_m = min(0.8*h + 16, 300) # size of kernel in meters
    return ndimage.gaussian_filter(quant, sigma=sigma_in_m/res)


def calcSx(xgrid, ygrid, zagl, A, dmax=500, method='linear', verbose=True):
    '''
    Sx is a measure of topographic shelter or exposure relative to a particular
    wind direction. Calculates a whole map for all points (xi, yi) in the domain.
    For each (xi, yi) pair, it uses all v points (xv, yv) upwind of (xi, yi) in
    the A wind direction, up to dmax.
    Winstral, A., Marks D. "Simulating wind fields and snow redistribution using
        terrain-based parameters to model snow accumulation and melt over a semi-
        arid mountain catchment" Hydrol. Process. 16, 3585–3603 (2002)
    Usage
    =====
    xx, yy : array
        meshgrid arrays of the region extent coordinates.
    zagl: arrayi, xr.DataArray
        Elevation map of the region
    A: float
        Wind direction (deg, wind direction convention)
    dmax: float
        Upwind extent of the search
    method: string
        griddata interpolation method. Options are 'nearest', 'linear', 'cubic'.
        Recommended linear or cubic.
    '''
    from scipy.interpolate import griddata
    import xarray as xr
    
    xx, yy = np.meshgrid(xgrid, ygrid, indexing='ij')

    # get resolution (assumes uniform resolution)
    res = xx[1,0] - xx[0,0]
    npoints = 1+int(dmax/res)
    if dmax < res:
        raise ValueError('dmax needs to be larger or equal to the resolution of the grid')
    
    # Get upstream direction
    A = A%360
    if    A==0:   upstreamDirX=0;  upstreamDirY=-1
    elif  A==90:  upstreamDirX=-1; upstreamDirY=0
    elif  A==180: upstreamDirX=0;  upstreamDirY=1
    elif  A==270: upstreamDirX=1;  upstreamDirY=0
    elif  A>0  and A<90:   upstreamDirX=-1; upstreamDirY=-1
    elif  A>90  and A<180:  upstreamDirX=-1; upstreamDirY=1
    elif  A>180 and A<270:  upstreamDirX=1;  upstreamDirY=1
    elif  A>270 and A<360:  upstreamDirX=1;  upstreamDirY=-1

    # change angle notation
    ang = np.deg2rad(270-A)

    # array for interpolation using griddata
    points = np.array( (xx.flatten(), yy.flatten()) ).T
    if isinstance(zagl, xr.DataArray):
        zagl = zagl.values
    values = zagl.flatten()

    # create rotated grid. This way we sample into a interpolated grid that has the exact points we need
    xmin = min(xx[:,0]);  xmax = max(xx[:,0])
    ymin = min(yy[0,:]);  ymax = max(yy[0,:])
    if A%90 == 0:
        # if flow is aligned, we don't need a new grid
        xrot = xx[:,0]
        yrot = yy[0,:]
        xxrot = xx
        yyrot = yy
        elevrot = zagl
    else:
        xrot = np.arange(xmin, xmax+0.1, abs(res*np.cos(ang)))
        yrot = np.arange(ymin, ymax+0.1, abs(res*np.sin(ang)))
        xxrot, yyrot = np.meshgrid(xrot, yrot, indexing='ij')
        elevrot = griddata( points, values, (xxrot, yyrot), method=method )

    # create empty rotated Sx array
    Sxrot = np.empty(np.shape(elevrot));  Sxrot[:,:] = np.nan

    for i, xi in enumerate(xrot):
        if verbose: print(f'Computing shelter angle Sx.. {100*(i+1)/len(xrot):.1f}%  ', end='\r')
        for j, yi in enumerate(yrot):

            # Get elevation profile along the direction asked
            isel = np.linspace(i-upstreamDirX*npoints+upstreamDirX, i, npoints, dtype=int)
            jsel = np.linspace(j-upstreamDirY*npoints+upstreamDirY, j, npoints, dtype=int)
            try:
                xsel = xrot[isel]
                ysel = yrot[jsel]
                elev = elevrot[isel,jsel]
            except IndexError:
                # At the borders, can't get a valid positions
                xsel = np.zeros(np.size(isel))  
                ysel = np.zeros(np.size(jsel))
                elev = np.zeros(np.size(isel))

            # elevation of (xi, yi), for convenience
            elevi = elev[-1]

            try:
                Sxrot[i,j] = np.nanmax(np.rad2deg( np.arctan( (elev[:-1] - elevi)/(((xsel[:-1]-xi)**2 + (ysel[:-1]-yi)**2)**0.5) ) ))
            except IndexError:
                raise

    if verbose: print(f'Computing shelter angle Sx..        ')
    # interpolate results back to original grid
    pointsrot = np.array( (xxrot.flatten(), yyrot.flatten()) ).T
    Sx = griddata( pointsrot, Sxrot.flatten(), (xx, yy), method=method )

    return Sx


def calcSlopeDegrees_richdem(z_mat: np.ndarray, res: float) -> np.ndarray:
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


def calcAspectDegrees_richdem(z_mat: np.ndarray, res: float) -> np.ndarray:
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
            #fval = val * (np.exp((in_val / val)**5) - 1) / (np.exp(1) - 1)
            fval=0.0
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
    """
    Returns field of smoothed random thermals from lognornal dist

    This method is only used if thermal_model is 'naive'
    """
    ysize, xsize = aspect.shape
    wt_init = np.zeros([ysize, xsize])
    border_x = int(0.05 * xsize)
    border_y = int(0.05 * ysize)
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

#def compute_terrain_linearity_index(
#    elevation: np.ndarray,
#    aspect: np.ndarray,
#    min_updraft_val: float = 0.
#) -> np.ndarray:
#    """ UNFINISHED Returns a measure of terrain linearity based on contiguous regions of similar aspect """
#    ysize, xsize = aspect.shape
#    for m in range(1, ysize):
#        for n in range(1, xsize):
#            count[m,n]=0.
#            for i in range(m-1,m+1)
#                for j in range(n-1,n+1)
#                    if abs(aspect[i,j]-aspect[m,n])<5:
#                        count[m,n]=count[m,n]+1
#   
#    return tli
        
        
# def computeDatetimeGain(datetime):
#
#     tofday_hour = datetime.hour
#     tofyear_mon = datetime.month
#
#     # Compute the diurnal cycle weights
#
#     tsunrise = 6       # time of sunrise, in hours. Tipically 6AM. Given in military time
#     tsunset = 18       # time of sunset, in hours. Tipically 6PM. Given in military time
#     maxfactor = 1.2  # factor by which the quantities will be multiplied by at the solar zenith
#     # factor by which the quantities will be multiplied by at night (could be negative, indicating a stable boundary layer)
#     minfactor = 0
#
#     tday = np.linspace(tsunrise, tsunset, 100)
#     period = tsunrise - tsunset
#     phase = period / 2 + tsunrise
#     amp = (maxfactor - minfactor) / 2
#     offset = (maxfactor + minfactor) / 2
#     tofday_weight = amp * \
#         np.cos((2 * np.pi * (1 / period) * (tday - phase))) + offset
#
#     # Add bounds of simulation times
#     tday = np.concatenate(([0], tday, [24]))
#     tofday_weight = np.concatenate(([minfactor], tofday_weight, [minfactor]))
#
#     # Compute the seasonal cycle weights
#
#     # month in which the summer begins. Left for generality. 1=Jan; 12=Dec.
#     moSummerStart = 4
#     # month in which the summer ends. Left for generality. 1=Jan; 12=Dec.
#     moSummerEnd = 9
#     maxfactor = 1.1  # factor by which the quantities will be multiplied by middle of Summer
#     minfactor = 0.5  # factor by which the quantities will be multiplied by at other seasons
#
#     tyear = np.linspace(moSummerStart, moSummerEnd, 100)
#     period = moSummerStart - moSummerEnd
#     phase = period / 2 + moSummerStart
#     amp = (maxfactor - minfactor) / 2
#     offset = (maxfactor + minfactor) / 2
#     tofyear_weight = amp * \
#         np.cos((2 * np.pi * (1 / period) * (tyear - phase))) + offset
#
#     # Add bounds of simulation times
#     tyear = np.concatenate(([0], tyear, [12]))
#     tofyear_weight = np.concatenate(([minfactor], tofyear_weight, [minfactor]))
#
#     diurnalgain = np.interp(tofday_hour, tday, tofday_weight)
#     seasonalgain = np.interp(tofyear_mon, tyear, tofyear_weight)
#
#     return diurnalgain, seasonalgain


def getRandomPointsWeighted (weight, n, nRealization=1):
    
    normalweight = weight/np.sum(weight)
    
    choicesum = np.zeros_like(weight).flatten()
    for i in range(nRealization):
        randindices = np.random.choice( np.arange(np.size(choicesum)), size=n, replace = False, p=normalweight.flatten())
        # create a current-iteration result
        choice = np.zeros_like(weight).flatten()
        choice[randindices] = 1
        # accumulate
        choicesum = choicesum +choice

    # sum of realizations
    choicesum = choicesum.reshape(np.shape(weight))
    # last realization, choice
    choice = choice.reshape(np.shape(weight))
    
    return choice, choicesum


def getObs_maxw(height, hrrr, southwest_lonlat, extent, res):
    # Add a weight based on experimental observations at the WFIP2 site if height is low
    import xarray as xr
    import os

    if height<=200:
        my_path = os.path.abspath(os.path.dirname(__file__))
        wfip = xr.open_dataset(os.path.join(my_path,'updraft','updraft_conditions_wfip2.nc'))
        rho = 1.225 # kg/m^3
        cp = 1005   # J/(kg*K)

        # Get mean wspd
        u, xx, yy = hrrr.get_single_var_on_grid(':UGRD:80 m above ground',       southwest_lonlat, extent, res)   # u component of the wind at 80 AGL
        v, xx, yy = hrrr.get_single_var_on_grid(':UGRD:80 m above ground',       southwest_lonlat, extent, res)   # u component of the wind at 80 AGL
        wspd = (u**2+v**2)**0.5
        meanwspd = np.mean(wspd)

        # Get heat flux
        gflux_Wm2, xx, yy = hrrr.get_single_var_on_grid(':(GFLUX):',      southwest_lonlat, extent, res)   # ground heat flux
        sensible,  xx, yy = hrrr.get_single_var_on_grid(':SHTFL:surface', southwest_lonlat, extent, res)   # sensible heat flux
        latent,    xx, yy = hrrr.get_single_var_on_grid(':LHTFL:surface', southwest_lonlat, extent, res)   # latent heat flux
        hfx = (sensible + latent - gflux_Wm2 )/(rho*cp)
        meanhfx = np.mean(hfx)

        # Get the vertical speed statistics
        wfiph = wfip.interp(height=height).squeeze(drop=True)
        wdata = wfiph.where( (wfiph.wind_speed>meanwspd-1 ) & ( wfiph.wind_speed<meanwspd+1) &
                             (wfiph.hfx>meanhfx-0.025 )     & ( wfiph.hfx<meanhfx+0.025),       drop=True )['vertical_air_velocity']#.to_dataframe().agg(['count','min','mean','max','std'])
        wmax = wdata.max().values
        
        return wmax


def compute_thermals_3d(
    aspect: np.ndarray,  # terrain aspect, used for weighting
    southwest_lonlat: Tuple[float, float], 
    extent: Tuple[float, float, float, float],  # xmin, ymin, xmax, ymax
    res: int,   # uniform resolution
    time,  # either tuple with [y, m, d, hour], or datetimeobject
    height: float = 150,
    wfipInformed: bool = True
    ) -> np.ndarray:
    '''
    Returns field of thermals based on Allen (2006)
    '''
    
    # TODO: Loop over a list of `time`s
    
    # Get string of time to pass to HRRR. Time can be passed either a tuple of datetime object
    if isinstance(time, datetime):
        timestr = f' {time.year}-{time.month:02d}-{time.day:02d} {time.hour:02d}:{time.minute:02d}'
    else:
        timestr = f'{time[0]}-{time[1]:02d}-{time[2]:02d} {time[3]:02d}:00'

    # Get hrrr data
    hrrr = HRRR(valid_date = timestr)

    # Compute convective velocity
    wstar,  xx, yy = hrrr.get_convective_velocity(southwest_lonlat, extent, res=res)
    # Compute albedo
    albedo, xx, yy = hrrr.get_albedo(southwest_lonlat, extent, res)
    # Get boundary layer height
    zi,     xx, yy = hrrr.get_single_var_on_grid(':(HPBL):',  # boundary layer height
                                                 southwest_lonlat,
                                                 extent,
                                                 res)
    print(f'albedo is {albedo}')
    wstar  = wstar.values
    try:
        albedo = albedo.values
    except AttributeError:  # it's an array already
        pass
    zi = zi[list(zi.keys())[0]].values


    if np.mean(zi) == np.nan:
        raise ValueError(f'The value obtained for the boundary layer height contains NaNs.',\
                         f'HRRR data is imcomplete at the site and time of interest.')

    # Define updraft shape factors
    r1r2shape = np.array([0.14, 0.25, 0.36, 0.47, 0.58, 0.69, 0.80])
    Kshape = np.array([[1.5352, 2.5826, -0.0113, -0.1950, 0.0008],
                       [1.5265, 3.6054, -0.0176, -0.1265, 0.0005],
                       [1.4866, 4.8356, -0.0320, -0.0818, 0.0001],
                       [1.2042, 7.7904,  0.0848, -0.0445, 0.0001],
                       [0.8816, 13.9720, 0.3404, -0.0216, 0.0001],
                       [0.7067, 23.9940, 0.5689, -0.0099, 0.0002],
                       [0.6189, 42.7965, 0.7157, -0.0033, 0.0001]])


    # Create weight for likeliness of thermals in space
    albedofactor = (0.1/(albedo)**0.5)
    spatialWeight = ( wstar**1 + albedofactor )**2
    # Mask the edges so no thermals there
    fringe= 2000 # in [m]
    ifringe = int(fringe/res)
    spatialWeight[0:ifringe,:] = spatialWeight[-ifringe:,:] = 0
    spatialWeight[:,0:ifringe] = spatialWeight[:,-ifringe:] = 0

    # Get thermal parameters
    ziavg = np.mean(zi)
    zzi = height/zi
    zziavg = height/ziavg
    if ziavg > 300:
        ValueError(f'The boundary layer is too shallow for thermals')

    # Calcualte average updraft size
    rbar=(.102*zzi**(1/3))*(1-(.25*zzi))*zi

    # Calculate average updraft strength (G. Young)
    wT = wstar * 0.85 * (zzi**(1/3)) * (1.3-zzi)

    # Size gain around a mean, based on albedo
    rgain =1.4*(0.4/(albedo))

    # Calculate inner and outer radius of rotated trapezoid updraft
    r2 = rbar*rgain;  r2[r2<10] = 10
    r1r2 = 0.0011*r2+0.14
    r1r2[r2>600] = 0.8
    r1 = r1r2*r2

    # Determine number of thermals
    nThermals = int ( 0.6*(extent[2]-extent[0])*(extent[3]-extent[1])/(ziavg*np.mean(r2)) )

    # Create strength gains, based on wstar
    wgain = 0.7*wstar

    # Multiply average updraft strength by the gain
    wTbar = wT*wgain

    # Calculate strength at center of rotated trapezoid updraft
    wpeak=(3*wTbar*((r2**3)-(r2**2)*r1)) / ((r2**3)-(r1**3))

    # Create a realization of thermal's center location
    print(f'Creating {nThermals} thermals. The average boundary layer height is {ziavg:.1f} m')
    wt_init, sumOfRealizations = getRandomPointsWeighted(weight=spatialWeight, n=nThermals, nRealization=1)

    # Get distances to closest thermal center
    wt_init1 = np.zeros_like(wt_init)
    wt_init1[wt_init>0]=1
    dist = ndimage.distance_transform_edt(np.logical_not(wt_init1)) * res

    # Calculate updraft velocity
    r=dist
    rr2=r/r2

    # Calculate shape parameters
    k1 = np.ones_like(r1r2)
    k2 = np.ones_like(r1r2)
    k3 = np.ones_like(r1r2)
    k4 = np.ones_like(r1r2)
    k1 = k1*Kshape[6,0];                                      k2 = k2*Kshape[6,1];                                      k3 = k3*Kshape[6,2];                                      k4 = k4*Kshape[6,3]
    k1[r1r2<(0.5*r1r2shape[6]+r1r2shape[5])] = Kshape[5,0];   k2[r1r2<(0.5*r1r2shape[6]+r1r2shape[5])] = Kshape[5,1];   k3[r1r2<(0.5*r1r2shape[6]+r1r2shape[5])] = Kshape[5,2];   k4[r1r2<(0.5*r1r2shape[6]+r1r2shape[5])] = Kshape[5,3]
    k1[r1r2<(0.5*r1r2shape[5]+r1r2shape[4])] = Kshape[4,0];   k2[r1r2<(0.5*r1r2shape[5]+r1r2shape[4])] = Kshape[4,1];   k3[r1r2<(0.5*r1r2shape[5]+r1r2shape[4])] = Kshape[4,2];   k4[r1r2<(0.5*r1r2shape[5]+r1r2shape[4])] = Kshape[4,3]
    k1[r1r2<(0.5*r1r2shape[4]+r1r2shape[3])] = Kshape[3,0];   k2[r1r2<(0.5*r1r2shape[4]+r1r2shape[3])] = Kshape[3,1];   k3[r1r2<(0.5*r1r2shape[4]+r1r2shape[3])] = Kshape[3,2];   k4[r1r2<(0.5*r1r2shape[4]+r1r2shape[3])] = Kshape[3,3]
    k1[r1r2<(0.5*r1r2shape[3]+r1r2shape[2])] = Kshape[2,0];   k2[r1r2<(0.5*r1r2shape[3]+r1r2shape[2])] = Kshape[2,1];   k3[r1r2<(0.5*r1r2shape[3]+r1r2shape[2])] = Kshape[2,2];   k4[r1r2<(0.5*r1r2shape[3]+r1r2shape[2])] = Kshape[2,3]
    k1[r1r2<(0.5*r1r2shape[2]+r1r2shape[1])] = Kshape[1,0];   k2[r1r2<(0.5*r1r2shape[2]+r1r2shape[1])] = Kshape[1,1];   k3[r1r2<(0.5*r1r2shape[2]+r1r2shape[1])] = Kshape[1,2];   k4[r1r2<(0.5*r1r2shape[2]+r1r2shape[1])] = Kshape[1,3]
    k1[r1r2<(0.5*r1r2shape[1]+r1r2shape[0])] = Kshape[0,0];   k2[r1r2<(0.5*r1r2shape[1]+r1r2shape[0])] = Kshape[0,1];   k3[r1r2<(0.5*r1r2shape[1]+r1r2shape[0])] = Kshape[0,2];   k4[r1r2<(0.5*r1r2shape[1]+r1r2shape[0])] = Kshape[0,3]

    # Calculate the smooth vertical velocity distribution
    ws = (1/(1+(k1*abs(rr2+k3))**k2)) + k4*rr2
    # no negative updrafts
    ws[ws<0] = 0
    # Set to zero if above the boundary layer
    ws[zi<height] = 0

    # Calculate downdraft velocity at edges of updrafts
    wl = (np.pi/6)*np.sin(rr2*np.pi)
    wl[( (dist<r1) | (rr2>2))] = 0
    wd = 2.5*wl*(zzi-0.5)
    wd[((zzi<0.5) | (zzi>0.9))] = 0
    wd[wd<0]=0

    # Combine fields
    w = wpeak*ws + wd*wTbar

    # Scale it to fit experimental data (optional)
    if wfipInformed:
        if height<= 200:
            wmax = getObs_maxw(height, hrrr, southwest_lonlat, extent, res)
            w = w*wmax/np.max(w)
        else:
            print('The height requested is higher than observations. Skipping correction.')

    # Environment sink
    # we = np.zeros_like(w)
    # Stretch updraft field to blend with sink at edge
    # w[dist>r1] = (w*(1-we/wpeak)+we)[dist>r1]
    
    print(f'compute_thermals_3d returning a thermal field of shape {np.shape(w)}')
    return w



# def compute_adjusted_orographic_updraft (
#     wspeedAtRefHeight: np.ndarray,
#     wdirn: np.ndarray,
#     elevation: np.ndarray,
#     tc: float,
#     res: float,
#     h: float = 80,
#     #min_updraft_val: float = -5.,
#     returnOriginal: bool =False
# ) -> np.ndarray:
#     '''
#     Returns the dimensional adjusted orographic updraft value
   
#     Parameters:
#     ===========
#     wspeedAtRefHeight:
#         Wind speed at a reference height, often 80 m AGL
#     wdirn:
#         Wind direction at a reference height
#     elevation:
#         Your z_mat
#     tc:
#         Terrain complexity.
#         Guideline: 0.2 for WY, 0.8 for Appalachian
#     res:
#         Resolution
#     h:
#         Height of interest. Defaults to 80
#     min_updraft_val:
#         Minimum value used to clip the final adjusted model. Placeholder.
#     returnOriginal:
#         Whether or not also return original model
       
#     Returns:
#     ========
#     w0adj:
#         numpy array containing dimensional w0 adjusted value
#     '''
   
#     # Constants for height adjustment
#     a=0.00004;  b=0.0028;   c=0.8
#     d=0.35;     e=0.095;    f= -0.09

#     # Compute dummy grid with proper resolution
#     xx, yy = np.meshgrid(np.arange(0,res*np.shape(elevation)[0], res),
#                          np.arange(0,res*np.shape(elevation)[1], res), indexing='ij')
#     # Compute shelterness angle (180 for flipped behavior)
#     wdir_sx =  (np.mean(wdirn)+90)%360  # wdir for sx due to weird convention
#     sx400 = calcSx(xx, yy, elevation, np.mean(wdir_sx)+180, 400)

#     # Get terrain quantities
#     sigma_in_m = min(0.8*h + 16, 300) # size of kernel in meters
#     zblur = ndimage.gaussian_filter(elevation, sigma=sigma_in_m/res)
#     slopeblur = calcSlopeDegrees(zblur, res)
#     aspectblur = calcAspectDegrees(zblur, res)
#     slope = calcSlopeDegrees(elevation, res)
#     aspect = calcAspectDegrees(elevation, res)

#     # Calculate adjusting factors
#     factor_height = ( a*h**2 + b*h + c ) * d**(-np.cos(np.deg2rad(slopeblur)) + e) + f
#     factor_sx = 1 + np.tan(np.deg2rad(sx400))
#     factor_tc = 1 + tc
#     # Combine all factors
#     F = factor_tc * factor_sx / factor_height

#     # Compute dimensional w0 based on original model and a reference wind speed at a reference height
#     w0 =  wspeedAtRefHeight * np.sin(np.deg2rad(slope)) * np.cos(np.deg2rad(((-np.mean(wdirn)+90)%360)-aspect))
#     w0blur = wspeedAtRefHeight * np.sin(np.deg2rad(slopeblur)) * np.cos(np.deg2rad(((-np.mean(wdirn)+90)%360)-aspectblur))
    
#     # Adjust w0
#     w0adj =  F * w0blur

#     if returnOriginal:
#         return w0adj, w0
#     else:
#         return w0adj

