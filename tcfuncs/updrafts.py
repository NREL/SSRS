from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import (griddata)

pd.set_option('display.float_format', lambda x: '%.1f' % x)

min_updraft = 0.  # not too small or else its affects the linear system solve


def weibull_percentage_above_threshold(
        updraft_median: np.ndarray,
        weibull_k: float,
        updraft_threshold: float
) -> np.ndarray:
    """return the percentage of updraft being more than the threshold based 
    on Weibull disstribution with givenn median """

    lamda = np.divide(updraft_median, np.log(2) ** (1 / weibull_k))
    lamda = lamda.clip(min=1e-5)
    # print(np.amin(lamda), np.amax(lamda))
    exponent = np.power(np.divide(updraft_threshold, lamda), weibull_k)
    perc = np.exp(-exponent).clip(min=min_updraft)
    # print(np.amin(perc), np.amax(perc))
    return perc


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
        timestamp: datetime,
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

    # Convert wind speed and wind diren data into components for interpolation
    wtk_wdirn = wtk_wdirn_deg * np.pi / 180.
    # wtk_wdirn = 1.5*np.pi*np.ones(wtk_wdirn_deg.shape)
    # wtk_wspeed = np.ones(wtk_wspeed.shape)
    # tr_slope = np.ones(tr_slope.shape)
    wtk_easterly = np.multiply(wtk_wspeed, np.sin(wtk_wdirn))
    wtk_northerly = np.multiply(wtk_wspeed, np.cos(wtk_wdirn))

    # interpolate individual wind speed components
    tr_xgrid = np.arange(tr_extent[0], tr_extent[1], tr_res)
    tr_ygrid = np.arange(tr_extent[2], tr_extent[3], tr_res)
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
        return orographs


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
        timestamp: datetime,
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

    # print('Thermals: {0:s}'.format(timestamp.strftime('%I %p, %x')), end=" ")

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

# %% junk from previous versions
# def compute_updrafts_wtk_randomly(
#     sample_count: int,
#     tr_xybnd: np.ndarray,
#     tr_features: np.ndarray,
#     wtk_indices: np.ndarray,
#     wtk_xygrid: np.ndarray,
#     wtk_source: str,
#     wtk_years: List[int],
#     wtk_months: List[int],
#     wtk_hours: List[int],
#     tr_res: float,
#     oro_height: int,
#     thermal_height: int,
#     interp_type: str,
#     max_cpu_usage: int,
# ) -> List:
#     """Extracts orographic updraft speed at random time instances within a
#     provided time range """

#     # initialize
#     print('\n--- Computing updrafts at randomly selected time instants')
#     n_cpu = min(multiprocessing.cpu_count(), sample_count, max_cpu_usage)
#     print('Requested {0:d} updraft calculations using {1:d} cores'.format(
#         sample_count, n_cpu))
#     print('Years: ', wtk_years, '\nMonths: ',
#           wtk_months, '\nHours: ', wtk_hours)

#     # create a list of random ids
#     datetime_list = []
#     for _ in range(sample_count):
#         rnd_year = random.choice(wtk_years)
#         rnd_month = random.choice(wtk_months)
#         rnd_day = random.choice(range(*monthrange(rnd_year, rnd_month))) + 1
#         rnd_hour = random.choice(wtk_hours)
#         datetime_list.append(datetime(rnd_year, rnd_month,rnd_day,rnd_hour))
#     print(datetime_list)
#    # compute orographic updrafts in parallel
#     with multiprocessing.Pool(n_cpu) as pool:
#         out_oro = pool.map(
#             lambda idt: compute_orographic_updraft_wtk(
#                 idt,
#                 tr_res,
#                 tr_xybnd,
#                 tr_features[1, :, :],
#                 tr_features[2, :, :],
#                 wtk_source,
#                 wtk_indices,
#                 wtk_xygrid,
#                 oro_height,
#                 interp_type
#             ),
#             datetime_list)
#     print('done with orographic', flush=True)
#     with multiprocessing.Pool(n_cpu) as pool:
#         out_thermal = pool.map(
#             lambda idt: compute_deardoff_velocity_wtk(
#                 idt,
#                 tr_res,
#                 tr_xybnd,
#                 wtk_source,
#                 wtk_indices,
#                 wtk_xygrid,
#                 thermal_height,
#                 interp_type
#             ),
#             datetime_list)
#     return datetime_list, out_oro, out_thermal


# # %% Summary
# def compute_orographic_updraft_wtk_summary(
#         tr_xygrid,
#         tr_slope,
#         tr_aspect,
#         wtk_source,
#         wtk_indices,
#         wtk_xygrid,
#         wtk_height,
#         wtk_years,
#         wtk_months,
#         wtk_hours,
#         interp_type
# ):
#     """Extracts orographic updraft speed for a randomly selected time
#     using WTK"""

#     no_of_data_points =
# len(wtk_years) * len(wtk_months) * 28 * len(wtk_hours)
#     # no_of_points_for_stats = min(no_of_data_points,
# max(5, no_of_data_points // 5))
#     no_of_points_for_stats = min(no_of_data_points, 36)

#     # print(no_of_data_points, no_of_points_for_stats)

#     output = compute_orographic_updraft_wtk_randomly(
#         no_of_points_for_stats,
#         tr_xygrid,
#         tr_slope,
#         tr_aspect,
#         wtk_source,
#         wtk_indices,
#         wtk_xygrid,
#         wtk_height,
#         wtk_years,
#         wtk_months,
#         wtk_hours,
#         interp_type
#     )
#     print('i am done')
#     stat_mat = []
#     for out in output:
#         run_id, updraft_data, _, _ = out
#         stat_mat.append(updraft_data)
#     print(len(stat_mat), np.shape(stat_mat[0]))
#     return stat_mat[0], np.mean(np.asarray(stat_mat), axis=0), np.std(np.asarray(stat_mat), axis=0)


# # %% Computed percentage time orograhic updraft greater than threshhold
# def compute_orographic_updraft_wtk_summary_old(
#         tr_xygrid,
#         tr_slope,
#         tr_aspect,
#         wtk_source,
#         wtk_indices,
#         wtk_xygrid,
#         wtk_height,
#         wtk_year,
#         wtk_months,
#         wtk_hours,
#         interp_type
# ):
#     """Extracts orograhic speed on a given WTK index """

#     print('\nExtracting orographic updraft summary for')
#     print('year ' + str(wtk_year) + ', months ' +
#           ",".join([str(xi) for xi in wtk_months]), ', hours ' +
#           ",".join([str(xi) for xi in wtk_hours]))

#     if wtk_source == 'AWS':
#         import h5pyd as hsds
#         fname = '/nrel/wtk/conus/wtk_conus_' + str(wtk_year) + '.h5'
#     elif wtk_source == 'EAGLE':
#         import h5py as hsds
#         fname = '/datasets/WIND/conus/v1.0.0/wtk_conus_' + \
#             str(wtk_year) + '.h5'

#     wtk_vars = ['windspeed', 'winddirection']
#     wtk_columns = [x + '_' + str(int(wtk_height)) + 'm' for x in wtk_vars]

#     with hsds.File(fname, mode='r') as f:
#         time_index = pd.to_datetime(f['time_index'][...].astype(str))
#         # df = pd.DataFrame(index=time_index, columns=wtk_columns)
#         month_bool = np.zeros(time_index.shape, dtype=bool)
#         for wtk_month in wtk_months:
#             month_bool = np.logical_or(
#                 time_index.month == wtk_month, month_bool)
#         wtk_bool = np.logical_and.reduce((month_bool,
#                                           time_index.hour >= min(wtk_hours),
#                                           time_index.hour <= max(wtk_hours)))
#         wtk_tindex = np.where(wtk_bool)[0]
#         wtk_n = wtk_indices.shape[0]
#         wtk_data = np.empty((len(wtk_columns), sample_n, wtk_n))
#         for k, wtk_column in enumerate(wtk_columns):
#             wtk_data[k, :, :] = f[wtk_column][wtk_tindex,
#                                               wtk_indices]
#  / f[wtk_column].attrs['scale_factor']

#     updraft_data = np.empty(
#         ((sample_n, tr_xygrid.shape[1], tr_xygrid.shape[1])))
#     vars_data = np.empty(
#         ((len(wtk_vars), tr_xygrid.shape[1], tr_xygrid.shape[1])))
#     xmesh, ymesh = np.meshgrid(tr_xygrid[0, :], tr_xygrid[1, :])
#     print('interpolating-', end="", flush=True)
#     for i in range(sample_n):
#         print('\n' + str(i) + '/' + str(sample_n), flush=True)
#         wtk_easterly = np.multiply(
#             wtk_data[i, 0, :], np.sin(wtk_data[i, 1, :]))
#         wtk_northerly = np.multiply(
#             wtk_data[i, 0, :], np.cos(wtk_data[i, 1, :]))
#         for j in range(len(wtk_vars)):
#             vars_data[j, :, :] = interpolate_rbf(wtk_xygrid, wtk_easterly,
#                                                  xmesh, ymesh, interp_type)
#             # interp_function = Rbf(wtk_xygrid[0, :], wtk_xygrid[1, :],
#             #                       wtk_data[j, :, i], function=interp_type)
#             # vars_data[j, :, :] = interp_function(xmesh, ymesh)
#         updraft_data[i, :, :] = compute_oro_updraft_mat(vars_data[0, :, :],
#                                                         vars_data[1, :, :],
#                                                         tr_slope,
#                                                         tr_aspect)

#     print('summarizing-', end="", flush=True)
#     print('Computing ratio ..', flush=True)
#     updraft_percentage = (updraft_data > 0.75).sum(axis=0) / float(sample_n)
#     print('Computing median ..', flush=True)
#     updraft_median = np.median(updraft_data, axis=0)
#     print('Computing mean ..', flush=True)
#     updraft_mean = np.mean(updraft_data, axis=0)
#     print('Computing percentile ..', flush=True)
#     updraft_nintyptl = np.percentile(updraft_data, 90, axis=0)
#     print('Computing std ..', flush=True)
#     updraft_std = np.std(updraft_data, axis=0)
#     run_id = 'H' + str(wtk_height) + '_y' + str(wtk_year) + 'm' + \
#         "m".join([str(xi) for xi in wtk_months])
#     np.save(data_dir + run_id + '_oro_ratio', updraft_percentage)
#     np.save(data_dir + run_id + '_oro_mean', updraft_mean)
#     np.save(data_dir + run_id + '_oro_median', updraft_median)
#     np.save(data_dir + run_id + '_oro_nintyptl', updraft_nintyptl)
#     np.save(data_dir + run_id + '_oro_std', updraft_std)
#     print('done\nWrote WTK data in {:s}'.format(data_dir))
#     return run_id, updraft_percentage, updraft_mean,
# updraft_median, updraft_std, updraft_nintyptl


# # %% Computed percentage time orograhic updraft greater than threshhold
# def extract_oro_updraft_summary_EAGLE(
#         tr_xygrid,
#         tr_slope,
#         tr_aspect,
#         wtk_source,
#         wtk_indices,
#         wtk_xygrid,
#         wtk_height,
#         wtk_year,
#         wtk_months,
#         wtk_hours,
#         interp_type,
#         data_dir: str,
# ):
#     """Extracts orograhic speed on a given WTK index """

#     print('\nExtracting orographic updraft summary for')
#     print('year ' + str(wtk_year) + ', months ' +
#           ",".join([str(xi) for xi in wtk_months]), ', hours ' +
#           ",".join([str(xi) for xi in wtk_hours]))

#     if wtk_source == 'AWS':
#         import h5pyd as hsds
#         fname = '/nrel/wtk/conus/wtk_conus_' + str(wtk_year) + '.h5'
#     elif wtk_source == 'EAGLE':
#         import h5py as hsds
#         fname = '/datasets/WIND/conus/v1.0.0/wtk_conus_' + \
#             str(wtk_year) + '.h5'

#     wtk_n = wtk_indices.shape[0]
#     wtk_vars = ['windspeed', 'winddirection']
#     wtk_columns = [x + '_' + str(int(wtk_height)) + 'm' for x in wtk_vars]
#     with hsds.File(fname, mode='r') as f:
#         time_index = pd.to_datetime(f['time_index'][...].astype(str))
#         df = pd.DataFrame(index=time_index, columns=wtk_columns)
#         print('Reading WTK:', end="")
#         for i in range(wtk_n):
#             if i % 100 == 0 or i == wtk_n - 1:
#                 print('{0:d}-'.format(i), end='', flush=True)
#             for k, wtk_column in enumerate(wtk_columns):
#                 ds = f[wtk_column]
#                 scale_factor = ds.attrs['scale_factor']
#                 df.loc[:, wtk_column] = ds[:, wtk_indices[i]] / scale_factor
#                 if i == 0 and k == 0:
#                     year_bool = df.index.year == wtk_year
#                     month_bool = np.logical_and(
#                         True, df.index.month == wtk_months[0])
#                     for month in wtk_months:
#                         month_bool = np.logical_or(df.index.month == month,
#                                                    month_bool)
#                 hour_bool = np.logical_and(df.index.hour >= min(wtk_hours),
#                                            df.index.hour <= max(wtk_hours))
#                     wtk_bool = np.logical_and.reduce((year_bool,
#                                                       month_bool,
#                                                       hour_bool))
#                     sample_n = np.sum(wtk_bool)
#                     wtk_data = np.empty((sample_n, len(wtk_columns), wtk_n))
#                     # wtk_wspeed = np.empty((sample_n, wtk_n))
#                     # wtk_wdirn = np.empty((sample_n, wtk_n))
#                 wtk_data[:, k, i] = df.loc[wtk_bool, wtk_column]

#     updraft_data = np.empty(
#         ((sample_n, tr_xygrid.shape[1], tr_xygrid.shape[1])))
#     vars_data = np.empty(
#         ((len(wtk_vars), tr_xygrid.shape[1], tr_xygrid.shape[1])))
#     xmesh, ymesh = np.meshgrid(tr_xygrid[0, :], tr_xygrid[1, :])
#     print('interpolating-', end="", flush=True)
#     for i in range(sample_n):
#         print('\n' + str(i) + '/' + str(sample_n), flush=True)
#         wtk_easterly = np.multiply(
#             wtk_data[i, 0, :], np.sin(wtk_data[i, 1, :]))
#         wtk_northerly = np.multiply(
#             wtk_data[i, 0, :], np.cos(wtk_data[i, 1, :]))
#         for j in range(len(wtk_vars)):
#             vars_data[j, :, :] = interpolate_rbf(wtk_xygrid, wtk_easterly,
#                                                  xmesh, ymesh, interp_type)
#             # interp_function = Rbf(wtk_xygrid[0, :], wtk_xygrid[1, :],
#             #                       wtk_data[j, :, i], function=interp_type)
#             # vars_data[j, :, :] = interp_function(xmesh, ymesh)
#         updraft_data[i, :, :] = compute_oro_updraft_mat(vars_data[0, :, :],
#                                                         vars_data[1, :, :],
#                                                         tr_slope,
#                                                         tr_aspect)

#     print('summarizing-', end="", flush=True)
#     print('Computing ratio ..', flush=True)
#     updraft_percentage = (updraft_data > 0.75).sum(axis=0) / float(sample_n)
#     print('Computing median ..', flush=True)
#     updraft_median = np.median(updraft_data, axis=0)
#     print('Computing mean ..', flush=True)
#     updraft_mean = np.mean(updraft_data, axis=0)
#     print('Computing percentile ..', flush=True)
#     updraft_nintyptl = np.percentile(updraft_data, 90, axis=0)
#     print('Computing std ..', flush=True)
#     updraft_std = np.std(updraft_data, axis=0)
#     run_id = 'H' + str(wtk_height) + '_y' + str(wtk_year) + 'm' + \
#         "m".join([str(xi) for xi in wtk_months])
#     np.save(data_dir + run_id + '_oro_ratio', updraft_percentage)
#     np.save(data_dir + run_id + '_oro_mean', updraft_mean)
#     np.save(data_dir + run_id + '_oro_median', updraft_median)
#     np.save(data_dir + run_id + '_oro_nintyptl', updraft_nintyptl)
#     np.save(data_dir + run_id + '_oro_std', updraft_std)
#     print('done\nWrote WTK data in {:s}'.format(data_dir))
#     return run_id, updraft_percentage, updraft_mean, updraft_median,
# updraft_std, updraft_nintyptl


# # %% To DO
# def extract_wtkdata_eagle(
#         wtk_ind,
#         wtk_xy,
#         xgrid,
#         ygrid,
#         wtk_vars,
#         wtk_height,
#         wtk_years,
#         wtk_seasons,
#         wtk_daytime,
#         wtk_dir: str,
# ):
#     """Extracts WTK data on a given grid (xgrid,ygrid) """

#     print('Extracting ', *wtk_vars, ' at ', wtk_height, 'meters ..')
#     wtk_n = wtk_ind.shape[0]
#     wtk_data = np.empty((len(wtk_vars), len(wtk_seasons), wtk_n))
#     for i in range(wtk_n):
#         if i % 20 == 0:
#             print('{0:d}-'.format(i), end='', flush=True)
#         dfwtk = pd.DataFrame({'': []})
#         for year in wtk_years:
#             file_name = '/datasets/WIND/conus/v1.0.0/wtk_conus_' + \
#                         str(year) + '.h5'
#             # print('Year {0:d}:'.format(year))
#             with h5py.File(file_name, mode='r') as f:
#                 time_index = pd.to_datetime(f['time_index'][...].astype(str))
#                 time_series = pd.DataFrame(index=time_index)
#                 for r, entity in enumerate(wtk_vars):
#                     # print('Reading {0:s} at {1:d}m'.format(entity, wtk_height))
#                     field_name = entity + '_' + str(int(wtk_height[r])) + 'm'
#                     ds = f[field_name]
#                     scale_factor = ds.attrs['scale_factor']
#                     time_series[entity] = ds[:, wtk_ind[i]] / scale_factor
#                 if dfwtk.empty:
#                     dfwtk = time_series
#                 else:
#                     dfwtk = dfwtk.append(time_series)
#         for k, iseason in enumerate(wtk_seasons):
#             season_bool = dfwtk.index.month == iseason[0]
#             for imonth in iseason:
#                 season_bool = np.logical_or(
#                     season_bool, dfwtk.index.month == imonth)
#             daytime_bool = np.logical_and(dfwtk.index.hour >= wtk_daytime[k][0],
#                                           dfwtk.index.hour <= wtk_daytime[k][1])
#             wtk_bool = np.logical_and(season_bool, daytime_bool)
#             for r, entity in enumerate(wtk_vars):
#                 wtk_data[r, k, i] = dfwtk.loc[wtk_bool, entity].median()
#     print('done')

#     # interpolate wind conditions onto finner resolution grid
#     print('Computing interpolated data for finer grid ..', end=" ")
#     xmin = min(min(xgrid), min(wtk_xy[0, :]))
#     xmax = max(max(xgrid), max(wtk_xy[0, :]))
#     ymin = min(min(ygrid), min(wtk_xy[1, :]))
#     ymax = max(max(ygrid), max(wtk_xy[1, :]))
#     xmesh, ymesh = np.meshgrid(xgrid, ygrid)
#     vars_data = np.empty(
#         (len(wtk_vars), len(wtk_seasons), xgrid.size, ygrid.size))
#     for r in range(len(wtk_vars)):
#         for k in range(len(wtk_seasons)):
#             fun_spline = SmoothBivariateSpline(wtk_xy[0, :], wtk_xy[1, :],
#                                                wtk_data[r, k, :],
#                                                bbox=[xmin, xmax, ymin, ymax],
#                                                s=wtk_n)
#             vars_data[r, k, :, :] = np.transpose(
#                 fun_spline(xgrid, ygrid, grid=True))
#             f_id = get_identifier(int(wtk_height[r]), k + 1)
#             np.save(wtk_dir + f_id + wtk_vars[r], vars_data[r, k, :, :])
#     print('done\nWrote WTK data in {:s}'.format(wtk_dir))
#     return vars_data


# # %% Extract WIND toolkit data from EAGLE
# def extract_wtkdata_eagle(
#         wtk_ind,
#         wtk_xy,
#         xgrid,
#         ygrid,
#         wtk_vars,
#         wtk_height,
#         wtk_years,
#         wtk_seasons,
#         wtk_daytime,
#         wtk_dir: str,
# ):
#     """Extracts WTK data on a given grid (xgrid,ygrid) """

#     print('Extracting ', *wtk_vars, ' at ', wtk_height, 'meters ..')
#     wtk_n = wtk_ind.shape[0]
#     wtk_data = np.empty((len(wtk_vars), len(wtk_seasons), wtk_n))
#     for i in range(wtk_n):
#         if i % 20 == 0:
#             print('{0:d}-'.format(i), end='', flush=True)
#         dfwtk = pd.DataFrame({'': []})
#         for year in wtk_years:
#             file_name = '/datasets/WIND/conus/v1.0.0/wtk_conus_' + \
#                         str(year) + '.h5'
#             # print('Year {0:d}:'.format(year))
#             with h5py.File(file_name, mode='r') as f:
#                 time_index = pd.to_datetime(f['time_index'][...].astype(str))
#                 time_series = pd.DataFrame(index=time_index)
#                 for r, entity in enumerate(wtk_vars):
#                     # print('Reading {0:s} at {1:d}m'.format(entity, wtk_height))
#                     field_name = entity + '_' + str(int(wtk_height[r])) + 'm'
#                     ds = f[field_name]
#                     scale_factor = ds.attrs['scale_factor']
#                     time_series[entity] = ds[:, wtk_ind[i]] / scale_factor
#                 if dfwtk.empty:
#                     dfwtk = time_series
#                 else:
#                     dfwtk = dfwtk.append(time_series)
#         for k, iseason in enumerate(wtk_seasons):
#             season_bool = dfwtk.index.month == iseason[0]
#             for imonth in iseason:
#                 season_bool = np.logical_or(
#                     season_bool, dfwtk.index.month == imonth)
#             daytime_bool = np.logical_and(dfwtk.index.hour >= wtk_daytime[k][0],
#                                           dfwtk.index.hour <= wtk_daytime[k][1])
#             wtk_bool = np.logical_and(season_bool, daytime_bool)
#             for r, entity in enumerate(wtk_vars):
#                 wtk_data[r, k, i] = dfwtk.loc[wtk_bool, entity].median()
#     print('done')

#     # interpolate wind conditions onto finner resolution grid
#     print('Computing interpolated data for finer grid ..', end=" ")
#     xmin = min(min(xgrid), min(wtk_xy[0, :]))
#     xmax = max(max(xgrid), max(wtk_xy[0, :]))
#     ymin = min(min(ygrid), min(wtk_xy[1, :]))
#     ymax = max(max(ygrid), max(wtk_xy[1, :]))
#     xmesh, ymesh = np.meshgrid(xgrid, ygrid)
#     vars_data = np.empty(
#         (len(wtk_vars), len(wtk_seasons), xgrid.size, ygrid.size))
#     for r in range(len(wtk_vars)):
#         for k in range(len(wtk_seasons)):
#             fun_spline = SmoothBivariateSpline(wtk_xy[0, :], wtk_xy[1, :],
#                                                wtk_data[r, k, :],
#                                                bbox=[xmin, xmax, ymin, ymax],
#                                                s=wtk_n)
#             vars_data[r, k, :, :] = np.transpose(
#                 fun_spline(xgrid, ygrid, grid=True))
#             f_id = get_identifier(int(wtk_height[r]), k + 1)
#             np.save(wtk_dir + f_id + wtk_vars[r], vars_data[r, k, :, :])
#     print('done\nWrote WTK data in {:s}'.format(wtk_dir))
#     return vars_data


# # %% Extracts orographic updraft randomly using data from aws
# def extract_oro_aws(
#         wtk_ind,
#         wtk_height: int,
#         wtk_year: int,
#         wtk_month: int,
#         wtk_time: int,
#         wtk_dir: str,
# ):
#     """Extracts WTK data on a given grid (xgrid,ygrid) """

#     # print('Extracting ', *wtk_vars, ' at ', wtk_height, 'meters ..')
#     entities = ('windspeed', 'windidirection')
#     # wtk_n = wtk_ind.shape[0]
#     # wtk_data = np.empty((len(wtk_vars), len(wtk_seasons), wtk_n))
#     for i in range(1):
#         if i % 20 == 0:
#             print('{0:d}-'.format(i), end='', flush=True)
#         with h5pyd.File('/nrel/wtk-us.h5', mode='r') as f:
#             dt = pd.DataFrame({'datetime': f['datetime'][:]})
#             dt['datetime'] = dt['datetime'].apply(dateutil.parser.parse)
#             time_index = pd.Index(dt['datetime'])
#             time_series = pd.DataFrame(index=time_index)
#             for r, entity in enumerate(entities):
#                 print('Reading {0:s} at {1:d}m'.format(entity, wtk_height))
#                 field_name = entity + '_' + str(wtk_height) + 'm'
#                 ds = f[field_name]
#                 scale_factor = ds.attrs['scale_factor']
#                 time_series[entity] = ds[:, wtk_ind] / scale_factor
#             year_bool = time_series.index.year == wtk_year
#             month_bool = time_series.index.month == wtk_month
#             time_bool = time_series.index.month == wtk_time
#             wtk_bool = np.logical_and(month_bool, time_bool)
#             wtk_allbool = np.logical_and(year_bool, wtk_bool)
#             ts_new = time_series.loc[wtk_allbool, :]
#     print(ts_new.head())
#     return ts_new


# # %% List all wfarms in a state
# def get_all_wfarms(
#         wf_state='WY',
#         wf_csv='./data/wfarm/uswtdb_v3_0_1_20200514.csv',
#         print_verbose=False
# ):
#     """List all wind farms in a particular state using the USGS provided csv"""

#     # Read raw turbine data and print out the columns
#     df = pd.read_csv(wf_csv)
#     if print_verbose:
#         print('Got these columns in the dataframe: \n', df.columns)

#     # List all windfarms in the given state, grouped by county
#     wf_counties = df.loc[df.t_state == wf_state, 't_county'].unique()
#     wf_names = []
#     if print_verbose:
#         print('Got these wfarms in {:s} (grouped by county):'.format(wf_state))
#     for icounty in wf_counties:
#         inames = list(df.loc[df.t_county == icounty, 'p_name'].unique())
#         wf_names = np.concatenate((wf_names, inames))
#         if print_verbose:
#             print(icounty, ': \n', inames)
#     return wf_names


# # %%
# def extract_wtkgrid_aws(
#         latlon,
#         wtk_fac: int,
#         wtk_dir: str
# ):
#     """Extracts WTK grid locations using lat/lon bounds"""

#     print('\n----- Extracting WTK grid locations')
#     with h5pyd.File('/nrel/wtk-us.h5', mode='r') as f:
#         print('Got these fields: \n', list(f))
#         print(f['coordinates'][0][:])
#         print(f['coordinates'][1].shape)
#         df = pd.DataFrame([f['coordinates'][0][:], f['coordinates'][1][:]],
#                           columns=['latitude', 'longitude'])
#         # df['latitude'] = f['coordinates'][0]
#         # df['longitude'] = f['coordinates'][1]
#         print(df.head())
#     lat_bool = np.logical_and(df.latitude > latlon[0][0],
#                               df.latitude < latlon[0][1])
#     lon_bool = np.logical_and(df.longitude > latlon[1][0],
#                               df.longitude < latlon[1][1])
#     all_bool = np.logical_and(lat_bool, lon_bool)
#     wtk_ind = df.loc[all_bool, :].index.to_numpy()
#     wtk_ind = wtk_ind[::wtk_fac]
#     wtk_n = wtk_ind.shape[0]
#     print('Got {0:5d} WTK grid points for the lat/lon range:'.format(wtk_n))
#     print('Lat range : {0:0.3f}, {1:0.3f}'.format(*latlon[0]))
#     print('Lon range: {0:0.3f}, {1:0.3f}'.format(*latlon[1]))
#     # Extract coordinates of WTK points
#     wtk_xy = np.empty((2, wtk_n))
#     for i, indname in enumerate(wtk_ind):
#         lat, lon = df.loc[indname, :].to_numpy()
#         ix, iy, _, _ = utm.from_latlon(lat, lon)
#         wtk_xy[0, i] = ix / 1000.
#         wtk_xy[1, i] = iy / 1000.
#     np.save(wtk_dir + 'wtk_ind.npy', wtk_ind)
#     np.save(wtk_dir + 'wtk_xy.npy', wtk_xy)
#     return wtk_ind, wtk_xy


# # %% Extract NSRDB data from EAGLE
# def extract_nsrdb_grid(
#         latlon,
#         wtk_fac=1
# ):
#     """Extracts NSRDB grid locations using lat/lon bounds"""

#     print('\n----- Extracting NSRDB grid locations')
#     with h5pyd.File('/nrel/nsrdb/v3/nsrdb_2012.h5', mode='r') as f:
#         print('Got these fields: \n', list(f))
#         df = pd.DataFrame(f['coordinates'][:], columns=[
#             'latitude', 'longitude'])
#     lat_bool = np.logical_and(df.latitude > latlon[0][0],
#                               df.latitude < latlon[0][1])
#     lon_bool = np.logical_and(df.longitude > latlon[1][0],
#                               df.longitude < latlon[1][1])
#     all_bool = np.logical_and(lat_bool, lon_bool)
#     wtk_ind = df.loc[all_bool, :].index.to_numpy()
#     wtk_ind = wtk_ind[::wtk_fac]
#     wtk_n = wtk_ind.shape[0]
#     print('Got {0:d} WTK grid points for the lat/lon range:'.format(wtk_n))
#     print('Lat range : {0:0.3f}, {1:0.3f}'.format(*latlon[0]))
#     print('Lon range: {0:0.3f}, {1:0.3f}'.format(*latlon[1]))
#     # Extract coordinates of WTK points
#     wtk_xy = np.empty((2, wtk_n))
#     for i, indname in enumerate(wtk_ind):
#         lat, lon = df.loc[indname, :].to_numpy()
#         ix, iy, _, _ = utm.from_latlon(lat, lon)
#         wtk_xy[0, i] = ix / 1000.
#         wtk_xy[1, i] = iy / 1000.
#     return wtk_ind, wtk_xy


# # %% Extract WIND toolkit data
# def extract_nsrdb_data(
#         wtk_dir: str,
#         wtk_ind, wtk_xy, xgrid, ygrid,
#         wtk_vars=('air_temperature', 'ghi'),
#         wtk_height=(40, 40),
#         wtk_years=list(range(2015, 2016)),
#         wtk_seasons=((3, 4, 5),),
#         wtk_daytime=((7, 17),)
# ):
#     """Extracts WTK data on a given grid (xgrid,ygrid) """

#     print('Extracting ', *wtk_vars, ' at ', wtk_height, 'meters ..')
#     wtk_n = wtk_ind.shape[0]
#     wtk_data = np.empty((len(wtk_vars), len(wtk_seasons), wtk_n))
#     for i in range(wtk_n):
#         if i % 20 == 0:
#             print('{0:d}-'.format(i), end='', flush=True)
#         dfwtk = pd.DataFrame({'': []})
#         for year in wtk_years:
#             file_name = '/nrel/nsrdb/v3/nsrdb_' + \
#                 str(year) + '.h5'
#             # print('Year {0:d}:'.format(year))
#             with h5pyd.File(file_name, mode='r') as f:
#                 time_index = pd.to_datetime(f['time_index'][...].astype(str))
#                 time_series = pd.DataFrame(index=time_index)
#                 for r, entity in enumerate(wtk_vars):
#                     # print('Reading {0:s} at {1:d}m'.format(entity, wtk_height))
#                     field_name = entity
#                     ds = f[field_name]
#                     # scale_factor = ds.attrs['scale_factor']
#                     time_series[entity] = ds[:, wtk_ind[i]]
#                 if dfwtk.empty:
#                     dfwtk = time_series
#                 else:
#                     dfwtk = dfwtk.append(time_series)
#         for k, iseason in enumerate(wtk_seasons):
#             season_bool = dfwtk.index.month == iseason[0]
#             for imonth in iseason:
#                 season_bool = np.logical_or(
#                     season_bool, dfwtk.index.month == imonth)
#             daytime_bool = np.logical_and(dfwtk.index.hour >= wtk_daytime[k][0],
#                                           dfwtk.index.hour <= wtk_daytime[k][1])
#             wtk_bool = np.logical_and(season_bool, daytime_bool)
#             for r, entity in enumerate(wtk_vars):
#                 wtk_data[r, k, i] = dfwtk.loc[wtk_bool, entity].median()
#     print('done')

#     # interpolate wind conditions onto finner resolution grid
#     print('Computing interpolated data for finer grid ..', end=" ")
#     xmin = min(min(xgrid), min(wtk_xy[0, :]))
#     xmax = max(max(xgrid), max(wtk_xy[0, :]))
#     ymin = min(min(ygrid), min(wtk_xy[1, :]))
#     ymax = max(max(ygrid), max(wtk_xy[1, :]))
#     xmesh, ymesh = np.meshgrid(xgrid, ygrid)
#     vars_data = np.empty(
#         (len(wtk_vars), len(wtk_seasons), xgrid.size, ygrid.size))
#     for r in range(len(wtk_vars)):
#         for k in range(len(wtk_seasons)):
#             fun_spline = SmoothBivariateSpline(wtk_xy[0, :], wtk_xy[1, :],
#                                                wtk_data[r, k, :],
#                                                bbox=[xmin, xmax, ymin, ymax],
#                                                s=wtk_n)
#             vars_data[r, k, :, :] = np.transpose(
#                 fun_spline(xgrid, ygrid, grid=True))
#             f_id = get_identifier(int(wtk_height[r]), k + 1)
#             np.save(wtk_dir + f_id + wtk_vars[r], vars_data[r, k, :, :])
#     print('done\nWrote WTK data in {:s}'.format(wtk_dir))
#     return vars_data


#
# def extract_orodata_eagle_no_interpolation(
#         wtk_indices,
#         wtk_grid,
#         xygrid,
#         tr_slope,
#         tr_aspect,
#         wtk_height,
#         wtk_year,
#         wtk_month,
#         wtk_time,
#         wtk_dir: str,
# ):
#     """Extracts orograhic speed on a given WTK index """

#     # print('Extracting ', *wtk_vars, ' at ', wtk_height, 'meters ..')
#     wtk_n = wtk_indices.shape[0]
#     # wtk_nodeindex = 0
#     wtk_vars = ('windspeed', 'winddirection')
#     my_columns = [*wtk_vars, 'oro_updraft']
#     file_name = '/datasets/WIND/conus/v1.0.0/wtk_conus_' + \
#         str(wtk_year) + '.h5'
#     out_data = np.empty(wtk_n)
#     # print('WTK grid point {0:d}-'.format(i), end='', flush=True)
#     with h5py.File(file_name, mode='r') as f:
#         time_index = pd.to_datetime(f['time_index'][...].astype(str))
#         time_series = pd.DataFrame(index=time_index, columns=my_columns)
#         for i in range(wtk_n):
#             for r, entity in enumerate(wtk_vars):
#                 # print('Reading {0:s} at {1:d}m'.format(entity, wtk_height))
#                 field_name = entity + '_' + str(int(wtk_height)) + 'm'
#                 ds = f[field_name]
#                 scale_factor = ds.attrs['scale_factor']
#                 time_series.loc[:, entity] = ds[:,
#                                                 wtk_indices[i]] / scale_factor
#             if i == 0:
#                 year_bool = time_series.index.year == wtk_year
#                 month_bool = time_series.index.month == wtk_month
#                 hour_bool = time_series.index.hour == wtk_time
#                 wtk_bool = np.logical_and(month_bool, hour_bool)
#                 wtk_allbool = np.logical_and(year_bool, wtk_bool)
#             tshort = time_series.loc[wtk_allbool, :]
#             idx = closest_point(wtk_grid[0, i], xygrid[0, :])
#             idy = closest_point(wtk_grid[1, i], xygrid[1, :])
#             tshort.loc[:, 'oro_updraft'] =
# np.vectorize(compute_oro_updraft)(tshort['windspeed'],
#                                        tshort['winddirection'],
#                                        tr_slope[idy, idx],
#                                                 tr_aspect[idy, idx])
#             out_data[i] = (tshort.loc[:, 'oro_updraft'].to_numpy()
#                            > 0.75).sum() * 100 / len(tshort.index)
#     # print(idx, idy, wtk_grid[:, 0], xygrid[0, idx], xygrid[1,idy])
#     print('done.')
#     return out_data


# function for extractin wind farm data from USGS provided csv file
# def extract_all_wfarm(
#     wf_state = 'WY',
#     wf_county= 'Converse County',
#     wf_csv = './data/wfarm/uswtdb_v3_0_1_20200514.csv',
#     print_verbose = False
#     ):

#   """Extracts all turbine data in a county using USGS csv"""

#   # Read raw turbine data and print out the columns
#   print('\n----- Extracting turbine data')
#   print('CSV file used: {:s}'.format(wf_csv))
#   wf_dir = os.path.dirname(wf_csv) + '/'
#   df = pd.read_csv(wf_csv)
#   if print_verbose:
#     print('Got these columns in the dataframe: \n', df.columns)

#   # List all windfarms in the given state, grouped by county
#   wf_counties = df.loc[df.t_state==wf_state,'t_county'].unique()
#   if print_verbose:
#     print('Got these wfarms in {:s} (grouped by county):'.format(wf_state))
#     for icounty in wf_counties:
#       print(icounty, ': ', df.loc[df.t_county==icounty,'p_name'].unique())

#   # Check if county name provided is present
#   if wf_county not in wf_counties:
#     raise NameError('Incorrect county name!')

#   # Extract lat.long data for turbines
#   print('Getting turbine data in {0:s} ({1:s}) ..'.format(wf_county,wf_state))
#   county_bool = np.logical_and(df.t_county==wf_county, df.t_state==wf_state)
#   wfarms = df.loc[county_bool,'p_name'].unique()
#   for wf in wfarms:
#     wf_bool = np.logical_and(df.t_state==wf_state, df.p_name==wf)
#     wf_cap = df.loc[wf_bool,'p_cap'].unique()
#     wf_size = df.loc[wf_bool,'p_tnum'].unique()
#     xy_latlon = df.loc[wf_bool,['ylat','xlong']].to_numpy()
#     if xy_latlon.shape[0] > 10 and max(wf_cap) > 10.:
#       print('{3:s} : {0:d}/{1:d}, {2:.1f} MW '.format(xy_latlon.shape[0],
#                                                wf_size[0],wf_cap[0],wf))
#       fname = wf_dir + wf.replace(" ", "_").lower()
#       x_meters, y_meters,_,_ = utm.from_latlon(xy_latlon[:,0],xy_latlon[:,1])
#       xy_kms = np.stack((x_meters/1000., y_meters/1000.), axis=-1)
#       np.savetxt(fname+'_latlon.txt', xy_latlon,fmt='%0.12g')
#       np.savetxt(fname+'_kms.txt', xy_kms,fmt='%0.12g')
#   print('Writing turbine data in {:s}'.format(wf_dir))
#   return xy_latlon, xy_kms

# tshort.loc[:,'oro_updraft'] =
# np.vectorize(compute_oro_updraft)(tshort['windspeed'],
# tshort['winddirection'],
# tr_slope[idy, idx],
# tr_aspect[idy, idx])
#        out_data[i] = (tshort.loc[:,'oro_updraft']
# .to_numpy() > 0.75).sum()*100/len(tshort.index)
# print(idx, idy, wtk_grid[:, 0], xygrid[0, idx], xygrid[1,idy])
# print('done.')


# %%
# %% Extracts orographic updraft randomly using WTK data
# def extract_orographic _updraft_wtk_random(
#         tr_xygrid,
#         tr_slope,
#         tr_aspect,
#         wtk_source,
#         wtk_indices,
#         wtk_xygrid,
#         wtk_height,
#         wtk_years,
#         wtk_months,
#         wtk_hours,
#         interp_type,
#         sample_id=0
# ):
#     """Extracts orographic updraft speed for a randomly selected time
#     using WTK"""

#     # Randomly select year/month/day/time
#     random_year = random.choice(wtk_years)
#     random_month = random.choice(wtk_months)
#     random_day = random.choice(range(1, 28))
#     random_hour = random.choice(wtk_hours)

#     # run ID used to save the data
#     run_id = 'H' + str(wtk_height) + '_y' + str(random_year) + \
#         'm' + str(random_month) + 'd' + str(random_day) + \
#         'h' + str(random_hour)
#     print('Computing orographic updraft at', run_id)

#     if wtk_source == 'AWS':
#         import h5pyd as hsds
#         fname = '/nrel/wtk/conus/wtk_conus_' + str(random_year) + '.h5'
#     elif wtk_source == 'EAGLE':
#         import h5py as hsds
#         fname = '/datasets/WIND/conus/v1.0.0/wtk_conus_' + \
#             str(random_year) + '.h5'
#     # print('Data source: ', wtk_source, ' - ', fname)

#     # Variables needed to extract for computing oro updraft
#     wtk_vars = ['windspeed', 'winddirection']
#     wtk_columns = [x + '_' + str(int(wtk_height)) + 'm' for x in wtk_vars]

#     # Read the WTK file and extract the data
#     wtk_n = wtk_indices.shape[0]
#     with hsds.File(fname, mode='r') as f:
#         time_index = pd.to_datetime(f['time_index'][...].astype(str))
#         df = pd.DataFrame(index=time_index, columns=wtk_columns)
#         wtk_bool = np.logical_and.reduce((df.index.month == random_month,
#                                           df.index.day == random_day,
#                                           df.index.hour == random_hour))
#         wtk_data = np.empty((len(wtk_vars), wtk_n))
#         # print('Reading WTK: ', end="")
#         for i in range(wtk_n):
#             # if i % 100 == 0 or i == wtk_n - 1:
#                 # print('{0:d}-'.format(i), end='', flush=True)
#             for k, wtk_column in enumerate(wtk_columns):
#                 ds = f[wtk_column]
#                 scale_factor = ds.attrs['scale_factor']
#                 df.loc[:, wtk_column] = f[wtk_column][:,
#                                         wtk_indices[i]] / scale_factor
#                 wtk_data[k, i] = df.loc[wtk_bool, wtk_column]
#     vars_data = np.empty(
#         ((len(wtk_vars), tr_xygrid.shape[1], tr_xygrid.shape[1])))
#     vars_data_mod = np.empty(
#         ((len(wtk_vars), tr_xygrid.shape[1], tr_xygrid.shape[1])))
#     wtk_data_mod = np.empty((len(wtk_vars), wtk_n))
#     # print('interpolating-', end="", flush=True)
#     xmesh, ymesh = np.meshgrid(tr_xygrid[0, :], tr_xygrid[1, :])
#     wtk_data_mod[0, :] = np.multiply(
#         wtk_data[0, :], np.sin(wtk_data[1, :] * np.pi / 180.))
#     wtk_data_mod[1, :] = np.multiply(
#         wtk_data[0, :], np.cos(wtk_data[1, :] * np.pi / 180.))
#     vars_data_mod[0, :, :] = interpolate_rbf(wtk_xygrid, wtk_data_mod[0, :],
#                                              xmesh, ymesh, interp_type)
#     vars_data_mod[1, :, :] = interpolate_rbf(wtk_xygrid, wtk_data_mod[1, :],
#                                              xmesh, ymesh, interp_type)
#     vars_data[0, :, :] = np.sqrt(
#         np.square(vars_data_mod[0, :, :]) + np.square(vars_data_mod[1, :, :]))
#     vars_data[1, :, :] = np.mod(np.arctan2(vars_data_mod[0, :, :],
#                                            vars_data_mod[1, :, :])
#                                 + 2. * np.pi,
#                                 2. * np.pi)
#     # for j in range(len(wtk_vars)):
#     #     interp_function = Rbf(wtk_xygrid[0, :], wtk_xygrid[1, :], wtk_data[j, :],
#     #                           function=interp_type)
#     #     vars_data[j, :, :] = interp_function(xmesh, ymesh)
#     updraft_data = compute_oro_updraft_mat(vars_data[0, :, :],
#                                            vars_data[1, :, :],
#                                            tr_slope,
#                                            tr_aspect)
#     # np.save(data_dir + run_id + '_oro_updraft', updraft_data)
#     # print('done\nWrote data in {:s}'.format(data_dir))
#     return run_id, updraft_data, vars_data_mod, wtk_data_mod
# def get_filenames_with_updrafts(my_dir, hgt, res, months, hours):
#     hgt_flag = 'hg' + str(hgt) + '.'
#     res_flag = '.r' + str(int(res * 1000.)) + '.'
#     m_flags = ['.m' + str(xi) + '.' for xi in months]
#     h_flags = ['.h' + str(xi) + '_' for xi in hours]
#     fnames = []
#     for fname in os.listdir(my_dir):
#         if (hgt_flag in fname and res_flag in fname):
#             for m_flag in m_flags:
#                 for h_flag in h_flags:
#                     if (m_flag in fname and h_flag in fname):
#                         fnames.append(fname)
#     return fnames
# # %% Sort indices
# def group_indices(ind: List[int]):
#     ind_grouped = []
#     starts_at = ind[0]
#     for i, j in zip(ind[:-1], ind[1:]):
#         if i != j - 1:
#             ind_grouped.append([int(starts_at), int(i)])
#             starts_at = j
#     return ind_grouped

# # %% Extract WIND toolkit data from EAGLE
# def extract_wtk_positions_old(
#         latlon,
#         wtk_source: str,
#         wtk_fac: int
# ):
#     """Extracts WTK grid locations using lat/lon bounds"""

#     pname, fname = get_wtk_sourceinfo(wtk_source, 2010)
#     hsds = importlib.import_module(pname)

#     with hsds.File(fname, mode='r') as f:
#         print('Got these fields: \n', list(f))
#         df = pd.DataFrame(f['coordinates'][:], columns=[
#             'latitude', 'longitude'])
#     lat_bool = np.logical_and(df.latitude > latlon[0, 0],
#                               df.latitude < latlon[0, 1])
#     lon_bool = np.logical_and(df.longitude > latlon[1, 0],
#                               df.longitude < latlon[1, 1])
#     all_bool = np.logical_and(lat_bool, lon_bool)
#     wtk_ind = df.loc[all_bool, :].index.to_numpy()
#     #wtk_ind = wtk_ind[::wtk_fac]
#     wtk_n = wtk_ind.shape[0]
#     print('Got {0:d} WTK grid points for the lat/lon range:'.format(wtk_n))
#     print('Lat range : {0:0.3f}, {1:0.3f}'.format(latlon[0, 0], latlon[0, 1]))
#     print('Lon range: {0:0.3f}, {1:0.3f}'.format(latlon[1, 0], latlon[1, 1]))
#     # Extract coordinates of WTK points
#     wtk_xy = np.empty((2, wtk_n))
#     for i, indname in enumerate(wtk_ind):
#         lat, lon = df.loc[indname, :].to_numpy()
#         ix, iy, _, _ = utm.from_latlon(lat, lon)
#         wtk_xy[0, i] = ix / 1000.
#         wtk_xy[1, i] = iy / 1000.
#     return wtk_ind, wtk_xy
# if wtk_source == 'AWS_2':
# #     wtk_indices_grouped = group_indices(wtk_indices)
# #     curi = 0
# #     for i in wtk_indices_grouped:
# #         ilen = i[1] - i[0]
# #         wtk_data[k, curi:curi + ilen] = f[wtk_column][time_index,
# #                                              i[0]:i[1]] / inorm
# #         curi += ilen + 1


# def get_run_ids_of_saved_updrafts(
#     my_dir: str,
#     hgt: int,
#     res: float,
#     months: List[int],
#     hours: List[int]
# ) -> List[str]:
#     """ get run IDs from all the saved files that match the user-supplied
#     WTK and model settings """

#     hgt_flag = 'hg' + str(hgt) + '.'
#     res_flag = '.r' + str(int(res * 1000.)) + '.'
#     m_flags = ['.m' + str(xi) + '.' for xi in months]
#     h_flags = ['.h' + str(xi) + '_' for xi in hours]
#     run_ids = set()
#     for fname in os.listdir(my_dir):
#         if (hgt_flag in fname and res_flag in fname):
#             for m_flag in m_flags:
#                 for h_flag in h_flags:
#                     if (m_flag in fname and h_flag in fname):
#                         id_string = fname.split("_")
#                         run_ids.add(id_string[0])
#     return list(run_ids)

# def extract_wfarm_data(
#     wf_state: str,
#     wf_names: List[str],
#     wf_csv: str
# ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
#     """Extracts wind farm data (in lat/lon and Kms) using USGS provided csv"""

#     # Read raw turbine data and print out the columns
#     print('\n--- Extracting wind turbine locations')
#     print('CSV file used: {:s}'.format(wf_csv))
#     df = pd.read_csv(wf_csv)

#     # check if wf_names correct for the given state
#     wf_all_names = get_all_wfarms(wf_state, wf_csv)
#     for wf in wf_names:
#         if wf not in wf_all_names:
#             raise NameError('[' + wf + '] not present in ' + wf_state + '!')

#     # Extract lat long data for turbines
#     print('State selected: {:s}'.format(wf_state))
#     wf_latlon, wf_kms = [], []
#     for wf in wf_names:
#         wf_bool = np.logical_and(df.t_state == wf_state, df.p_name == wf)
#         wf_cap = df.loc[wf_bool, 'p_cap'].unique()
#         wf_hgt = df.loc[wf_bool, 't_ttlh'].unique().max()
#         xy_latlon = df.loc[wf_bool, ['ylat', 'xlong']].to_numpy()
#         print('{2:s} : {0:d} turbines, {1:.1f} MW, {3:.1f} m high'.format(
#             xy_latlon.shape[0], wf_cap[0], wf, wf_hgt))
#         x_meters, y_meters, _, _ = utm.from_latlon(
#             xy_latlon[:, 0], xy_latlon[:, 1])
#         xy_kms = np.stack((x_meters / 1000., y_meters / 1000.), axis=-1)
#         wf_latlon.append(xy_latlon)
#         wf_kms.append(xy_kms)
#     southwest_kms = np.amin(np.asarray(
#         [x.min(axis=0) for x in wf_kms]), axis=0)
#     northeast_kms = np.amax(np.asarray(
#         [x.max(axis=0) for x in wf_kms]), axis=0)
#     print('Lat range: {:.2f} Km'.format(northeast_kms[0] - southwest_kms[0]))
#     print('Lon range: {:.2f} Km'.format(northeast_kms[1] - southwest_kms[1]))
#     return wf_latlon, wf_kms
