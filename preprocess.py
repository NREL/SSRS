import pathos.multiprocessing as mp
import numpy as np
import pandas as pd

from config import *
from tools.common import *
from tools.wfarm import *
from tools.terrain import *
from tools.wtk import *
from tools.updrafts import *


# Intialize
config = setup_config()

# figure out if terrain needs to be updated, or saved data can be reused
print('\n--- Extracting terrain (SRTM) and turbine data (USWTB)')
use_saved_terrain = False
extent_km, extent_lonlat = calculate_terrain_extent(
    config.terrain_southwest_lonlat, config.terrain_width)
if file_exists(config.terrain_data_dir, 'extent_km.txt'):
    extent_km_saved = load_data(config.terrain_data_dir, 'extent_km.txt')
    terrain_alt = load_data(config.terrain_data_dir, 'terrain_altitude.npy')
    M, N = terrain_alt.shape
    xmin, xmax, ymin, ymax = extent_km_saved
    xsize = int((xmax - xmin) * 1000. // config.terrain_res)
    ysize = int((ymax - ymin) * 1000. // config.terrain_res)
    if (extent_km_saved == extent_km).all() and M == ysize and N == xsize:
        use_saved_terrain = True
        print('Found saved terrain data that matches the {0:s} config setting'
              .format(config.run_name))
    else:
        remove_dirs_in(config.data_dir)

# load or compute terrain info
if use_saved_terrain:
    terrain_slope = load_data(config.terrain_data_dir, 'terrain_slope.npy')
    terrain_aspect = load_data(config.terrain_data_dir, 'terrain_aspect.npy')
else:
    extent_km, extent_lonlat = calculate_terrain_extent(
        config.terrain_southwest_lonlat, config.terrain_width)
    terrain_alt = extract_terrain_altitude_srtm(
        extent_lonlat, extent_km, config.terrain_res, config.terrain_data_dir)
    terrain_slope, terrain_aspect = compute_slope_and_aspect(
        terrain_alt, config.terrain_res)
    makedir_if_not_exists(config.terrain_data_dir)
    save_data(config.terrain_data_dir, 'extent_km.txt', extent_km)
    save_data(config.terrain_data_dir, 'extent_lonlat.txt', extent_lonlat)
    save_data(config.terrain_data_dir, 'terrain_altitude.npy', terrain_alt)
    save_data(config.terrain_data_dir, 'terrain_slope.npy', terrain_slope)
    save_data(config.terrain_data_dir, 'terrain_aspect.npy', terrain_aspect)
print('Lat bounds: {0:.2f}, {1:.2f}'.format(*extent_lonlat[2:4]))
print('Lon bounds: {0:.2f}, {1:.2f}'.format(*extent_lonlat[0:2]))
print('Terrain grid resolution: {0:.1f} meters'.format(config.terrain_res))
print('Terrain grid shape:', np.shape(terrain_alt))


# load wind turbine data
print('Importing turbine data from USWTB')
wfarm_df = extract_wfarm_data_using_lonlat(
    extent_lonlat, config.wfarm_minimum_hubheight)
save_data(config.terrain_data_dir, 'wfarm_data.csv', wfarm_df)
print_windfarm_details(wfarm_df)


# WTK source information
oro_varnames = ('windspeed_' + str(config.wtk_orographic_height) + 'm',
                'winddirection_' + str(config.wtk_orographic_height) + 'm')
thermal_varnames = ('pressure_' + str(config.wtk_thermal_height) + 'm',
                    'temperature_' + str(config.wtk_thermal_height) + 'm',
                    'boundary_layer_height', 'surface_heat_flux')
varnames = oro_varnames
compute_thermals = False
if config.wtk_data_source == 'WTK_AWS':
    create_hscfg_file(config.wtk_aws_key)
    wtk_pname = 'h5pyd'
    wtk_years = list(range(2007, 2015))
    wtk_fnames = ['/nrel/wtk/conus/wtk_conus_' + str(yr) + '.h5'
                  for yr in wtk_years]
elif config.wtk_data_source == 'WTK_EAGLE':
    wtk_pname = 'h5py'
    wtk_years = list(range(2007, 2015))
    wtk_fnames = ['/datasets/WIND/conus/v1.0.0/wtk_conus_' + str(yr) + '.h5'
                  for yr in wtk_years]
elif config.wtk_data_source == 'WTK_LED_EAGLE':
    wtk_pname = 'h5py'
    wtk_years = list(range(2018, 2019))
    wtk_fnames = ['/lustre/eaglefs/shared-projects/wtk-led/' +
                  'ERA5_En1/wtk_ERA5_En1_' + str(yr) + '.h5'
                  for yr in wtk_years]
    varnames += thermal_varnames
    compute_thermals = True


# Extract WTK data source locations
if config.mode in ['seasonal', 'snapshot']:
    print('\n--- Extracting atmospheric data from', config.wtk_data_source)
    print('Mode: ', config.mode)
    wtklocs_exists = file_exists(config.terrain_data_dir, 'wtk_indices.txt')
    if wtklocs_exists:
        print('Using saved WTK locations')
        wtk_indices = load_data(config.terrain_data_dir, 'wtk_indices.txt',
                                dtype=int)
        wtk_xylocs = load_data(config.terrain_data_dir, 'wtk_xylocs.npy')
    else:
        wtk_indices, wtk_xylocs, wtk_fields = extract_wtk_locations(
            extent_lonlat, extent_km, wtk_pname, wtk_fnames[0])
        save_data(config.terrain_data_dir, 'wtk_fields.txt', wtk_fields)
        save_data(config.terrain_data_dir, 'wtk_indices.txt', wtk_indices,
                  fmt='%i')
        save_data(config.terrain_data_dir, 'wtk_xylocs.npy', wtk_xylocs)
    print('Got {0:d} WTK data source locations'.format(wtk_indices.size))


# Determine the datetimes for importing atmospehric data
dtimes = []
makedir_if_not_exists(config.mode_data_dir)
if config.mode == 'seasonal':
    print('Requested {0:d} counts:'.format(config.wtk_seasonal_count), end="")
    print(' months=', config.wtk_seasonal_months, end="")
    print(', Time of day=', config.wtk_seasonal_timeofday)
    if config.wtk_use_saved_data:
        saved_count = count_dirs_in(config.mode_data_dir)
        print('Found {0:d} saved instance(s) (wtk_use_saved_data=True)'
              .format(saved_count))
    else:
        remove_dirs_in(config.mode_data_dir)
        print('Deleting saved data (wtk_use_saved_data=False)')
        saved_count = 0
    rem_count = config.wtk_seasonal_count - saved_count
    if rem_count > 0:
        sun_lonlat = [(extent_lonlat[0] + extent_lonlat[1]) / 2.,
                      (extent_lonlat[2] + extent_lonlat[3]) / 2.]
        dtimes = get_random_datetimes(
            rem_count, sun_lonlat, wtk_years, config.wtk_seasonal_months,
            config.wtk_seasonal_timeofday)
elif config.mode == 'snapshot':
    dtime = datetime(*config.wtk_snapshot_datetime)
    dtime_id = dtime.strftime(config.dtime_format)
    data_dir = config.mode_data_dir + dtime_id + '/'
    snapshot_exists = file_exists(data_dir, dtime_id + '_wtk.csv')
    if snapshot_exists and config.wtk_use_saved_data:
        print('Found saved snapshot instance')
    else:
        dtimes = [dtime]
        rem_count = 1
elif config.mode == 'predefined':  # predefined mode
    print('\n--- Computing orographic updrafts in predefined mode')
    print('Wind speed: {0:6.1f}, dirn (CW from N): {1:6.1f}'
          .format(config.predefined_windspeed, config.predefined_winddirn))
    wspeed = config.predefined_windspeed * np.ones(np.shape(terrain_alt))
    wdirn = config.predefined_winddirn * np.ones(np.shape(terrain_alt))
    orograph = orographic_updraft_function(
        wspeed, wdirn * np.pi / 180., terrain_slope, terrain_aspect)
    dtime_id = get_predefined_mode_id(config.predefined_windspeed,
                                      config.predefined_winddirn)
    data_dir = config.mode_data_dir + dtime_id + '/'
    makedir_if_not_exists(data_dir)
    save_data(data_dir, dtime_id + '_orograph.npy',
              orograph.astype(np.float32))
else:
    print('Incorrect mode:', config.mode)
    print('Options: seasonal, snapshot, predefined')


# extract WTK data and compute updrafts
if dtimes:
    n_cpu = min(len(dtimes), config.max_cores)

    print('Extracting {0:d} new instances:'.format(rem_count))
    with mp.Pool(n_cpu) as pool:
        wtkdfs = pool.map(lambda dtime: extract_wtk_data(
            dtime, wtk_indices, varnames, wtk_pname,
            wtk_fnames[wtk_years.index(dtime.year)]
        ), dtimes)

    start_time = initiate_timer('Computing orographic updraft')
    with mp.Pool(n_cpu) as pool:
        orographs = pool.map(lambda idf: compute_orographic_updraft(
            config.terrain_res,
            extent_km,
            terrain_slope,
            terrain_aspect,
            wtk_indices,
            wtk_xylocs[0, :],
            wtk_xylocs[1, :],
            idf[oro_varnames[0]].values.flatten(),
            idf[oro_varnames[1]].values.flatten(),
            config.wtk_interpolation_type
        ), wtkdfs)
    print_elapsed_time(start_time)

    if compute_thermals:
        start_time = initiate_timer('Computing potential temperature')
        with mp.Pool(n_cpu) as pool:
            pot_temps = pool.map(lambda idf: compute_potential_temperature(
                idf[thermal_varnames[0]].values.flatten(),
                idf[thermal_varnames[1]].values.flatten()
            ), wtkdfs)
        print_elapsed_time(start_time)

        start_time = initiate_timer('Computing deardoff velocity')
        with mp.Pool(n_cpu) as pool:
            deardoffs = pool.map(lambda i: compute_deardoff_velocity(
                config.terrain_res,
                extent_km,
                wtk_indices,
                wtk_xylocs[0, :],
                wtk_xylocs[1, :],
                pot_temps[i],
                wtkdfs[i][thermal_varnames[2]].values.flatten(),
                wtkdfs[i][thermal_varnames[3]].values.flatten(),
                config.wtk_interpolation_type
            ), range(len(wtkdfs)))
        print_elapsed_time(start_time)

        start_time = initiate_timer('Computing thermal updraft')
        with mp.Pool(n_cpu) as pool:
            thermals = pool.map(lambda i: compute_thermal_updraft(
                config.wtk_thermals_agl,
                deardoffs[i],
                wtkdfs[i][thermal_varnames[2]].values.flatten()
            ), range(len(wtkdfs)))
        print_elapsed_time(start_time)

    for i in range(len(wtkdfs)):
        dtime_id = dtimes[i].strftime(config.dtime_format)
        data_dir = config.mode_data_dir + dtime_id + '/'
        makedir_if_not_exists(data_dir)
        save_data(data_dir, dtime_id + '_wtk.csv', wtkdfs[i])
        save_data(data_dir, dtime_id + '_orograph.npy', orographs[i])
        if compute_thermals:
            save_data(data_dir, dtime_id + '_potential_temperature.npy',
                      pot_temps[i])
            save_data(data_dir, dtime_id + '_thermal.npy', thermals[i])
