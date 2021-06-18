import pathos.multiprocessing as mp

from tcfuncs.common import *
from tcfuncs.config_tools import *
from tcfuncs.terrain import *
from tcfuncs.wfarm import *
from tcfuncs.wtk import *

# Intialize
config = initialize_code(os.path.basename(__file__), eval(sys.argv[1]))
create_hscfg_file(config.AWS_api_key)  # only when reading AWS WTK data

# Extract wind farm data # print(get_all_wfarms('WY'))
df_turbines = extract_turbine_data(
    config.wfarm_state,
    config.wfarm_names,
    config.wfarm_fname
)
save_data(config.data_dir, 'turbine_data.csv', df_turbines)

# Determine terrain bounds
southwest_lonlat = list(df_turbines[['xlong', 'ylat']].min())  # southwest
# southwest_lonlat = [-105.773689, 42.944092]
df_terrain_bounds = get_terrain_bounds(
    southwest_lonlat,
    config.terrain_southwest_pad,
    config.terrain_width
)
save_data(config.data_dir, 'terrain_bounds.csv', df_terrain_bounds)

# Extract terrain data
sys.path.append('/Users/rsandhu/opt')
terrain_altitude = extract_terrain_elevation_srtm(
    df_terrain_bounds.xlong,
    df_terrain_bounds.ylat,
    df_terrain_bounds.xkm,
    df_terrain_bounds.ykm,
    config.terrain_res
)
terrain_slope, terrain_aspect = compute_slope_and_aspect(
    terrain_altitude,
    config.terrain_res
)
save_data(config.data_dir, 'terrain_altitude.npy',
          terrain_altitude.astype(np.float32))
save_data(config.data_dir, 'terrain_slope.npy',
          terrain_slope.astype(np.float32))
save_data(config.data_dir, 'terrain_aspect.npy',
          terrain_aspect.astype(np.float32))

# Extract WTK locations
wtk_years, module_name, fname_at = get_wtk_sourceinfo(config.wtk_source)
df_wtk_locations = extract_wtk_locations(
    df_terrain_bounds.xlong,
    df_terrain_bounds.ylat,
    module_name,
    fname_at(wtk_years[0])
)
# drop that weird point
df_wtk_locations.drop(df_wtk_locations.index[194], inplace=True)
save_data(config.data_dir, 'wtk_locations.csv', df_wtk_locations)

# Extract WTK data
# wtk_var_names = config.oro_varnames + config.thermal_varnames
wtk_var_names = config.oro_varnames
if config.wtk_mode == 'seasonal':
    datetimes = get_saved_datetimes(config.data_dir, config.datetime_format,
                                    '_wtk.csv')
    print('\n--- Extracting WTK data (Seasonal): {0:d} requested, {1:d} exist'
          .format(config.seasonal_updraft_count, len(datetimes)))
    if config.seasonal_updraft_count > len(datetimes):
        print('years: ', wtk_years, '\nmonths: ', config.wtk_seasonal_months,
              '\nTime of day: ', config.wtk_seasonal_timeofday)
        start_time = initiate_timer('Randomly selected time instants:\n')
        mean_lonlat = list(df_terrain_bounds[['xlong', 'ylat']].mean())
        remaining_count = config.seasonal_updraft_count - len(datetimes)
        new_datetimes = get_random_datetimes(remaining_count,
                                             mean_lonlat,
                                             wtk_years,
                                             config.wtk_seasonal_months,
                                             config.wtk_seasonal_timeofday,
                                             config.wfarm_timezone)
        n_cpu = min(len(new_datetimes), config.max_cores)
        with mp.Pool(n_cpu) as pool:
            new_wtkdfs = pool.map(lambda idt: extract_wtk_data(
                idt,
                df_wtk_locations.index,
                wtk_var_names,
                module_name,
                fname_at(idt.year)
            ), new_datetimes)
        for new_datetime, wtkdf in zip(new_datetimes, new_wtkdfs):
            datetime_id = new_datetime.strftime(config.datetime_format)
            data_dirname = config.data_dir + datetime_id + '/'
            makedir_if_not_exists(data_dirname)
            save_data(data_dirname, datetime_id + '_wtk.csv', wtkdf)
        print_elapsed_time(start_time)
elif config.wtk_mode == 'snapshot':
    dtime = datetime(*config.wtk_snapshot_datetime)
    print('\n--- Extracting WTK data (Snapshot): ')
    wtkdf = extract_wtk_data(
        dtime,
        df_wtk_locations.index,
        wtk_var_names,
        module_name,
        fname_at(dtime.year))
    dtime_id = dtime.strftime(config.datetime_format)
    data_dirname = config.data_dir + dtime_id + '/'
    makedir_if_not_exists(data_dirname)
    save_data(data_dirname, dtime_id + '_wtk.csv', wtkdf)
