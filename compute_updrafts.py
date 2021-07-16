import pathos.multiprocessing as mp
from config import *
import sys
from tcfuncs.common import *
from tcfuncs.config_tools import initialize_code
from tcfuncs.updrafts import *

# Intialize
#config = initialize_code(os.path.basename(__file__), eval(sys.argv[1]))
config = initialize_code('config.py', eval(sys.argv[1]))

# Load saved turbine/terrain data
try:
    df_wtk_locations = load_data(config.data_dir, 'wtk_locations.csv')
    wtk_xgrid = df_wtk_locations.to_numpy().T[2]
    wtk_ygrid = df_wtk_locations.to_numpy().T[3]
    df_terrain_bounds = load_data(config.data_dir, 'terrain_bounds.csv')
    tr_extent = df_terrain_bounds.to_numpy().T[2:4, :].flatten()
    tr_xgrid = np.arange(tr_extent[0], tr_extent[1], config.terrain_res)
    tr_ygrid = np.arange(tr_extent[2], tr_extent[3], config.terrain_res)
    tr_slope = load_data(config.data_dir, 'terrain_slope.npy')
    tr_aspect = load_data(config.data_dir, 'terrain_aspect.npy')
    tr_gridsize = tr_slope.shape
except:
    raise Exception('\nRun extract_data.py first!')

# Load wind toolkit data
datetimes = get_saved_datetimes(
    config.data_dir, config.datetime_format, '_wtk.csv')
if not datetimes:
    raise Exception('\nRun extract_data.py first!')
datetime_ids = [t.strftime(config.datetime_format) for t in datetimes]
data_dirnames = [config.data_dir + datetime_ids[i] +
                 '/' for i in range(len(datetimes))]
wtk_dfs = [load_data(data_dirnames[i], datetime_ids[i] + '_wtk.csv',
                     header=[0, 1]) for i in range(len(datetimes))]
n_cpu = min(len(datetimes), config.max_cores)
print('\n--- Computing atmospheric vars for {0:d} datetimes using {1:d} cores'
      .format(len(datetimes), n_cpu))

# potential temperature
# start_time = initiate_timer('potential temperature')
# with mp.Pool(n_cpu) as pool:
#     ptemps = pool.map(
#         lambda i: compute_potential_temperature(
#             wtk_dfs[i][config.thermal_varnames[0]].to_numpy().flatten(),
#             wtk_dfs[i][config.thermal_varnames[1]].to_numpy().flatten()
#         ), range(len(datetimes)))
# for i in range(len(datetimes)):
#     wtk_dfs[i][('potential_temperature', 'C')] = ptemps[i].astype(np.float32)
#     save_data(data_dirnames[i], datetime_ids[i] + '_wtk.csv', wtk_dfs[i])
# print_elapsed_time(start_time)

# orographic updraft
start_time = initiate_timer('orographic updraft')
with mp.Pool(n_cpu) as pool:
    orographs = pool.map(
        lambda i: compute_orographic_updraft(
            datetimes[i],
            config.terrain_res,
            tr_extent,
            tr_slope,
            tr_aspect,
            df_wtk_locations.index,
            wtk_xgrid,
            wtk_ygrid,
            wtk_dfs[i][config.oro_varnames[0]].to_numpy().flatten(),
            wtk_dfs[i][config.oro_varnames[1]].to_numpy().flatten(),
            config.interp_type
        ), range(len(datetimes)))
for i in range(len(datetimes)):
    orograph = orographs[i].astype(np.float32)
    save_data(data_dirnames[i], datetime_ids[i] + '_orograph.npy', orograph)
print_elapsed_time(start_time)

# # deardoff velocity
# start_time = initiate_timer('deardoff velocity')
# with mp.Pool(n_cpu) as pool:
#     deardoffs = pool.map(
#         lambda i: compute_deardoff_velocity(
#             datetimes[i],
#             config.terrain_res,
#             tr_extent,
#             df_wtk_locations.index,
#             wtk_xgrid,
#             wtk_ygrid,
#             ptemps[i],
#             wtk_dfs[i][config.thermal_varnames[2]].to_numpy().flatten(),
#             wtk_dfs[i][config.thermal_varnames[3]].to_numpy().flatten(),
#             config.interp_type
#         ), range(len(datetimes)))
# for i in range(len(datetimes)):
#     deardoff = deardoffs[i][0].astype(np.float32)
#     save_data(data_dirnames[i], datetime_ids[i] + '_deardoff.npy', deardoff)
# print_elapsed_time(start_time)

# # thermal updrafts
# start_time = initiate_timer(
#     'thermals at ' + str(config.thermal_altitude) + ' m altitude')
# with mp.Pool(n_cpu) as pool:
#     thermals = pool.map(
#         lambda i: compute_thermal_updraft(
#             config.thermal_altitude,
#             deardoffs[i][0],
#             deardoffs[i][1]
#         ), range(len(datetimes)))
# for i in range(len(datetimes)):
#     thermal = thermals[i].astype(np.float32)
#     save_data(data_dirnames[i], datetime_ids[i] + '_thermal.npy', thermal)
# print_elapsed_time(start_time)


# percentage orographic above threshold
# start_time = initiate_timer('probability updraft above threshold')
# with mp.Pool(n_cpu) as pool:
#     oro_percs = pool.map(
#         lambda i: weibull_percentage_above_threshold(
#             orographs[i],
#             config.weibull_k,
#             config.updraft_threshold
#         ), range(len(datetimes)))
# for i in range(len(datetimes)):
#     oro_perc = oro_percs[i].astype(np.float32)
#     save_data(data_dirnames[i], datetime_ids[i] + '_prob.npy', oro_perc)
# print_elapsed_time(start_time)


# stats of updrafts
# index = df_wtk_locations.index[0]
# for i in range(len(datetimes)):
#     orographs[i][50, 50]


# for debugging thermal updraft calculation
# check_updraft_computation = 0
# if check_updraft_computation:
#     print('\n---- Debugging:')
#     my_cmap = 'bwr'
#     index = 0
#     timestamp = datetimes[index]
#     datetime_id = timestamp.strftime(config.datetime_format)
#     dfwtk = load_data(config.data_dir, datetime_id + '_wtk.csv')

#     deardoff, intdata = compute_deardoff_velocity(
#         timestamp,
#         config.terrain_res,
#         tr_extent,
#         df_wtk_locations.index,
#         wtk_xgrid,
#         wtk_ygrid,
#         dfwtk[config.thermal_varnames[0]].to_numpy(),
#         dfwtk[config.thermal_varnames[1]].to_numpy(),
#         dfwtk[config.thermal_varnames[2]].to_numpy(),
#         dfwtk[config.thermal_varnames[3]].to_numpy(),
#         config.rbf_interp_type,
#         check_updraft_computation)

#     fig, ax = plt.subplots(figsize=config.fig_size)
#     cm = ax.imshow(deardoff, cmap=my_cmap, origin='lower', vmin=0., vmax=2.)
#     cb, lg = create_gis_axis(fig, ax, cm, 10)
#     plt.title(timestamp.strftime('%I %p, %x'))
#     cb.set_label('Deardoff velocity')
#     save_fig(fig, config.fig_dir, datetime_id + '_deardoff.png')

#     for i, varname in enumerate(config.thermal_varnames):
#         print('{0:s}: {1:.2f},{2:.2f} / {3:.2f},{4:.2f} '.format(
#             varname, dfwtk[varname].min(), dfwtk[varname].max(),
#             np.amin(intdata[i, :, :]), np.amax(intdata[i, :, :])))
#         fig, ax = plt.subplots(figsize=config.fig_size)
#         cm = ax.pcolormesh(tr_xgrid, tr_ygrid, intdata[i, :, :], cmap=my_cmap)
#         cm = ax.scatter(wtk_xgrid, wtk_ygrid, c=dfwtk[varname], cmap=my_cmap)
#         cb, lg = create_gis_axis(fig, ax, cm, 10)
#         plt.title(timestamp.strftime('%I %p, %x'))
#         cb.set_label(varname)
#         save_fig(fig, config.fig_dir, datetime_id + '_' + varname + '.png')

#     # thermal updraft calculation at multiple altitutdes
#     heights = [50, 100, 200, 500, 1000, 1500, 2000]
#     cbounds = [0, 2.]
#     tcmap = get_transparent_cmap('brg_r', config.updraft_threshold, cbounds)
#     print('Percentage times updraft above {0:4.2f} m/s:'.format(
#         config.updraft_threshold))
#     for z in heights:
#         thermals = compute_thermals_at_height_z(z, deardoff, intdata[2])
#         prec_usable = 100 * np.count_nonzero(
#             thermals > config.updraft_threshold) / thermals.size
#         print('{0:6.1f} m: {1:4.1f} %'.format(z, prec_usable))

#         fig, ax = plt.subplots(figsize=config.fig_size)
#         cm = ax.imshow(thermals, origin='lower',
#                        cmap=my_cmap, vmin=cbounds[0], vmax=cbounds[1])
#         cb, lg = create_gis_axis(fig, ax, cm, 10)
#         cb.set_ticks([0., config.updraft_threshold, cbounds[1]])
#         plt.title(timestamp.strftime('%I %p, %x'))
#         cb.set_label('Thermal updraft at ' + str(z) + 'm altitude')
#         save_fig(fig, config.fig_dir, datetime_id +
#                  '_thermal_' + str(int(z)) + 'm.png')

#         # fig, ax = plt.subplots(figsize=config.fig_size)
#         # cm = ax.imshow(tr_features[0, :, :], origin='lower',
#         #                cmap='Greys', alpha=0.9)
#         # cm = ax.imshow(thermals, origin='lower',
#         #                cmap=tcmap, vmin=0., vmax=2 * config.updraft_threshold)
#         # cb, lg = create_gis_axis(fig, ax, cm, 10)
#         # cb.set_ticks([0., config.updraft_threshold,
#         #               2 * config.updraft_threshold])
#         # plt.title(timestamp.strftime('%I %p, %x'))
#         # cb.set_label('Usable thermal updraft at ' + str(z) + 'm altitude')
#         # save_fig(fig, config.fig_dir, datetime_id +
#         #          '_thermal_' + str(int(z)) + 'm.png')


#     # orographic updrafts
#     orographs, intdata = compute_orographic_updraft(
#         timestamp,
#         config.terrain_res,
#         tr_extent,
#         tr_slope,
#         tr_aspect,
#         df_wtk_locations.index,
#         wtk_xgrid,
#         wtk_ygrid,
#         dfwtk[config.oro_varnames[0]].to_numpy(),
#         dfwtk[config.oro_varnames[1]].to_numpy(),
#         config.rbf_interp_type,
#         check_updraft_computation
#     )

#     fig, ax = plt.subplots(figsize=config.fig_size)
#     cm = ax.imshow(orographs, origin='lower',
#                    cmap=my_cmap, vmin=0., vmax=2 * config.updraft_threshold)
#     cb, lg = create_gis_axis(fig, ax, cm, 10)
#     plt.title(timestamp.strftime('%I %p, %x'))
#     cb.set_label('Orographic updraft')
#     save_fig(fig, config.fig_dir, datetime_id + '_orographic.png')

#     for i, varname in enumerate(config.oro_varnames):
#         print('{0:s}: {1:.2f},{2:.2f} / {3:.2f},{4:.2f} '.format(
#             varname, dfwtk[varname].min(), dfwtk[varname].max(),
#             np.amin(intdata[i, :, :]), np.amax(intdata[i, :, :])))
#         fig, ax = plt.subplots(figsize=config.fig_size)
#         cm = ax.pcolormesh(tr_xgrid, tr_ygrid, intdata[i, :, :], cmap=my_cmap)
#         cm = ax.scatter(wtk_xgrid, wtk_ygrid, c=dfwtk[varname], cmap=my_cmap)
#         cb, lg = create_gis_axis(fig, ax, cm, 10)
#         cb.set_label(varname)
#         plt.title(timestamp.strftime('%I %p, %x'))
#         save_fig(fig, config.fig_dir, datetime_id + '_' + varname + '.png')
#     plt.close('all')
