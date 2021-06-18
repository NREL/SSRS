#!/usr/bin/env python
# coding: utf-8

import matplotlib.patches as mpatches
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde

from tcfuncs.common import *
# Load local module that contains model specific functions
# import json
from tcfuncs.config_tools import initialize_code, makedir_if_not_exists
from tcfuncs.tcmodel import *

# Intialize
config = initialize_code(os.path.basename(__file__), eval(sys.argv[1]))
max_cores_for_plotting = 64
# delete_files_of_type(config.fig_dir, '.png')
try:
    print('\n--- Generating plots')
    df_turb = load_data(config.data_dir, 'turbine_data.csv')
    df_wtk_locations = load_data(config.data_dir, 'wtk_locations.csv')
    wtk_xgrid = df_wtk_locations.to_numpy().T[2]
    wtk_ygrid = df_wtk_locations.to_numpy().T[3]
    df_terrain_bounds = load_data(config.data_dir, 'terrain_bounds.csv')
    tr_extent = df_terrain_bounds.to_numpy().T[2:4, :].flatten()
    tr_xgrid = np.arange(tr_extent[0], tr_extent[1], config.terrain_res)
    tr_ygrid = np.arange(tr_extent[2], tr_extent[3], config.terrain_res)
    tr_xmesh, tr_ymesh = np.meshgrid(tr_xgrid, tr_ygrid)
    tr_altitude = load_data(config.data_dir, 'terrain_altitude.npy')
    tr_slope = load_data(config.data_dir, 'terrain_slope.npy')
    tr_aspect = load_data(config.data_dir, 'terrain_aspect.npy')
    tr_gridsize = tr_altitude.shape
except:
    raise Exception('\nRun extract_data.py first!')


def plot_all_windfarms(ax, lbl=True):  # wind farm locations
    mrkrs = ('1b', '2b', '3b', '4y', '+r', 'xc')
    for i, wfarm in enumerate(config.wfarm_names):
        turb_xgrid = df_turb.loc[df_turb.p_name == wfarm, 'xkm'].to_numpy()
        turb_ygrid = df_turb.loc[df_turb.p_name == wfarm, 'ykm'].to_numpy()
        if i == 0:
            l, = ax.plot(turb_xgrid, turb_ygrid, mrkrs[i], markersize=5,
                         alpha=0.5, label='Turbine locations')
        else:
            l, = ax.plot(turb_xgrid, turb_ygrid,
                         mrkrs[i], markersize=5, alpha=0.5)
        if lbl:
            l.set_label(config.wfarm_labels[i])
        if i == 0 or i == 1 or i == 2:
            width = max(turb_xgrid) - min(turb_xgrid) + 2
            height = max(turb_ygrid) - min(turb_ygrid) + 2
            rect = mpatches.Rectangle((min(turb_xgrid) - 1, min(turb_ygrid) - 1),
                                      width, height,
                                      linewidth=1, edgecolor='k',
                                      facecolor='none', zorder=20)
            ax.add_patch(rect)


def plot_wtk_locations(ax, lbl=False):  # wtk xy locations
    l, = ax.plot(wtk_xgrid, wtk_ygrid, '*k', markersize=1.75)
    # for i, v in enumerate(df_wtk_locations.index):
    #     ax.text(wtk_xgrid[i], wtk_ygrid[i] + 0.15, "%d" % (i),
    #             ha="center", fontsize=4)
    if lbl:
        l.set_label('WTK')


def plot_background_terrain(ax):  # background terrain
    cm = ax.imshow(tr_altitude, alpha=0.75, cmap='Greys',
                   origin='lower', extent=tr_extent)
    return cm


def plot_wtk_data(dtime):
    cmap = 'viridis'  # 'coolwarm'  # 'bwr'
    dtime_id = dtime.strftime(config.datetime_format)
    data_dir = config.data_dir + dtime_id + '/'
    df = load_data(data_dir, dtime_id + '_wtk.csv', header=[0, 1])
    fig_dir = config.fig_dir + dtime_id + '/'
    makedir_if_not_exists(fig_dir)
    for i, (varname, varunits) in enumerate(df.columns):
        vardata = df.loc[:, varname].to_numpy().flatten()
        fig, ax = plt.subplots(figsize=config.fig_size)
        plot_all_windfarms(ax, False)
        vargrid = griddata(np.array([wtk_xgrid, wtk_ygrid]).T, vardata,
                           (tr_xmesh, tr_ymesh), method=config.interp_type)
        if (varname == 'surface_heat_flux'):
            cmap = 'viridis'
        elif (varname == 'temperature_100m'):
            cmap = 'viridis'
        elif (varname == 'boundary_layer_height'):
            cmap = 'viridis'
        else:
            cmap = 'viridis'
        if varname == 'winddirection_100m':
            cm = ax.imshow(vargrid, extent=tr_extent,
                           origin='lower', cmap=cmap, alpha=0.8)
            # cm = ax.imshow(vargrid, extent=tr_extent,
            #                origin='lower', cmap='hsv', vmin=0., vmax=360.)
        else:
            cm = ax.imshow(vargrid, extent=tr_extent,
                           origin='lower', cmap=cmap, alpha=0.8)

        # cm = ax.scatter(wtk_xgrid, wtk_ygrid, s=5, c=vardata, cmap=cmap)
        cb, _ = create_gis_axis(fig, ax, cm)
        # if varname == 'winddirection_100m':
        #     cb.set_ticks(np.linspace(0, 360, 9))
        plt.xlim(tr_extent[0:2])
        plt.ylim(tr_extent[2:4])
        # plt.title(dtime.strftime('%I %p, %x'))
        # cb.set_label(varname + ' (' + varunits + ')')
        cb.set_label(config.var_labels[i])
        save_fig(fig, fig_dir, dtime_id + '_' + varname + '.png', 400)


def plot_interpolated_data(dtime, fnames, lbls):
    # cmap = get_transparent_cmap('brg_r', config.updraft_threshold, cbounds)
    cmap = 'viridis'
    dtime_id = dtime.strftime(config.datetime_format)
    data_dir = config.data_dir + dtime_id + '/'
    fig_dir = config.fig_dir + dtime_id + '/'
    makedir_if_not_exists(fig_dir)
    for fname, lbl in zip(fnames, lbls):
        vardata = load_data(data_dir, dtime_id + fname + '.npy')
        fig, ax = plt.subplots(figsize=config.fig_size)
        if fname == '_thermal':
            cm = ax.imshow(vardata, cmap=cmap,
                           origin='lower', extent=tr_extent)
        else:
            cm = ax.imshow(vardata, cmap='Oranges', origin='lower', extent=tr_extent,
                           vmin=0., vmax=1.)
        plot_all_windfarms(ax, False)
        cb, _ = create_gis_axis(fig, ax, cm)
        # plt.title(dtime.strftime('%I %p, %x'))
        cb.set_label(lbl)
        save_fig(fig, fig_dir, dtime_id + fname + '.png', 400)
    # orograph = load_data(data_dir, dtime_id + '_orograph.npy')
    # thermal = load_data(data_dir, dtime_id + '_thermal.npy')
    # net = np.add(orograph, thermal)
    # fig, ax = plt.subplots(figsize=config.fig_size)
    # cm = ax.imshow(net, cmap=cmap, origin='lower', extent=tr_extent,
    #                vmin=0., vmax=1.)
    # cb, _ = create_gis_axis(fig, ax, cm)
    # #plt.title(dtime.strftime('%I %p, %x'))
    # cb.set_label('Orographic + thermal updraft (m/s)')
    # save_fig(fig, fig_dir, dtime_id + '_net.png', 400)


def plot_pdfs_of_wtk_variables(wtk_index, dtimes):
    fig_dir = config.fig_dir + str(wtk_index) + '/'
    makedir_if_not_exists(fig_dir)
    dtime_ids = [t.strftime(config.datetime_format) for t in dtimes]
    data_dirs = [config.data_dir + dtime_ids[i] +
                 '/' for i in range(len(dtimes))]
    wtk_dfs = [load_data(data_dirs[i], dtime_ids[i] + '_wtk.csv',
                         header=[0, 1]) for i in range(len(dtimes))]
    # wtk_smpls = np.empty((len(datetimes), wtk_dfs[0].columns.shape[0]))
    for j, (varname, varunits) in enumerate(wtk_dfs[0].columns):
        pointdata = np.array([])
        for i, df in enumerate(wtk_dfs):
            pointdata = np.append(pointdata, df.iloc[wtk_index, j])
        fig, ax = plt.subplots(figsize=(4, 3))
        xmin, xmid, xmax = get_min_mid_max(pointdata, 0.25)
        density = gaussian_kde(pointdata)
        xs = np.linspace(xmin, xmax, 100)
        ax.plot(xs, density(xs), '-b')
        create_pdf_axis(fig, ax, xmin, xmid, xmax)
        plt.xlabel(varname + ' (' + varunits + ')')
        save_fig(fig, fig_dir, str(wtk_index) + '_' + varname + '.png', 400)


def plot_pdfs_of_derived_variables(wtk_index, dtimes, fnames, lbls):
    fig_dir = config.fig_dir + str(wtk_index) + '/'
    makedir_if_not_exists(fig_dir)
    dtime_ids = [t.strftime(config.datetime_format) for t in dtimes]
    data_dirs = [config.data_dir + dtime_ids[i] +
                 '/' for i in range(len(dtimes))]
    idx = (np.abs(tr_xgrid - wtk_xgrid[wtk_index])).argmin()
    idy = (np.abs(tr_ygrid - wtk_ygrid[wtk_index])).argmin()
    # print('x-loc: ', tr_xgrid[idx], wtk_xgrid[loc_index])
    # print('y-loc: ', tr_ygrid[idy], wtk_ygrid[loc_index])
    for fname, lbl in zip(fnames, lbls):
        pointdata = np.array([])
        for i in range(len(dtimes)):
            vardata = load_data(data_dirs[i], dtime_ids[i] + fname + '.npy')
            pointdata = np.append(pointdata, vardata[idy, idx])
        fig, ax = plt.subplots(figsize=(4, 3))
        xmin, xmid, xmax = get_min_mid_max(pointdata, 0.25)
        density = gaussian_kde(pointdata)
        xs = np.linspace(xmin, xmax, 100)
        ax.plot(xs, density(xs), '-b')
        create_pdf_axis(fig, ax, xmin, xmid, xmax)
        save_fig(fig, fig_dir, str(wtk_index) + fname + '.png', 400)


def plot_tcmodel_tracks_and_pmap(dtime, cases):
    pmap_krad = 10  # higher -> smoother pmap
    padding = [x * 2. for x in [1, -1, 1, -1]]
    pmap_extent = [tr_extent[i] + padding[i] for i in range(len(padding))]
    pmap_extent_index = get_minmax_indices(tr_xgrid, tr_ygrid, pmap_extent)
    dtime_id = dtime.strftime(config.datetime_format)
    run_id = dtime_id + '_' + config.bndry_condition + '_'
    data_dir = config.data_dir + dtime_id + '/'
    fig_dir = config.fig_dir + dtime_id + '/'
    makedir_if_not_exists(fig_dir)
    for _, lbl in enumerate(cases):
        if file_exists(data_dir, run_id + lbl + '_tracks.pkl'):
            tracks = load_data(data_dir, run_id + lbl + '_tracks.pkl')
            fig, ax = plt.subplots(figsize=config.fig_size)
            cm = plot_background_terrain(ax)
            lwidth = 0.15 if len(tracks) > 251 else 0.45
            for kk, track in enumerate(tracks):
                if kk == 0:
                    ax.plot(tr_xgrid[track[:, 1]], tr_ygrid[track[:, 0]],
                            '-r', linewidth=lwidth, alpha=0.9,
                            label='Simulated paths')
                ax.plot(tr_xgrid[track[:, 1]], tr_ygrid[track[:, 0]],
                        '-r', linewidth=lwidth, alpha=0.4)
            plot_all_windfarms(ax, False)
            cb, lg = create_gis_axis(fig, ax, cm)
            cb.remove()
            lg.legendHandles[0].set_linewidth(1.0)
            # plt.title(dtime.strftime('%I %p, %x'))
            save_fig(fig, fig_dir, run_id + lbl + '_tracks.png', 400)

            counts = compute_count_matrix(tr_gridsize, tracks)
            pmap = compute_presence(counts, pmap_extent_index, pmap_krad)
            pmap /= np.amax(pmap)
            fig, ax = plt.subplots(figsize=config.fig_size)
            cm = ax.imshow(pmap, extent=pmap_extent, origin='lower',
                           cmap='Reds', alpha=0.8, vmax=0.6)
            plot_all_windfarms(ax, False)
            cb, _ = create_gis_axis(fig, ax, cm)
            cb.remove()
            # plt.title(dtime.strftime('%I %p, %x'))
            save_fig(fig, fig_dir, run_id + lbl + '_pmap.png', 400)

            pmap_wfarm_pad_wf = 1.
            pmap_wf_rad_wf = 4
            for i, wf in enumerate(config.wfarm_names):
                tx = df_turb.loc[df_turb.p_name == wf, 'xkm'].to_numpy()
                ty = df_turb.loc[df_turb.p_name == wf, 'ykm'].to_numpy()
                pmap_extent_wf = [min(tx) - pmap_wfarm_pad_wf,
                                  max(tx) + pmap_wfarm_pad_wf,
                                  min(ty) - pmap_wfarm_pad_wf,
                                  max(ty) + pmap_wfarm_pad_wf]
                pmap_extent_index_wf = get_minmax_indices(
                    tr_xgrid, tr_ygrid, pmap_extent_wf)

                pmap_wf = compute_presence(
                    counts, pmap_extent_index_wf, pmap_wf_rad_wf)
                pmap_wf /= np.amax(pmap_wf)
                fig, ax = plt.subplots()
                # print(pmap_extent, pmap_extent_wf)
                cm = ax.imshow(pmap_wf, extent=pmap_extent_wf, origin='lower',
                               cmap='Reds', alpha=0.8, vmax=0.6)
                l, = ax.plot(tx, ty, '1b', markersize=10, alpha=0.85)
                cb, _ = create_gis_axis(fig, ax, cm, 1)
                cb.remove()
                ax.set_xlim([pmap_extent_wf[0], pmap_extent_wf[1]])
                ax.set_ylim([pmap_extent_wf[2], pmap_extent_wf[3]])
                plt.title(config.wfarm_labels[i])
                save_fig(fig, fig_dir, lbl + '_' +
                         wf.replace(" ", "_") + '_pmap_mean.png', 400)


def plot_tcmodel_potential(dtime, cases):
    cmap = 'tab20b'
    dtime_id = dtime.strftime(config.datetime_format)
    data_dir = config.data_dir + dtime_id + '/'
    fig_dir = config.fig_dir + dtime_id + '/'
    makedir_if_not_exists(fig_dir)
    run_id = dtime_id + '_' + config.bndry_condition + '_'
    for j, lbl in enumerate(cases):
        if file_exists(data_dir, run_id + lbl + '_potential.npy'):
            potential = load_data(data_dir, run_id + lbl + '_potential.npy')
            fig, ax = plt.subplots(figsize=config.fig_size)
            # cm = ax.imshow(potential, cmap=cmap,
            #                origin='lower', extent=tr_extent)
            lvls = np.linspace(0, np.amax(potential), 11)
            cm = ax.contourf(potential, lvls, cmap='YlOrBr', extent=tr_extent)
            cs = ax.contour(potential, lvls[1:-1],
                            linewidths=0.1, extent=tr_extent)
            ax.clabel(cs, fmt='%d', colors='k', fontsize=10)
            cb, _ = create_gis_axis(fig, ax, cm)
            cb.remove()
            # plt.title(dtime.strftime('%I %p, %x'))
            save_fig(fig, fig_dir, run_id + lbl + '_potential.png', 400)


def plot_tcmodel_summary(lbl, dtimes):
    pmap_krad = 10
    padding = [x * 2. for x in [1, -1, 1, -1]]
    fig_dir = config.fig_dir + 'summary/'
    makedir_if_not_exists(fig_dir)
    dtime_ids = [t.strftime(config.datetime_format) for t in dtimes]
    data_dirs = [config.data_dir + dtime_ids[i] +
                 '/' for i in range(len(dtimes))]
    counts_list = []
    k = 0
    for i in range(len(dtimes)):
        thermals = load_data(data_dirs[i], dtime_ids[i] + '_thermal.npy')
        med_thermal = np.mean(thermals)
        run_id = dtime_ids[i] + '_' + config.bndry_condition + '_'
        if med_thermal < 0.5:
            k += 1
            if file_exists(data_dirs[i], run_id + lbl + '_tracks.pkl'):
                tracks = load_data(data_dirs[i], run_id + lbl + '_tracks.pkl')
                counts = compute_count_matrix(tr_gridsize, tracks)
                # counts_list.append(np.divide(counts, np.amax(counts)))
                counts_list.append(counts)
        else:
            print(lbl, i, '/', k + 1, ' : ', dtime_ids[i], ' : ', med_thermal)
    save_a = np.array([len(dtimes), k])
    save_data(config.data_dir, 'perc_usable_thermals.txt', save_a)
    # print('i am here')
    counts_list = np.asarray(counts_list)
    pmap_extent = [tr_extent[i] + padding[i] for i in range(len(padding))]
    pmap_extent_index = get_minmax_indices(tr_xgrid, tr_ygrid, pmap_extent)
    counts_mean = np.mean(counts_list, axis=0)
    pmap = compute_presence(counts_mean, pmap_extent_index, pmap_krad)
    pmap /= np.amax(pmap)
    fig, ax = plt.subplots(figsize=config.fig_size)
    cm = ax.imshow(pmap, extent=pmap_extent, origin='lower',
                   cmap='Reds', alpha=0.9, vmax=0.9)
    plot_all_windfarms(ax, False)
    cb, _ = create_gis_axis(fig, ax, cm)
    cb.remove()
    # plt.title('Mean')
    save_data(config.data_dir, lbl + '_counts_mean.npy', counts_mean)
    save_fig(fig, fig_dir, lbl + '_pmap_mean.png', 400)

    counts_std = np.std(counts_list, axis=0)
    pmap = compute_presence(counts_std, pmap_extent_index, pmap_krad)
    pmap /= np.amax(pmap)
    fig, ax = plt.subplots(figsize=config.fig_size)
    cm = ax.imshow(pmap, extent=tr_extent, origin='lower',
                   cmap='Reds', alpha=0.9, vmax=0.9)
    plot_all_windfarms(ax, False)
    cb, _ = create_gis_axis(fig, ax, cm)
    cb.remove()
    plt.title('Standard deviation')
    save_data(config.data_dir, lbl + '_counts_std.npy', counts_std)
    save_fig(fig, fig_dir, lbl + '_pmap_std.png', 400)

    print('i am here')
    pmap_wfarm_pad = 1.
    pmap_wf_rad = 4
    for i, wf in enumerate(config.wfarm_names):
        tx = df_turb.loc[df_turb.p_name == wf, 'xkm'].to_numpy()
        ty = df_turb.loc[df_turb.p_name == wf, 'ykm'].to_numpy()
        pmap_extent = [min(tx) - pmap_wfarm_pad, max(tx) + pmap_wfarm_pad,
                       min(ty) - pmap_wfarm_pad, max(ty) + pmap_wfarm_pad]
        pmap_extent_index = get_minmax_indices(tr_xgrid, tr_ygrid, pmap_extent)

        pmap = compute_presence(counts_mean, pmap_extent_index, pmap_wf_rad)
        pmap /= np.amax(pmap)
        fig, ax = plt.subplots()
        cm = ax.imshow(pmap, extent=pmap_extent, origin='lower',
                       cmap='Reds', alpha=0.9, vmax=0.9)
        l, = ax.plot(tx, ty, '1b', markersize=10)
        cb, _ = create_gis_axis(fig, ax, cm, 1)
        cb.remove()
        plt.title(config.wfarm_labels[i])
        # plt.title('Mean')
        save_fig(fig, fig_dir, lbl + '_' + wf + '_pmap_mean.png', 400)

        # pmap = compute_presence(counts_std, pmap_extent_index, pmap_wf_rad)
        # pmap /= np.amax(pmap)
        # fig, ax = plt.subplots()
        # cm = ax.imshow(pmap, extent=pmap_extent, origin='lower',
        #                cmap='Blues', alpha=0.8, vmax=0.6)
        # l, = ax.plot(tx, ty, '1k', markersize=5)
        # cb, _ = create_gis_axis(fig, ax, cm, 1)
        # cb.remove()
        # plt.title('Standard deviation')
        # save_fig(fig, fig_dir, lbl + '_' + wf + '_pmap_mean.png')


# Plot terrain features
start_time = initiate_timer('plotting terrain features')
fig_dir = config.fig_dir + 'terrain/'
makedir_if_not_exists(fig_dir)

fig, ax = plt.subplots(figsize=config.fig_size)
cm = ax.imshow(tr_altitude,
               cmap='terrain', origin='lower', extent=tr_extent)
plot_all_windfarms(ax)
cb, lg = create_gis_axis(fig, ax, cm)
cb.set_label('Elevation (Km)')
save_fig(fig, fig_dir, 'terrain_elevation.png')

smax = 20.
fig, ax = plt.subplots(figsize=config.fig_size)
cm = ax.imshow(tr_slope * 180 / np.pi, vmin=0., vmax=smax,
               cmap='YlGnBu', origin='lower', extent=tr_extent)
plot_all_windfarms(ax)
cb, lg = create_gis_axis(fig, ax, cm)
cb.set_ticks(np.linspace(0, smax, 5))
cb.set_label('Slope (Degrees)')
save_fig(fig, fig_dir, 'terrain_slope.png')

fig, ax = plt.subplots(figsize=config.fig_size)
cm = ax.imshow(tr_aspect, vmin=0., vmax=2 * np.pi,
               cmap='hsv', origin='lower', extent=tr_extent)
plot_all_windfarms(ax)
cb, lg = create_gis_axis(fig, ax, cm)
cb.set_label('Aspect')
cb.set_ticks(np.linspace(0, 2 * np.pi, 9))
cb.set_ticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'])
save_fig(fig, fig_dir, 'terrain_aspect.png')

fig, ax = plt.subplots(figsize=config.fig_size)
cm = ax.imshow(tr_altitude, cmap='terrain', origin='lower', extent=tr_extent)
plot_wtk_locations(ax)
plot_all_windfarms(ax)
cb, lg = create_gis_axis(fig, ax, cm)
cb.set_label('Elevation (Km)')
save_fig(fig, fig_dir, 'terrain_with_wtkpoints.png')
print_elapsed_time(start_time)

# wtk data
datetimes = get_saved_datetimes(
    config.data_dir, config.datetime_format, '_wtk.csv')
if not datetimes:
    raise Exception('\nRun extract_data.py first!')
print('{0:d} instances of WTK data'.format(len(datetimes)), flush=True)
start_time = initiate_timer('  plotting WTK variables')
for dtime in datetimes:
    plot_wtk_data(dtime)
print_elapsed_time(start_time)

# interpolated variables
fnames = ('_orograph',)
lbls = ('Orographic updraft (m/s)',)
datetimes = get_saved_datetimes(
    config.data_dir, config.datetime_format, '_orograph.npy')
if not datetimes:
    sys.exit('No updraft data found!')
print('{0:d} instances of updrafts'.format(len(datetimes)), flush=True)
n_cpu = min(len(datetimes), max_cores_for_plotting)
start_time = initiate_timer('  plotting derived variables')
for dtime in datetimes:
    plot_interpolated_data(dtime, fnames, lbls)
print_elapsed_time(start_time)

# TCmodel output
datetimes = get_saved_datetimes(
    config.data_dir, config.datetime_format, '_tracks.pkl')
if not datetimes:
    sys.exit('No TCmodel output found!')
cases = ('orograph',)
print('{0:d} instances of TCmodel output'.format(len(datetimes)))
n_cpu = min(len(datetimes), max_cores_for_plotting)

start_time = initiate_timer('  plotting eagle tracks and presence maps')
for dtime in datetimes:
    plot_tcmodel_tracks_and_pmap(dtime, cases)
print_elapsed_time(start_time)

start_time = initiate_timer('  plotting migration energy potential')
for dtime in datetimes:
    plot_tcmodel_potential(dtime, cases)
print_elapsed_time(start_time)

# n_cpu = min(len(datetimes), max_cores_for_plotting)
# start_time = initiate_timer('  plotting WTK variables')
# with mp.Pool(n_cpu) as pool:
#     out = pool.map(lambda dtime: plot_wtk_data(
#         dtime
#     ), datetimes)
# print_elapsed_time(start_time)

# wtk_indices = (202, 437)
# n_cpu = min(len(wtk_indices), max_cores_for_plotting)
# start_time = initiate_timer('  plotting temporal pdfs')
# with mp.Pool(n_cpu) as pool:
#     out = pool.map(lambda ind: plot_pdfs_of_wtk_variables(
#         ind, datetimes
#     ), wtk_indices)
# print_elapsed_time(start_time)


# Interpolated data
# fnames = ('_orograph', '_deardoff', '_thermal', '_prob')
# lbls = ('Orographic updraft (m/s)', 'Convective velocity scale (m/s)',
#         'Thermal updraft (m/s)', 'Prob (updraft > threshold)')
# datetimes = get_saved_datetimes(
#     config.data_dir, config.datetime_format, '_orograph.npy')
# if not datetimes:
#     sys.exit('No updraft data found!')
# print('{0:d} instances of updrafts'.format(len(datetimes)), flush=True)

# n_cpu = min(len(datetimes), max_cores_for_plotting)
# start_time = initiate_timer('  plotting derived variables')
# with mp.Pool(n_cpu) as pool:
#     out = pool.map(lambda dtime: plot_interpolated_data(
#         dtime, fnames, lbls
#     ), datetimes)
# print_elapsed_time(start_time)

# n_cpu = min(len(wtk_indices), max_cores_for_plotting)
# start_time = initiate_timer('  plotting temporal pdfs')
# with mp.Pool(n_cpu) as pool:
#     out = pool.map(lambda ind: plot_pdfs_of_derived_variables(
#         ind, datetimes, fnames, lbls
#     ), wtk_indices)
# print_elapsed_time(start_time)


# TCmodel output
# datetimes = get_saved_datetimes(
#     config.data_dir, config.datetime_format, '_tracks.pkl')
# if not datetimes:
#     sys.exit('No TCmodel output found!')
# cases = ('orograph', 'prob')
# #cases = ('orograph', 'thermal', 'net', 'prob')
# print('{0:d} instances of TCmodel output'.format(len(datetimes)))

# n_cpu = min(len(datetimes), max_cores_for_plotting)
# start_time = initiate_timer('  plotting eagle tracks and presence maps')
# with mp.Pool(n_cpu) as pool:
#     out = pool.map(lambda dtime: plot_tcmodel_tracks_and_pmap(
#         dtime, cases
#     ), datetimes)
# print_elapsed_time(start_time)

# start_time = initiate_timer('  plotting migration energy potential')
# with mp.Pool(n_cpu) as pool:
#     oro_percs = pool.map(lambda dtime: plot_tcmodel_potential(
#         dtime, cases
#     ), datetimes)
# print_elapsed_time(start_time)

# summary pmaps
# if len(datetimes) < 4:
#     sys.exit('Not enough TCmodel output for computing summary stats!')
# start_time = initiate_timer('  plotting summarized presence maps')
# n_cpu = min(len(cases), max_cores_for_plotting)
# with mp.Pool(n_cpu) as pool:
#     oro_percs = pool.map(lambda lbl: plot_tcmodel_summary(
#         lbl, datetimes
#     ), cases)
# print_elapsed_time(start_time)

# start_time = initiate_timer(str(len(datetimes)) + ' WTK instances')
# wtk_cmp = 'coolwarm'  # 'bwr'
# for i in range(len(datetimes)):
#     makedir_if_not_exists(fig_dirnames[i])
#     for varname, varunits in wtk_dfs[i].columns:
#         vardata = wtk_dfs[i].loc[:, varname].to_numpy().flatten()
#         fig, ax = plt.subplots(figsize=config.fig_size)
#         plot_all_windfarms(ax, False)
#         vargrid = griddata(np.array([wtk_xgrid, wtk_ygrid]).T, vardata,
#                            (tr_xmesh, tr_ymesh), method=config.interp_type)
#         cm = ax.imshow(vargrid, extent=tr_extent, origin='lower', cmap=wtk_cmp)
#         cm = ax.scatter(wtk_xgrid, wtk_ygrid, c=vardata, cmap=wtk_cmp)
#         cb, lg = create_gis_axis(fig, ax, cm, 10)
#         plt.xlim(tr_extent[0:2])
#         plt.ylim(tr_extent[2:4])
#         #plot_wtk_locations(ax)
#         plt.title(datetimes[i].strftime('%I %p, %x'))
#         cb.set_label(varname + ' (' + varunits + ')')
#         save_fig(fig, fig_dirnames[i],
#                  datetime_ids[i] + '_' + varname + '.png')
# print_elapsed_time(start_time)


# start_time = initiate_timer(str(len(datetimes)) + ' updraft calculations')
# cbounds = [0, 2.0]
# updraft_cmp = get_transparent_cmap('brg_r', config.updraft_threshold, cbounds)
# updraft_cmp = 'bwr'
# for i in range(len(datetimes)):
#     orograph = load_data(data_dirnames[i], datetime_ids[i] + '_orograph.npy')
#     fig, ax = plt.subplots(figsize=config.fig_size)
#     plot_background_terrain(ax)
#     plot_all_windfarms(ax, False)
#     cm = ax.imshow(orograph, vmin=cbounds[0], vmax=cbounds[1],
#                    cmap=updraft_cmp, origin='lower', extent=tr_extent)
#     cb, lg = create_gis_axis(fig, ax, cm, 10)
#     #cb.set_ticks([0., config.updraft_threshold, cbounds[1]])
#     plt.title(datetimes[i].strftime('%I %p, %x'))
#     cb.set_label('Orographic updraft (m/s)')
#     save_fig(fig, fig_dirnames[i], datetime_ids[i] + '_orograph.png')

#     deardoff = load_data(data_dirnames[i], datetime_ids[i] + '_deardoff.npy')
#     fig, ax = plt.subplots(figsize=config.fig_size)
#     plot_background_terrain(ax)
#     plot_all_windfarms(ax, False)
#     cm = ax.imshow(deardoff, vmin=cbounds[0], vmax=cbounds[1],
#                    cmap=updraft_cmp, origin='lower', extent=tr_extent)
#     cb, lg = create_gis_axis(fig, ax, cm, 10)
#     #cb.set_ticks([0., config.updraft_threshold, cbounds[1]])
#     plt.title(datetimes[i].strftime('%I %p, %x'))
#     cb.set_label('Deardoff velocity (m/s)')
#     save_fig(fig, fig_dirnames[i], datetime_ids[i] + '_deardoff.png')

#     thermal = load_data(data_dirnames[i], datetime_ids[i] + '_thermal.npy')
#     fig, ax = plt.subplots(figsize=config.fig_size)
#     plot_background_terrain(ax)
#     plot_all_windfarms(ax, False)
#     cm = ax.imshow(thermal, vmin=cbounds[0], vmax=cbounds[1],
#                    cmap=updraft_cmp, origin='lower', extent=tr_extent)
#     cb, lg = create_gis_axis(fig, ax, cm, 10)
#     cb.set_ticks([0., config.updraft_threshold, cbounds[1]])
#     plt.title(datetimes[i].strftime('%I %p, %x'))
#     cb.set_label('Thermal updraft (m/s)')
#     save_fig(fig, fig_dirnames[i], datetime_ids[i] + '_thermal.png')
# print_elapsed_time(start_time)

# tc_run_ids = get_ids_of_saved_files(config.data_dir, '_tracks.pkl')
# cumulative_counts = np.zeros(tr_features.shape[1:])
# terrain_xygrid_trimmed = terrain_xygrid[:, config.pmap_trim_edges:-
#                                         config.pmap_trim_edges]
# if not tc_run_ids:
#     print('No track data found! \nRun run_tcmodel.py')
# else:
#     print('Plotting {0:d} model outputs .. '.format(len(tc_run_ids)),
#           end='', flush=True)
#     for i, tc_run_id in enumerate(tc_run_ids):
#         try:
#             _, year, month, day, hour = extract_from_wtk_run_id(tc_run_id)
#             timestamp = datetime(year, month, day, hour)
#             tracks = load_data(config.data_dir, tc_run_id +
#                                '_thermal_tracks.pkl')
#             potential = load_data(
#                 config.data_dir, tc_run_id + '_thermal_potential.npy')
#             counts = load_data(config.data_dir, tc_run_id +
#                                '_thermal_counts.npy')
#             cumulative_counts += counts
#         except:
#             print(tc_run_id, ' .. missing full output!')
#         else:
#             if not os.path.exists(config.fig_dir + tc_run_id + '_thermal_tracks.png'):
#                 fig, ax = plt.subplots(figsize=config.fig_size)
#                 cm = ax.pcolormesh(terrain_xygrid[0, :], terrain_xygrid[1, :],
#                                    tr_features[0, :, :], cmap='terrain')
#                 lwidth = 0.25 if len(tracks) > 251 else 0.95
#                 for track in tracks:
#                     ax.plot(terrain_xygrid[0, track[:, 1]],
#                             terrain_xygrid[1, track[:, 0]],
#                             '-r', linewidth=lwidth, alpha=0.4)
#                 plot_all_windfarms(ax, False)
#                 cb, lg = create_gis_axis(fig, ax, cm, 10)
#                 cb.set_label('Terrain elevation (Km)')
#                 plt.title(timestamp.strftime('%I %p, %d %b %Y'))
#                 save_fig(fig, config.fig_dir, tc_run_id +
#                          '_thermal_tracks.png')

#             if not os.path.exists(config.fig_dir + tc_run_id + '_thermal_potential.png'):
#                 lvls = np.linspace(0, np.amax(potential), 6)
#                 fig, ax = plt.subplots(figsize=config.fig_size)
#                 cm = ax.contourf(terrain_xygrid[0, :], terrain_xygrid[1, :],
#                                  potential, lvls, cmap='YlOrBr')
#                 plot_all_windfarms(ax, False)
#                 cb, lg = create_gis_axis(fig, ax, cm, 10)
#                 cb.set_label('Migration potential')
#                 cb.set_ticks(lvls)
#                 plt.title(timestamp.strftime('%I %p, %d %b %Y'))
#                 save_fig(fig, config.fig_dir, tc_run_id +
#                          '_thermal_potential.png')

#             if not os.path.exists(config.fig_dir + tc_run_id + '_thermal_presence.png'):
#                 presence_prob = compute_presence_probability(
#                     counts, config.kernel_radius, config.pmap_trim_edges)
#                 fig, ax = plt.subplots(figsize=config.fig_size)
#                 cm = ax.pcolormesh(terrain_xygrid_trimmed[0, :],
#                                    terrain_xygrid_trimmed[1, :],
#                                    presence_prob, cmap='Reds', vmin=0., vmax=1.)
#                 plot_all_windfarms(ax, False)
#                 cb, lg = create_gis_axis(fig, ax, cm, 10)
#                 cb.set_label('Relative eagle presence map')
#                 plt.title(timestamp.strftime('%I %p, %d %b %Y'))
#                 save_fig(fig, config.fig_dir, tc_run_id +
#                          '_thermal_presence.png')
#         plt.close('all')
#     print('done')

# if not tc_run_ids:
#     print('No track data found! \nRun run_tcmodel.py')
# else:
#     print('Plotting {0:d} model outputs .. '.format(len(tc_run_ids)),
#           end='', flush=True)
#     for i, tc_run_id in enumerate(tc_run_ids):
#         try:
#             _, year, month, day, hour = extract_from_wtk_run_id(tc_run_id)
#             timestamp = datetime(year, month, day, hour)
#             tracks = load_data(config.data_dir, tc_run_id + '_oro_tracks.pkl')
#             potential = load_data(
#                 config.data_dir, tc_run_id + '_oro_potential.npy')
#             counts = load_data(config.data_dir, tc_run_id + '_oro_counts.npy')
#             cumulative_counts += counts
#         except:
#             print(tc_run_id, ' .. missing full output!')
#         else:
#             if not os.path.exists(config.fig_dir + tc_run_id + 'oro_tracks.png'):
#                 fig, ax = plt.subplots(figsize=config.fig_size)
#                 cm = ax.pcolormesh(terrain_xygrid[0, :], terrain_xygrid[1, :],
#                                    tr_features[0, :, :], cmap='terrain')
#                 lwidth = 0.25 if len(tracks) > 251 else 0.95
#                 for track in tracks:
#                     ax.plot(terrain_xygrid[0, track[:, 1]],
#                             terrain_xygrid[1, track[:, 0]],
#                             '-r', linewidth=lwidth, alpha=0.4)
#                 plot_all_windfarms(ax, False)
#                 cb, lg = create_gis_axis(fig, ax, cm, 10)
#                 cb.set_label('Terrain elevation (Km)')
#                 plt.title(timestamp.strftime('%I %p, %d %b %Y'))
#                 save_fig(fig, config.fig_dir, tc_run_id + '_oro_tracks.png')

#             if not os.path.exists(config.fig_dir + tc_run_id + '_oro_potential.png'):
#                 lvls = np.linspace(0, np.amax(potential), 6)
#                 fig, ax = plt.subplots(figsize=config.fig_size)
#                 cm = ax.contourf(terrain_xygrid[0, :], terrain_xygrid[1, :],
#                                  potential, lvls, cmap='YlOrBr')
#                 plot_all_windfarms(ax, False)
#                 cb, lg = create_gis_axis(fig, ax, cm, 10)
#                 cb.set_label('Migration potential')
#                 cb.set_ticks(lvls)
#                 plt.title(timestamp.strftime('%I %p, %d %b %Y'))
#                 save_fig(fig, config.fig_dir, tc_run_id + '_oro_potential.png')

#             if not os.path.exists(config.fig_dir + tc_run_id + '_oro_presence.png'):
#                 presence_prob = compute_presence_probability(
#                     counts, config.kernel_radius, config.pmap_trim_edges)
#                 fig, ax = plt.subplots(figsize=config.fig_size)
#                 cm = ax.pcolormesh(terrain_xygrid_trimmed[0, :],
#                                    terrain_xygrid_trimmed[1, :],
#                                    presence_prob, cmap='Reds', vmin=0., vmax=1.)
#                 plot_all_windfarms(ax, False)
#                 cb, lg = create_gis_axis(fig, ax, cm, 10)
#                 cb.set_label('Relative eagle presence map')
#                 plt.title(timestamp.strftime('%I %p, %d %b %Y'))
#                 save_fig(fig, config.fig_dir, tc_run_id + '_oro_presence.png')
#         plt.close('all')
#     print('done')


# if not tc_run_ids:
#     print('No track data found! \nRun run_tcmodel.py')
# else:
#     print('Plotting {0:d} model outputs .. '.format(len(tc_run_ids)),
#           end='', flush=True)
#     for i, tc_run_id in enumerate(tc_run_ids):
#         try:
#             _, year, month, day, hour = extract_from_wtk_run_id(tc_run_id)
#             timestamp = datetime(year, month, day, hour)
#             tracks = load_data(config.data_dir, tc_run_id + '_both_tracks.pkl')
#             potential = load_data(
#                 config.data_dir, tc_run_id + '_both_potential.npy')
#             counts = load_data(config.data_dir, tc_run_id + '_both_counts.npy')
#             cumulative_counts += counts
#         except:
#             print(tc_run_id, ' .. missing full output!')
#         else:
#             if not os.path.exists(config.fig_dir + tc_run_id + '_tracks.png'):
#                 fig, ax = plt.subplots(figsize=config.fig_size)
#                 cm = ax.pcolormesh(terrain_xygrid[0, :], terrain_xygrid[1, :],
#                                    tr_features[0, :, :], cmap='terrain')
#                 lwidth = 0.25 if len(tracks) > 251 else 0.95
#                 for track in tracks:
#                     ax.plot(terrain_xygrid[0, track[:, 1]],
#                             terrain_xygrid[1, track[:, 0]],
#                             '-r', linewidth=lwidth, alpha=0.4)
#                 plot_all_windfarms(ax, False)
#                 cb, lg = create_gis_axis(fig, ax, cm, 10)
#                 cb.set_label('Terrain elevation (Km)')
#                 plt.title(timestamp.strftime('%I %p, %d %b %Y'))
#                 save_fig(fig, config.fig_dir, tc_run_id + '_both_tracks.png')

#             if not os.path.exists(config.fig_dir + tc_run_id + '_both_potential.png'):
#                 lvls = np.linspace(0, np.amax(potential), 6)
#                 fig, ax = plt.subplots(figsize=config.fig_size)
#                 cm = ax.contourf(terrain_xygrid[0, :], terrain_xygrid[1, :],
#                                  potential, lvls, cmap='YlOrBr')
#                 plot_all_windfarms(ax, False)
#                 cb, lg = create_gis_axis(fig, ax, cm, 10)
#                 cb.set_label('Migration potential')
#                 cb.set_ticks(lvls)
#                 plt.title(timestamp.strftime('%I %p, %d %b %Y'))
#                 save_fig(fig, config.fig_dir, tc_run_id +
#                          '_both_potential.png')

#             if not os.path.exists(config.fig_dir + tc_run_id + '_both_presence.png'):
#                 presence_prob = compute_presence_probability(
#                     counts, config.kernel_radius, config.pmap_trim_edges)
#                 fig, ax = plt.subplots(figsize=config.fig_size)
#                 cm = ax.pcolormesh(terrain_xygrid_trimmed[0, :],
#                                    terrain_xygrid_trimmed[1, :],
#                                    presence_prob, cmap='Reds', vmin=0., vmax=1.)
#                 plot_all_windfarms(ax, False)
#                 cb, lg = create_gis_axis(fig, ax, cm, 10)
#                 cb.set_label('Relative eagle presence map')
#                 plt.title(timestamp.strftime('%I %p, %d %b %Y'))
#                 save_fig(fig, config.fig_dir, tc_run_id + '_both_presence.png')
#         plt.close('all')
#     print('done')

#     # %% Plotting summary features
#     print('Plotting cumulative presence map .. ', end='', flush=True)
#     #cumulative_counts /= cumulative_counts.sum(axis=1)[:,None]
#     cumulative_counts[cumulative_counts <= 1] = 0.
#     #cumulative_counts[cumulative_counts > 2] = 1.
#     cumulative_prob = compute_presence_probability(
#         cumulative_counts, config.kernel_radius, config.pmap_trim_edges)
#     #cumulative_prob /= np.amax(cumulative_prob)
#     #cumulative_prob /= cumulative_prob.sum(axis=1)[:,None]
#     fig, ax = plt.subplots(figsize=config.fig_size)
#     cm = ax.pcolormesh(terrain_xygrid_trimmed[0, :], terrain_xygrid_trimmed[1, :],
#                        cumulative_prob, cmap='Reds', vmin=0.)
#     plot_all_windfarms(ax)
#     cb, lg = create_gis_axis(fig, ax, cm, 10)
#     cb.remove()
#     cb.set_label('Relative eagle presence map')
#     save_fig(fig, config.fig_dir, 'summary_presence.png')
#     save_data(config.data_dir, 'summary_presence.npy', cumulative_prob)
#     print('done')

# # %% Plotting summary stat
# print('Plotting summary stat of orographic updrafts .. ', end='', flush=True)
# updraft_stats = []
# label_stats = []
# updraft_stats.append(
#     np.sum(updrafts > config.threshold_updraft, axis=0, dtype=np.float32)
#     / len(wtk_run_ids))
# label_stats.append('Precentage time orographic updraft above threshold')

# updraft_stats.append(np.median(updrafts, axis=0))
# label_stats.append('Median (mps)')

# updraft_stats.append(np.std(updrafts, axis=0).astype(np.float32))
# label_stats.append('Standard devation')

# for istat, ilbl in zip(updraft_stats, label_stats):
#     fig, ax = plt.subplots(figsize=config.fig_size)
#     cm = ax.pcolormesh(terrain_xygrid[0, :], terrain_xygrid[1, :], istat,
#                        cmap='YlOrBr')
#     cb, lg = create_gis_axis(fig, ax, cm, 10)
#     cb.set_label(ilbl)
#     plot_all_windfarms(ax)
#     save_fig(fig, config.fig_dir, ('summary_orographic_' +
#                                    ilbl.replace(" ", "_")).lower() + '.png')
#     save_data(config.data_dir, ('summary_orographic_' +
#                                 ilbl.replace(" ", "_")).lower() + '.npy',
#               updraft_stats[2])
# print('done')

# # %% finish
# plt.close('all')
# print_elapsed_time(start_time)
