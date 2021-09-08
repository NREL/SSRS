import matplotlib.patches as mpatches
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde
from datetime import datetime
from matplotlib.colors import LogNorm

from config import *
from tools.common import *
from tools.tracks import *


# Intialize
config = setup_config()


#  plot setting
fig_hgt = 6.0  # increasing this increases plot size and fonts become smaller
fig_aspect = config.terrain_width[0] / config.terrain_width[1]
fig_size = (fig_hgt * fig_aspect, fig_hgt)
fig_dpi = 200  # increase this to get finer plots
maximum_dtimes_to_plot = 8  # to save computation time in seasonal mode
km_bar = min([1, 5, 10], key=lambda x: abs(x - config.terrain_width[0] // 4))
wfarm_mrkr_size = 4
wfarm_draw_box = False
wfarm_mrkr_style = ('1k', '2k', '3k', '4k', '+k', 'xk', '*k', '.k', 'ok')
# color-blind freindly colormaps: viridis, cividis, plasma, inferno, magma


# Load the terrain and wfarm data for plotting
try:
    print('\n--- Generating plots')
    wfarm_df = load_data(config.terrain_data_dir, 'wfarm_data.csv')
    extent_km = load_data(config.terrain_data_dir, 'extent_km.txt')
    extent_lonlat = load_data(config.terrain_data_dir, 'extent_lonlat.txt')
    terrain_alt = load_data(config.terrain_data_dir, 'terrain_altitude.npy')
    terrain_slope = load_data(config.terrain_data_dir, 'terrain_slope.npy')
    terrain_aspect = load_data(config.terrain_data_dir, 'terrain_aspect.npy')
    res_km = config.terrain_res / 1000.
    terrain_xgrid = np.arange(extent_km[0], extent_km[1], res_km)
    terrain_ygrid = np.arange(extent_km[2], extent_km[3], res_km)
    terrain_xmesh, terrain_ymesh = np.meshgrid(terrain_xgrid, terrain_ygrid)
    M, N = terrain_alt.shape
except:
    exit('No saved terrain data found! Run preprocess.py first.')


def plot_turbines(ax, set_lbl=True):  # function for plotting turbines
    for i, wf_name in enumerate(wfarm_df['p_name'].unique()):
        mrkr = wfarm_mrkr_style[i % len(wfarm_mrkr_style)]
        wf_xloc = wfarm_df.loc[wfarm_df['p_name'] == wf_name, 'xkm'].values
        wf_yloc = wfarm_df.loc[wfarm_df['p_name'] == wf_name, 'ykm'].values
        ax.plot(wf_xloc, wf_yloc, mrkr, markersize=wfarm_mrkr_size, alpha=0.75,
                label=wf_name if set_lbl else "")
        if wfarm_draw_box:
            width = max(wf_xloc) - min(wf_xloc) + 2
            height = max(wf_yloc) - min(wf_yloc) + 2
            rect = mpatches.Rectangle((min(wf_xloc) - 1, min(wf_yloc) - 1),
                                      width, height,
                                      linewidth=1, edgecolor='k',
                                      facecolor='none', zorder=20)
            ax.add_patch(rect)


def plot_background_terrain(ax):  # background terrain
    cm = ax.imshow(terrain_alt, alpha=0.75, cmap='Greys',
                   origin='lower', extent=extent_km)
    return cm


# terrain altitude
print('Plotting terrain altitude/slope/aspect...')
fig, ax = plt.subplots(figsize=fig_size)
cm = ax.imshow(terrain_alt, cmap='terrain', origin='lower',
               extent=extent_km)
plot_turbines(ax)
cb, lg = create_gis_axis(fig, ax, cm, km_bar)
cb.set_label('Altitude (km)')
save_fig(fig, config.terrain_fig_dir, 'terrain_altitude.png', fig_dpi)

# terrain slope
fig, ax = plt.subplots(figsize=fig_size)
cm = ax.imshow(terrain_slope * 180 / np.pi, vmin=0.,
               cmap='magma_r', origin='lower', extent=extent_km)
plot_turbines(ax)
cb, lg = create_gis_axis(fig, ax, cm, km_bar)
cb.set_label('Slope (Degrees)')
save_fig(fig, config.terrain_fig_dir, 'terrain_slope.png', fig_dpi)

# terrain aspect
fig, ax = plt.subplots(figsize=fig_size)
cm = ax.imshow(terrain_aspect, vmin=0., vmax=2 * np.pi, alpha=0.75,
               cmap='twilight', origin='lower', extent=extent_km)
plot_turbines(ax)
cb, lg = create_gis_axis(fig, ax, cm, km_bar)
cb.set_ticks(np.linspace(0, 2 * np.pi, 9))
cb.set_ticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'])
save_fig(fig, config.terrain_fig_dir, 'terrain_aspect.png', fig_dpi)


try:
    wtk_indices = load_data(config.terrain_data_dir, 'wtk_indices.txt',
                            dtype=int)
    wtk_xylocs = load_data(config.terrain_data_dir, 'wtk_xylocs.npy')
    wtk_xgrid = wtk_xylocs[0, :]
    wtk_ygrid = wtk_xylocs[1, :]
except:
    pass
else:
    # terrain with wtk plots
    fig, ax = plt.subplots(figsize=fig_size)
    cm = ax.imshow(terrain_alt, cmap='terrain', origin='lower',
                   extent=extent_km)
    ax.plot(wtk_xgrid, wtk_ygrid, '+k', markersize=3, label='WTK')
    # for i, v in enumerate(wtk_indices):
    #     ax.text(wtk_xgrid[i], wtk_ygrid[i] + 0.15, "%d" % (i),
    #             ha="center", fontsize=1)
    _, _ = create_gis_axis(fig, ax, None, km_bar)
    save_fig(fig, config.terrain_fig_dir, 'terrain_wtk_locations.png', fig_dpi)


# determine which time instances to plot for a given mode
if config.mode == 'seasonal':
    dtime_ids = get_dirs_in(config.mode_data_dir)
    if len(dtime_ids) > maximum_dtimes_to_plot:
        dtime_ids = dtime_ids[:maximum_dtimes_to_plot]
elif config.mode == 'snapshot':
    dtime = datetime(*config.wtk_snapshot_datetime)
    dtime_ids = [dtime.strftime(config.dtime_format)]
elif config.mode == 'predefined':
    dtime_ids = [get_predefined_mode_id(config.predefined_windspeed,
                                        config.predefined_winddirn)]
else:
    print('Incorrect mode:', config.mode)
    exit('Options: seasonal, snapshot, predefined')
data_dirs = [config.mode_data_dir + x + '/' for x in dtime_ids]
fig_dirs = [config.mode_fig_dir + x + '/' for x in dtime_ids]


def plot_atmospheric_data(df, fdir, fid, to_plot_turbines=True):
    """ plots atmospheric data from wtk in pandas dataframe format"""
    for _, (varname, varunits) in enumerate(df.columns):
        vardata = df.loc[:, varname].values.flatten()
        vargrid = griddata(np.array([wtk_xgrid, wtk_ygrid]).T, vardata,
                           (terrain_xmesh, terrain_ymesh),
                           method=config.wtk_interpolation_type)
        fig, ax = plt.subplots(figsize=fig_size)
        cm = ax.imshow(vargrid, cmap='viridis', origin='lower',
                       extent=extent_km, alpha=0.75)
        if to_plot_turbines:
            plot_turbines(ax)
        cb, _ = create_gis_axis(fig, ax, cm, km_bar)
        cb.set_label(varname + ' (' + varunits + ')')
        save_fig(fig, fdir, fid + '_' + varname + '.png', fig_dpi)


if config.mode != 'predefined':
    print('Plotting WTK atmospheric data...')
    for i, dtime_id in enumerate(dtime_ids):
        dtime = datetime.strptime(dtime_id, config.dtime_format)
        #print(dtime.strftime('%I %p, %d %b %Y'))
        makedir_if_not_exists(fig_dirs[i])
        fname = dtime_id + '_wtk.csv'
        if file_exists(data_dirs[i], fname):
            idf = load_data(data_dirs[i], fname, header=[0, 1])
            plot_atmospheric_data(idf, fig_dirs[i], dtime_id)
        else:
            exit(fname + ' not found! Run preprocess.py first.')


def plot_orographic_updraft(Amat, fdir, fid, to_plot_turbines=True):
    """ plots orographic updrafts """
    fig, ax = plt.subplots(figsize=fig_size)
    cm = ax.imshow(Amat, cmap='cividis', origin='lower',
                   extent=extent_km, vmin=0., vmax=1.0, alpha=0.75)
    if to_plot_turbines:
        plot_turbines(ax)
    cb, _ = create_gis_axis(fig, ax, cm, km_bar)
    cb.set_label('Orographic updraft (m/s)')
    save_fig(fig, fdir, fid + '_orograph.png', fig_dpi)


print('Plotting orographic updrafts...')
for i, dtime_id in enumerate(dtime_ids):
    fname = dtime_id + '_orograph.npy'
    if file_exists(data_dirs[i], fname):
        makedir_if_not_exists(fig_dirs[i])
        idata = load_data(data_dirs[i], fname)
        plot_orographic_updraft(idata, fig_dirs[i], dtime_id)
    else:
        exit(fname + ' not found! Run preprocess.py first.')


def plot_migration_energy(Amat, fdir, fid, to_plot_turbines=False):
    """ plots orographic migration potential energy """
    fig, ax = plt.subplots(figsize=fig_size)
    lvls = np.linspace(0, np.amax(Amat), 11)
    cm = ax.contourf(Amat, lvls, cmap='cividis', extent=extent_km)
    if to_plot_turbines:
        plot_turbines(ax)
    cb, _ = create_gis_axis(fig, ax, cm, km_bar)
    cb.set_label('Migration potential')
    save_fig(fig, fdir, fid + '_orograph_energy.png', fig_dpi)


def plot_simulated_tracks(sim_tracks, fdir, fid, to_plot_turbines=True):
    """ plots simulated tracks """
    fig, ax = plt.subplots(figsize=fig_size)
    _ = plot_background_terrain(ax)
    lwidth = 0.1 if len(sim_tracks) > 251 else 0.4
    for itrack in sim_tracks:
        ax.plot(terrain_xgrid[itrack[:, 1]], terrain_ygrid[itrack[:, 0]],
                '-r', linewidth=lwidth, alpha=0.5)
    if to_plot_turbines:
        plot_turbines(ax)
    _, _ = create_gis_axis(fig, ax)
    save_fig(fig, fdir, fid + '_tracks.png', fig_dpi)


def plot_presence_probability(prprob, fdir, fid, to_plot_turbines=True):
    """ plots presence probability """
    fig, ax = plt.subplots(figsize=fig_size)
    cm = ax.imshow(prprob, extent=extent_km, origin='lower',
                   cmap='Reds', alpha=0.75, norm=LogNorm(vmin=0.1, vmax=0.9))
    if to_plot_turbines:
        plot_turbines(ax)
    _, _ = create_gis_axis(fig, ax)
    save_fig(fig, fdir, fid + '_presence.png', fig_dpi)


print('Plotting simulated tracks/presence...', end="", flush=True)
nfound = 0
gradius = int(config.presence_smoothing_radius // config.terrain_res)
for i, dtime_id in enumerate(dtime_ids):
    run_id = dtime_id + '_' + config.track_direction
    fname = run_id + '_tracks.pkl'
    if file_exists(data_dirs[i], fname):
        #print(run_id)
        idata = load_data(data_dirs[i], run_id + '_orograph_energy.npy')
        plot_migration_energy(idata, fig_dirs[i], run_id)
        idata = load_data(data_dirs[i], fname)
        plot_simulated_tracks(idata, fig_dirs[i], run_id)
        presenceprob = compute_presence_probability(idata, (M, N), gradius)
        plot_presence_probability(presenceprob, fig_dirs[i], run_id)
        nfound += 1
print('{0:3d} found'.format(nfound))


def plot_potential_temperature(Amat, fdir, fid, to_plot_turbines=True):
    """ plots potential temperature """
    fig, ax = plt.subplots(figsize=fig_size)
    cm = ax.imshow(Amat, cmap='magma_r', origin='lower',
                   extent=extent_km, vmin=0., alpha=0.75)
    if to_plot_turbines:
        plot_turbines(ax)
    cb, _ = create_gis_axis(fig, ax, cm, km_bar)
    cb.set_label('Potential temperature (K)')
    save_fig(fig, fdir, fid + '_potential_temperature.png', fig_dpi)


def plot_thermal_updraft(Amat, fdir, fid, to_plot_turbines=True):
    """ plots thermal updrafts """
    fig, ax = plt.subplots(figsize=fig_size)
    cm = ax.imshow(Amat, cmap='magma_r', origin='lower',
                   extent=extent_km, vmin=0., alpha=0.75)
    if to_plot_turbines:
        plot_turbines(ax)
    cb, _ = create_gis_axis(fig, ax, cm, km_bar)
    cb.set_label('Thermal updraft (m/s)')
    save_fig(fig, fdir, fid + '_thermal.png', fig_dpi)


print('Plotting thermal updrafts...', end="", flush=True)
nfound = 0
for i, dtime_id in enumerate(dtime_ids):
    fname = dtime_id + '_thermal.npy'
    if file_exists(data_dirs[i], fname):
        idata = load_data(data_dirs[i], dtime_id +
                          '_potential_temperature.npy')
        plot_potential_temperature(idata, fig_dirs[i], dtime_id)
        idata = load_data(data_dirs[i], fname)
        plot_thermal_updraft(idata, fig_dirs[i], dtime_id)
        nfound += 1
print('{0:3d} found'.format(nfound))





######### Extra ##############
    # def plot_wtk_data(dtime):
    #     var_labels = ('Wind speed (m/s)',
    #                   'Wind direction (Deg)',
    #                   'Pressure (Pa)', 'Temperature (C)',
    #                   'Boundary layer height (m)',
    #                   'Surface heat flux (K.m/s)',
    #                   'Potential temperature (C)')
    #     cmap = 'viridis'  # 'coolwarm'  # 'bwr'
    #     dtime_id = dtime.strftime(config.dtime_format)
    #     data_dir = config.data_dir + dtime_id + '/'
    #     df = load_data(data_dir, dtime_id + '_wtk.csv', header=[0, 1])
    #     fig_dir = terrain_fig_dir + dtime_id + '/'
    #     makedir_if_not_exists(fig_dir)
    #     for i, (varname, varunits) in enumerate(df.columns):
    #         vardata = df.loc[:, varname].to_numpy().flatten()
    #         fig, ax = plt.subplots(figsize=config.fig_size)
    #         plot_turbines(ax, False)
    #         vargrid = griddata(np.array([wtk_xgrid, wtk_ygrid]).T, vardata,
    #                            (tr_xmesh, tr_ymesh), method=config.interp_type)
    #         if (varname == 'surface_heat_flux'):
    #             cmap = 'viridis'
    #         elif (varname == 'temperature_100m'):
    #             cmap = 'viridis'
    #         elif (varname == 'boundary_layer_height'):
    #             cmap = 'viridis'
    #         else:
    #             cmap = 'viridis'
    #         if varname == 'winddirection_100m':
    #             cm = ax.imshow(vargrid, extent=tr_extent,
    #                            origin='lower', cmap=cmap, alpha=0.8)
    #             # cm = ax.imshow(vargrid, extent=tr_extent,
    #             #                origin='lower', cmap='hsv', vmin=0., vmax=360.)
    #         else:
    #             cm = ax.imshow(vargrid, extent=tr_extent,
    #                            origin='lower', cmap=cmap, alpha=0.8)

    #         # cm = ax.scatter(wtk_xgrid, wtk_ygrid, s=5, c=vardata, cmap=cmap)
    #         cb, _ = create_gis_axis(fig, ax, cm)
    #         # if varname == 'winddirection_100m':
    #         #     cb.set_ticks(np.linspace(0, 360, 9))
    #         plt.xlim(tr_extent[0:2])
    #         plt.ylim(tr_extent[2:4])
    #         # plt.title(dtime.strftime('%I %p, %x'))
    #         # cb.set_label(varname + ' (' + varunits + ')')
    #         cb.set_label(var_labels[i])
    #         save_fig(fig, fig_dir, dtime_id + '_' + varname + '.png', 400)

    # def plot_interpolated_data(dtime, fnames, lbls):
    #     # cmap = get_transparent_cmap('brg_r', config.updraft_threshold, cbounds)
    #     cmap = 'viridis'
    #     dtime_id = dtime.strftime(config.dtime_format)
    #     data_dir = config.data_dir + dtime_id + '/'
    #     fig_dir = terrain_fig_dir + dtime_id + '/'
    #     makedir_if_not_exists(fig_dir)
    #     for fname, lbl in zip(fnames, lbls):
    #         vardata = load_data(data_dir, dtime_id + fname + '.npy')
    #         fig, ax = plt.subplots(figsize=config.fig_size)
    #         if fname == '_thermal':
    #             cm = ax.imshow(vardata, cmap=cmap,
    #                            origin='lower', extent=tr_extent)
    #         else:
    #             cm = ax.imshow(vardata, cmap='Oranges', origin='lower', extent=tr_extent,
    #                            vmin=0., vmax=1.)
    #         plot_turbines(ax, False)
    #         cb, _ = create_gis_axis(fig, ax, cm)
    #         # plt.title(dtime.strftime('%I %p, %x'))
    #         cb.set_label(lbl)
    #         save_fig(fig, fig_dir, dtime_id + fname + '.png', 400)
    #     # orograph = load_data(data_dir, dtime_id + '_orograph.npy')
    #     # thermal = load_data(data_dir, dtime_id + '_thermal.npy')
    #     # net = np.add(orograph, thermal)
    #     # fig, ax = plt.subplots(figsize=config.fig_size)
    #     # cm = ax.imshow(net, cmap=cmap, origin='lower', extent=tr_extent,
    #     #                vmin=0., vmax=1.)
    #     # cb, _ = create_gis_axis(fig, ax, cm)
    #     # #plt.title(dtime.strftime('%I %p, %x'))
    #     # cb.set_label('Orographic + thermal updraft (m/s)')
    #     # save_fig(fig, fig_dir, dtime_id + '_net.png', 400)

    # def plot_pdfs_of_wtk_variables(wtk_index, dtimes):
    #     fig_dir = terrain_fig_dir + str(wtk_index) + '/'
    #     makedir_if_not_exists(fig_dir)
    #     dtime_ids = [t.strftime(config.dtime_format) for t in dtimes]
    #     data_dirs = [config.data_dir + dtime_ids[i] +
    #                  '/' for i in range(len(dtimes))]
    #     wtk_dfs = [load_data(data_dirs[i], dtime_ids[i] + '_wtk.csv',
    #                          header=[0, 1]) for i in range(len(dtimes))]
    #     # wtk_smpls = np.empty((len(datetimes), wtk_dfs[0].columns.shape[0]))
    #     for j, (varname, varunits) in enumerate(wtk_dfs[0].columns):
    #         pointdata = np.array([])
    #         for i, df in enumerate(wtk_dfs):
    #             pointdata = np.append(pointdata, df.iloc[wtk_index, j])
    #         fig, ax = plt.subplots(figsize=(4, 3))
    #         xmin, xmid, xmax = get_min_mid_max(pointdata, 0.25)
    #         density = gaussian_kde(pointdata)
    #         xs = np.linspace(xmin, xmax, 100)
    #         ax.plot(xs, density(xs), '-b')
    #         create_pdf_axis(fig, ax, xmin, xmid, xmax)
    #         plt.xlabel(varname + ' (' + varunits + ')')
    #         save_fig(fig, fig_dir, str(wtk_index) + '_' + varname + '.png', 400)

    # def plot_pdfs_of_derived_variables(wtk_index, dtimes, fnames, lbls):
    #     fig_dir = terrain_fig_dir + str(wtk_index) + '/'
    #     makedir_if_not_exists(fig_dir)
    #     dtime_ids = [t.strftime(config.dtime_format) for t in dtimes]
    #     data_dirs = [config.data_dir + dtime_ids[i] +
    #                  '/' for i in range(len(dtimes))]
    #     idx = (np.abs(tr_xgrid - wtk_xgrid[wtk_index])).argmin()
    #     idy = (np.abs(tr_ygrid - wtk_ygrid[wtk_index])).argmin()
    #     # print('x-loc: ', tr_xgrid[idx], wtk_xgrid[loc_index])
    #     # print('y-loc: ', tr_ygrid[idy], wtk_ygrid[loc_index])
    #     for fname, lbl in zip(fnames, lbls):
    #         pointdata = np.array([])
    #         for i in range(len(dtimes)):
    #             vardata = load_data(data_dirs[i], dtime_ids[i] + fname + '.npy')
    #             pointdata = np.append(pointdata, vardata[idy, idx])
    #         fig, ax = plt.subplots(figsize=(4, 3))
    #         xmin, xmid, xmax = get_min_mid_max(pointdata, 0.25)
    #         density = gaussian_kde(pointdata)
    #         xs = np.linspace(xmin, xmax, 100)
    #         ax.plot(xs, density(xs), '-b')
    #         create_pdf_axis(fig, ax, xmin, xmid, xmax)
    #         save_fig(fig, fig_dir, str(wtk_index) + fname + '.png', 400)

    # def plot_tcmodel_tracks_and_pmap(dtime, cases):
    #     pmap_krad = 10  # higher -> smoother pmap
    #     padding = [x * 2. for x in [1, -1, 1, -1]]
    #     pmap_extent = [tr_extent[i] + padding[i] for i in range(len(padding))]
    #     pmap_extent_index = get_minmax_indices(tr_xgrid, tr_ygrid, pmap_extent)
    #     dtime_id = dtime.strftime(config.dtime_format)
    #     run_id = dtime_id + '_' + config.bndry_condition + '_'
    #     data_dir = config.data_dir + dtime_id + '/'
    #     fig_dir = terrain_fig_dir + dtime_id + '/'
    #     makedir_if_not_exists(fig_dir)
    #     for _, lbl in enumerate(cases):
    #         if file_exists(data_dir, run_id + lbl + '_tracks.pkl'):
    #             tracks = load_data(data_dir, run_id + lbl + '_tracks.pkl')
    #             fig, ax = plt.subplots(figsize=config.fig_size)
    #             cm = plot_background_terrain(ax)
    #             lwidth = 0.15 if len(tracks) > 251 else 0.45
    #             for kk, track in enumerate(tracks):
    #                 if kk == 0:
    #                     ax.plot(tr_xgrid[track[:, 1]], tr_ygrid[track[:, 0]],
    #                             '-r', linewidth=lwidth, alpha=0.9,
    #                             label='Simulated paths')
    #                 ax.plot(tr_xgrid[track[:, 1]], tr_ygrid[track[:, 0]],
    #                         '-r', linewidth=lwidth, alpha=0.4)
    #             plot_turbines(ax, False)
    #             cb, lg = create_gis_axis(fig, ax, cm)
    #             cb.remove()
    #             lg.legendHandles[0].set_linewidth(1.0)
    #             # plt.title(dtime.strftime('%I %p, %x'))
    #             save_fig(fig, fig_dir, run_id + lbl + '_tracks.png', 400)

    #             counts = compute_count_matrix(tr_gridsize, tracks)
    #             pmap = compute_presence(counts, pmap_extent_index, pmap_krad)
    #             pmap /= np.amax(pmap)
    #             fig, ax = plt.subplots(figsize=config.fig_size)
    #             cm = ax.imshow(pmap, extent=pmap_extent, origin='lower',
    #                            cmap='Reds', alpha=0.8, vmax=0.6)
    #             plot_turbines(ax, False)
    #             cb, _ = create_gis_axis(fig, ax, cm)
    #             cb.remove()
    #             # plt.title(dtime.strftime('%I %p, %x'))
    #             save_fig(fig, fig_dir, run_id + lbl + '_pmap.png', 400)

    #             pmap_wfarm_pad_wf = 1.
    #             pmap_wf_rad_wf = 4
    #             for i, wf in enumerate(config.wfarm_names):
    #                 tx = wfarm_df.loc[wfarm_df.p_name == wf, 'xkm'].to_numpy()
    #                 ty = wfarm_df.loc[wfarm_df.p_name == wf, 'ykm'].to_numpy()
    #                 pmap_extent_wf = [min(tx) - pmap_wfarm_pad_wf,
    #                                   max(tx) + pmap_wfarm_pad_wf,
    #                                   min(ty) - pmap_wfarm_pad_wf,
    #                                   max(ty) + pmap_wfarm_pad_wf]
    #                 pmap_extent_index_wf = get_minmax_indices(
    #                     tr_xgrid, tr_ygrid, pmap_extent_wf)

    #                 pmap_wf = compute_presence(
    #                     counts, pmap_extent_index_wf, pmap_wf_rad_wf)
    #                 pmap_wf /= np.amax(pmap_wf)
    #                 fig, ax = plt.subplots()
    #                 # print(pmap_extent, pmap_extent_wf)
    #                 cm = ax.imshow(pmap_wf, extent=pmap_extent_wf, origin='lower',
    #                                cmap='Reds', alpha=0.8, vmax=0.6)
    #                 l, = ax.plot(tx, ty, '1b', markersize=10, alpha=0.85)
    #                 cb, _ = create_gis_axis(fig, ax, cm, 1)
    #                 cb.remove()
    #                 ax.set_xlim([pmap_extent_wf[0], pmap_extent_wf[1]])
    #                 ax.set_ylim([pmap_extent_wf[2], pmap_extent_wf[3]])
    #                 plt.title(config.wfarm_labels[i])
    #                 save_fig(fig, fig_dir, lbl + '_' +
    #                          wf.replace(" ", "_") + '_pmap_mean.png', 400)

    # def plot_tcmodel_potential(dtime, cases):
    #     cmap = 'tab20b'
    #     dtime_id = dtime.strftime(config.dtime_format)
    #     data_dir = config.data_dir + dtime_id + '/'
    #     fig_dir = terrain_fig_dir + dtime_id + '/'
    #     makedir_if_not_exists(fig_dir)
    #     run_id = dtime_id + '_' + config.bndry_condition + '_'
    #     for j, lbl in enumerate(cases):
    #         if file_exists(data_dir, run_id + lbl + '_potential.npy'):
    #             potential = load_data(data_dir, run_id + lbl + '_potential.npy')
    #             fig, ax = plt.subplots(figsize=config.fig_size)
    #             # cm = ax.imshow(potential, cmap=cmap,
    #             #                origin='lower', extent=tr_extent)
    #             lvls = np.linspace(0, np.amax(potential), 11)
    #             cm = ax.contourf(potential, lvls, cmap='YlOrBr', extent=tr_extent)
    #             cs = ax.contour(potential, lvls[1:-1],
    #                             linewidths=0.1, extent=tr_extent)
    #             ax.clabel(cs, fmt='%d', colors='k', fontsize=10)
    #             cb, _ = create_gis_axis(fig, ax, cm)
    #             cb.remove()
    #             # plt.title(dtime.strftime('%I %p, %x'))
    #             save_fig(fig, fig_dir, run_id + lbl + '_potential.png', 400)

    # def plot_tcmodel_summary(lbl, dtimes):
    #     pmap_krad = 10
    #     padding = [x * 2. for x in [1, -1, 1, -1]]
    #     fig_dir = terrain_fig_dir + 'summary/'
    #     makedir_if_not_exists(fig_dir)
    #     dtime_ids = [t.strftime(config.dtime_format) for t in dtimes]
    #     data_dirs = [config.data_dir + dtime_ids[i] +
    #                  '/' for i in range(len(dtimes))]
    #     counts_list = []
    #     k = 0
    #     for i in range(len(dtimes)):
    #         thermals = load_data(data_dirs[i], dtime_ids[i] + '_thermal.npy')
    #         med_thermal = np.mean(thermals)
    #         run_id = dtime_ids[i] + '_' + config.bndry_condition + '_'
    #         if med_thermal < 0.5:
    #             k += 1
    #             if file_exists(data_dirs[i], run_id + lbl + '_tracks.pkl'):
    #                 tracks = load_data(data_dirs[i], run_id + lbl + '_tracks.pkl')
    #                 counts = compute_count_matrix(tr_gridsize, tracks)
    #                 # counts_list.append(np.divide(counts, np.amax(counts)))
    #                 counts_list.append(counts)
    #         else:
    #             print(lbl, i, '/', k + 1, ' : ', dtime_ids[i], ' : ', med_thermal)
    #     save_a = np.array([len(dtimes), k])
    #     save_data(config.data_dir, 'perc_usable_thermals.txt', save_a)
    #     # print('i am here')
    #     counts_list = np.asarray(counts_list)
    #     pmap_extent = [tr_extent[i] + padding[i] for i in range(len(padding))]
    #     pmap_extent_index = get_minmax_indices(tr_xgrid, tr_ygrid, pmap_extent)
    #     counts_mean = np.mean(counts_list, axis=0)
    #     pmap = compute_presence(counts_mean, pmap_extent_index, pmap_krad)
    #     pmap /= np.amax(pmap)
    #     fig, ax = plt.subplots(figsize=config.fig_size)
    #     cm = ax.imshow(pmap, extent=pmap_extent, origin='lower',
    #                    cmap='Reds', alpha=0.9, vmax=0.9)
    #     plot_turbines(ax, False)
    #     cb, _ = create_gis_axis(fig, ax, cm)
    #     cb.remove()
    #     # plt.title('Mean')
    #     save_data(config.data_dir, lbl + '_counts_mean.npy', counts_mean)
    #     save_fig(fig, fig_dir, lbl + '_pmap_mean.png', 400)

    #     counts_std = np.std(counts_list, axis=0)
    #     pmap = compute_presence(counts_std, pmap_extent_index, pmap_krad)
    #     pmap /= np.amax(pmap)
    #     fig, ax = plt.subplots(figsize=config.fig_size)
    #     cm = ax.imshow(pmap, extent=tr_extent, origin='lower',
    #                    cmap='Reds', alpha=0.9, vmax=0.9)
    #     plot_turbines(ax, False)
    #     cb, _ = create_gis_axis(fig, ax, cm)
    #     cb.remove()
    #     plt.title('Standard deviation')
    #     save_data(config.data_dir, lbl + '_counts_std.npy', counts_std)
    #     save_fig(fig, fig_dir, lbl + '_pmap_std.png', 400)

    #     print('i am here')
    #     pmap_wfarm_pad = 1.
    #     pmap_wf_rad = 4
    #     for i, wf in enumerate(config.wfarm_names):
    #         tx = wfarm_df.loc[wfarm_df.p_name == wf, 'xkm'].to_numpy()
    #         ty = wfarm_df.loc[wfarm_df.p_name == wf, 'ykm'].to_numpy()
    #         pmap_extent = [min(tx) - pmap_wfarm_pad, max(tx) + pmap_wfarm_pad,
    #                        min(ty) - pmap_wfarm_pad, max(ty) + pmap_wfarm_pad]
    #         pmap_extent_index = get_minmax_indices(tr_xgrid, tr_ygrid, pmap_extent)

    #         pmap = compute_presence(counts_mean, pmap_extent_index, pmap_wf_rad)
    #         pmap /= np.amax(pmap)
    #         fig, ax = plt.subplots()
    #         cm = ax.imshow(pmap, extent=pmap_extent, origin='lower',
    #                        cmap='Reds', alpha=0.9, vmax=0.9)
    #         l, = ax.plot(tx, ty, '1b', markersize=10)
    #         cb, _ = create_gis_axis(fig, ax, cm, 1)
    #         cb.remove()
    #         plt.title(config.wfarm_labels[i])
    #         # plt.title('Mean')
    #         save_fig(fig, fig_dir, lbl + '_' + wf + '_pmap_mean.png', 400)

    #         # pmap = compute_presence(counts_std, pmap_extent_index, pmap_wf_rad)
    #         # pmap /= np.amax(pmap)
    #         # fig, ax = plt.subplots()
    #         # cm = ax.imshow(pmap, extent=pmap_extent, origin='lower',
    #         #                cmap='Blues', alpha=0.8, vmax=0.6)
    #         # l, = ax.plot(tx, ty, '1k', markersize=5)
    #         # cb, _ = create_gis_axis(fig, ax, cm, 1)
    #         # cb.remove()
    #         # plt.title('Standard deviation')
    #         # save_fig(fig, fig_dir, lbl + '_' + wf + '_pmap_mean.png')

    # # wtk data
    # datetimes = get_saved_datetimes(
    #     config.data_dir, config.dtime_format, '_wtk.csv')
    # if not datetimes:
    #     raise Exception('\nRun extract_data.py first!')
    # print('{0:d} instances of WTK data'.format(len(datetimes)), flush=True)
    # start_time = initiate_timer('  plotting WTK variables')
    # for dtime in datetimes:
    #     plot_wtk_data(dtime)
    # print_elapsed_time(start_time)

    # # interpolated variables
    # fnames = ('_orograph',)
    # lbls = ('Orographic updraft (m/s)',)
    # datetimes = get_saved_datetimes(
    #     config.data_dir, config.dtime_format, '_orograph.npy')
    # if not datetimes:
    #     sys.exit('No updraft data found!')
    # print('{0:d} instances of updrafts'.format(len(datetimes)), flush=True)
    # n_cpu = min(len(datetimes), max_cores_for_plotting)
    # start_time = initiate_timer('  plotting derived variables')
    # for dtime in datetimes:
    #     plot_interpolated_data(dtime, fnames, lbls)
    # print_elapsed_time(start_time)

    # # TCmodel output
    # datetimes = get_saved_datetimes(
    #     config.data_dir, config.dtime_format, '_tracks.pkl')
    # if not datetimes:
    #     sys.exit('No TCmodel output found!')
    # cases = ('orograph',)
    # print('{0:d} instances of TCmodel output'.format(len(datetimes)))
    # n_cpu = min(len(datetimes), max_cores_for_plotting)

    # start_time = initiate_timer('  plotting eagle tracks and presence maps')
    # for dtime in datetimes:
    #     plot_tcmodel_tracks_and_pmap(dtime, cases)
    # print_elapsed_time(start_time)

    # start_time = initiate_timer('  plotting migration energy potential')
    # for dtime in datetimes:
    #     plot_tcmodel_potential(dtime, cases)
    # print_elapsed_time(start_time)

    # # n_cpu = min(len(datetimes), max_cores_for_plotting)
    # # start_time = initiate_timer('  plotting WTK variables')
    # # with mp.Pool(n_cpu) as pool:
    # #     out = pool.map(lambda dtime: plot_wtk_data(
    # #         dtime
    # #     ), datetimes)
    # # print_elapsed_time(start_time)

    # wtk_indices = (202, 437)
    # n_cpu = min(len(wtk_indices), max_cores_for_plotting)
    # start_time = initiate_timer('  plotting temporal pdfs')
    # with mp.Pool(n_cpu) as pool:
    #     out = pool.map(lambda ind: plot_pdfs_of_wtk_variables(
    #         ind, datetimes
    #     ), wtk_indices)
    # print_elapsed_time(start_time)

    # # Interpolated data
    # # fnames = ('_orograph', '_deardoff', '_thermal', '_prob')
    # # lbls = ('Orographic updraft (m/s)', 'Convective velocity scale (m/s)',
    # #         'Thermal updraft (m/s)', 'Prob (updraft > threshold)')
    # # datetimes = get_saved_datetimes(
    # #     config.data_dir, config.dtime_format, '_orograph.npy')
    # # if not datetimes:
    # #     sys.exit('No updraft data found!')
    # # print('{0:d} instances of updrafts'.format(len(datetimes)), flush=True)

    # # n_cpu = min(len(datetimes), max_cores_for_plotting)
    # # start_time = initiate_timer('  plotting derived variables')
    # # with mp.Pool(n_cpu) as pool:
    # #     out = pool.map(lambda dtime: plot_interpolated_data(
    # #         dtime, fnames, lbls
    # #     ), datetimes)
    # # print_elapsed_time(start_time)

    # # n_cpu = min(len(wtk_indices), max_cores_for_plotting)
    # # start_time = initiate_timer('  plotting temporal pdfs')
    # # with mp.Pool(n_cpu) as pool:
    # #     out = pool.map(lambda ind: plot_pdfs_of_derived_variables(
    # #         ind, datetimes, fnames, lbls
    # #     ), wtk_indices)
    # # print_elapsed_time(start_time)

    # # TCmodel output
    # # datetimes = get_saved_datetimes(
    # #     config.data_dir, config.dtime_format, '_tracks.pkl')
    # # if not datetimes:
    # #     sys.exit('No TCmodel output found!')
    # # cases = ('orograph', 'prob')
    # # #cases = ('orograph', 'thermal', 'net', 'prob')
    # # print('{0:d} instances of TCmodel output'.format(len(datetimes)))

    # # n_cpu = min(len(datetimes), max_cores_for_plotting)
    # # start_time = initiate_timer('  plotting eagle tracks and presence maps')
    # # with mp.Pool(n_cpu) as pool:
    # #     out = pool.map(lambda dtime: plot_tcmodel_tracks_and_pmap(
    # #         dtime, cases
    # #     ), datetimes)
    # # print_elapsed_time(start_time)

    # # start_time = initiate_timer('  plotting migration energy potential')
    # # with mp.Pool(n_cpu) as pool:
    # #     oro_percs = pool.map(lambda dtime: plot_tcmodel_potential(
    # #         dtime, cases
    # #     ), datetimes)
    # # print_elapsed_time(start_time)

    # # summary pmaps
    # # if len(datetimes) < 4:
    # #     sys.exit('Not enough TCmodel output for computing summary stats!')
    # # start_time = initiate_timer('  plotting summarized presence maps')
    # # n_cpu = min(len(cases), max_cores_for_plotting)
    # # with mp.Pool(n_cpu) as pool:
    # #     oro_percs = pool.map(lambda lbl: plot_tcmodel_summary(
    # #         lbl, datetimes
    # #     ), cases)
    # # print_elapsed_time(start_time)

    # # start_time = initiate_timer(str(len(datetimes)) + ' WTK instances')
    # # wtk_cmp = 'coolwarm'  # 'bwr'
    # # for i in range(len(datetimes)):
    # #     makedir_if_not_exists(fig_dirnames[i])
    # #     for varname, varunits in wtk_dfs[i].columns:
    # #         vardata = wtk_dfs[i].loc[:, varname].to_numpy().flatten()
    # #         fig, ax = plt.subplots(figsize=config.fig_size)
    # #         plot_turbines(ax, False)
    # #         vargrid = griddata(np.array([wtk_xgrid, wtk_ygrid]).T, vardata,
    # #                            (tr_xmesh, tr_ymesh), method=config.interp_type)
    # #         cm = ax.imshow(vargrid, extent=tr_extent, origin='lower', cmap=wtk_cmp)
    # #         cm = ax.scatter(wtk_xgrid, wtk_ygrid, c=vardata, cmap=wtk_cmp)
    # #         cb, lg = create_gis_axis(fig, ax, cm, 10)
    # #         plt.xlim(tr_extent[0:2])
    # #         plt.ylim(tr_extent[2:4])
    # #         #plot_wtk_locations(ax)
    # #         plt.title(datetimes[i].strftime('%I %p, %x'))
    # #         cb.set_label(varname + ' (' + varunits + ')')
    # #         save_fig(fig, fig_dirnames[i],
    # #                  datetime_ids[i] + '_' + varname + '.png')
    # # print_elapsed_time(start_time)

    # # start_time = initiate_timer(str(len(datetimes)) + ' updraft calculations')
    # # cbounds = [0, 2.0]
    # # updraft_cmp = get_transparent_cmap('brg_r', config.updraft_threshold, cbounds)
    # # updraft_cmp = 'bwr'
    # # for i in range(len(datetimes)):
    # #     orograph = load_data(data_dirnames[i], datetime_ids[i] + '_orograph.npy')
    # #     fig, ax = plt.subplots(figsize=config.fig_size)
    # #     plot_background_terrain(ax)
    # #     plot_turbines(ax, False)
    # #     cm = ax.imshow(orograph, vmin=cbounds[0], vmax=cbounds[1],
    # #                    cmap=updraft_cmp, origin='lower', extent=tr_extent)
    # #     cb, lg = create_gis_axis(fig, ax, cm, 10)
    # #     #cb.set_ticks([0., config.updraft_threshold, cbounds[1]])
    # #     plt.title(datetimes[i].strftime('%I %p, %x'))
    # #     cb.set_label('Orographic updraft (m/s)')
    # #     save_fig(fig, fig_dirnames[i], datetime_ids[i] + '_orograph.png')

    # #     deardoff = load_data(data_dirnames[i], datetime_ids[i] + '_deardoff.npy')
    # #     fig, ax = plt.subplots(figsize=config.fig_size)
    # #     plot_background_terrain(ax)
    # #     plot_turbines(ax, False)
    # #     cm = ax.imshow(deardoff, vmin=cbounds[0], vmax=cbounds[1],
    # #                    cmap=updraft_cmp, origin='lower', extent=tr_extent)
    # #     cb, lg = create_gis_axis(fig, ax, cm, 10)
    # #     #cb.set_ticks([0., config.updraft_threshold, cbounds[1]])
    # #     plt.title(datetimes[i].strftime('%I %p, %x'))
    # #     cb.set_label('Deardoff velocity (m/s)')
    # #     save_fig(fig, fig_dirnames[i], datetime_ids[i] + '_deardoff.png')

    # #     thermal = load_data(data_dirnames[i], datetime_ids[i] + '_thermal.npy')
    # #     fig, ax = plt.subplots(figsize=config.fig_size)
    # #     plot_background_terrain(ax)
    # #     plot_turbines(ax, False)
    # #     cm = ax.imshow(thermal, vmin=cbounds[0], vmax=cbounds[1],
    # #                    cmap=updraft_cmp, origin='lower', extent=tr_extent)
    # #     cb, lg = create_gis_axis(fig, ax, cm, 10)
    # #     cb.set_ticks([0., config.updraft_threshold, cbounds[1]])
    # #     plt.title(datetimes[i].strftime('%I %p, %x'))
    # #     cb.set_label('Thermal updraft (m/s)')
    # #     save_fig(fig, fig_dirnames[i], datetime_ids[i] + '_thermal.png')
    # # print_elapsed_time(start_time)

    # # tc_run_ids = get_ids_of_saved_files(config.data_dir, '_tracks.pkl')
    # # cumulative_counts = np.zeros(tr_features.shape[1:])
    # # terrain_xygrid_trimmed = terrain_xygrid[:, config.pmap_trim_edges:-
    # #                                         config.pmap_trim_edges]
    # # if not tc_run_ids:
    # #     print('No track data found! \nRun run_tcmodel.py')
    # # else:
    # #     print('Plotting {0:d} model outputs .. '.format(len(tc_run_ids)),
    # #           end='', flush=True)
    # #     for i, tc_run_id in enumerate(tc_run_ids):
    # #         try:
    # #             _, year, month, day, hour = extract_from_wtk_run_id(tc_run_id)
    # #             timestamp = datetime(year, month, day, hour)
    # #             tracks = load_data(config.data_dir, tc_run_id +
    # #                                '_thermal_tracks.pkl')
    # #             potential = load_data(
    # #                 config.data_dir, tc_run_id + '_thermal_potential.npy')
    # #             counts = load_data(config.data_dir, tc_run_id +
    # #                                '_thermal_counts.npy')
    # #             cumulative_counts += counts
    # #         except:
    # #             print(tc_run_id, ' .. missing full output!')
    # #         else:
    # #             if not os.path.exists(terrain_fig_dir + tc_run_id + '_thermal_tracks.png'):
    # #                 fig, ax = plt.subplots(figsize=config.fig_size)
    # #                 cm = ax.pcolormesh(terrain_xygrid[0, :], terrain_xygrid[1, :],
    # #                                    tr_features[0, :, :], cmap='terrain')
    # #                 lwidth = 0.25 if len(tracks) > 251 else 0.95
    # #                 for track in tracks:
    # #                     ax.plot(terrain_xygrid[0, track[:, 1]],
    # #                             terrain_xygrid[1, track[:, 0]],
    # #                             '-r', linewidth=lwidth, alpha=0.4)
    # #                 plot_turbines(ax, False)
    # #                 cb, lg = create_gis_axis(fig, ax, cm, 10)
    # #                 cb.set_label('Terrain elevation (Km)')
    # #                 plt.title(timestamp.strftime('%I %p, %d %b %Y'))
    # #                 save_fig(fig, terrain_fig_dir, tc_run_id +
    # #                          '_thermal_tracks.png')

    # #             if not os.path.exists(terrain_fig_dir + tc_run_id + '_thermal_potential.png'):
    # #                 lvls = np.linspace(0, np.amax(potential), 6)
    # #                 fig, ax = plt.subplots(figsize=config.fig_size)
    # #                 cm = ax.contourf(terrain_xygrid[0, :], terrain_xygrid[1, :],
    # #                                  potential, lvls, cmap='YlOrBr')
    # #                 plot_turbines(ax, False)
    # #                 cb, lg = create_gis_axis(fig, ax, cm, 10)
    # #                 cb.set_label('Migration potential')
    # #                 cb.set_ticks(lvls)
    # #                 plt.title(timestamp.strftime('%I %p, %d %b %Y'))
    # #                 save_fig(fig, terrain_fig_dir, tc_run_id +
    # #                          '_thermal_potential.png')

    # #             if not os.path.exists(terrain_fig_dir + tc_run_id + '_thermal_presence.png'):
    # #                 presence_prob = compute_presence_probability(
    # #                     counts, config.kernel_radius, config.pmap_trim_edges)
    # #                 fig, ax = plt.subplots(figsize=config.fig_size)
    # #                 cm = ax.pcolormesh(terrain_xygrid_trimmed[0, :],
    # #                                    terrain_xygrid_trimmed[1, :],
    # #                                    presence_prob, cmap='Reds', vmin=0., vmax=1.)
    # #                 plot_turbines(ax, False)
    # #                 cb, lg = create_gis_axis(fig, ax, cm, 10)
    # #                 cb.set_label('Relative eagle presence map')
    # #                 plt.title(timestamp.strftime('%I %p, %d %b %Y'))
    # #                 save_fig(fig, terrain_fig_dir, tc_run_id +
    # #                          '_thermal_presence.png')
    # #         plt.close('all')
    # #     print('done')

    # # if not tc_run_ids:
    # #     print('No track data found! \nRun run_tcmodel.py')
    # # else:
    # #     print('Plotting {0:d} model outputs .. '.format(len(tc_run_ids)),
    # #           end='', flush=True)
    # #     for i, tc_run_id in enumerate(tc_run_ids):
    # #         try:
    # #             _, year, month, day, hour = extract_from_wtk_run_id(tc_run_id)
    # #             timestamp = datetime(year, month, day, hour)
    # #             tracks = load_data(config.data_dir, tc_run_id + '_oro_tracks.pkl')
    # #             potential = load_data(
    # #                 config.data_dir, tc_run_id + '_oro_potential.npy')
    # #             counts = load_data(config.data_dir, tc_run_id + '_oro_counts.npy')
    # #             cumulative_counts += counts
    # #         except:
    # #             print(tc_run_id, ' .. missing full output!')
    # #         else:
    # #             if not os.path.exists(terrain_fig_dir + tc_run_id + 'oro_tracks.png'):
    # #                 fig, ax = plt.subplots(figsize=config.fig_size)
    # #                 cm = ax.pcolormesh(terrain_xygrid[0, :], terrain_xygrid[1, :],
    # #                                    tr_features[0, :, :], cmap='terrain')
    # #                 lwidth = 0.25 if len(tracks) > 251 else 0.95
    # #                 for track in tracks:
    # #                     ax.plot(terrain_xygrid[0, track[:, 1]],
    # #                             terrain_xygrid[1, track[:, 0]],
    # #                             '-r', linewidth=lwidth, alpha=0.4)
    # #                 plot_turbines(ax, False)
    # #                 cb, lg = create_gis_axis(fig, ax, cm, 10)
    # #                 cb.set_label('Terrain elevation (Km)')
    # #                 plt.title(timestamp.strftime('%I %p, %d %b %Y'))
    # #                 save_fig(fig, terrain_fig_dir, tc_run_id + '_oro_tracks.png')

    # #             if not os.path.exists(terrain_fig_dir + tc_run_id + '_oro_potential.png'):
    # #                 lvls = np.linspace(0, np.amax(potential), 6)
    # #                 fig, ax = plt.subplots(figsize=config.fig_size)
    # #                 cm = ax.contourf(terrain_xygrid[0, :], terrain_xygrid[1, :],
    # #                                  potential, lvls, cmap='YlOrBr')
    # #                 plot_turbines(ax, False)
    # #                 cb, lg = create_gis_axis(fig, ax, cm, 10)
    # #                 cb.set_label('Migration potential')
    # #                 cb.set_ticks(lvls)
    # #                 plt.title(timestamp.strftime('%I %p, %d %b %Y'))
    # #                 save_fig(fig, terrain_fig_dir, tc_run_id + '_oro_potential.png')

    # #             if not os.path.exists(terrain_fig_dir + tc_run_id + '_oro_presence.png'):
    # #                 presence_prob = compute_presence_probability(
    # #                     counts, config.kernel_radius, config.pmap_trim_edges)
    # #                 fig, ax = plt.subplots(figsize=config.fig_size)
    # #                 cm = ax.pcolormesh(terrain_xygrid_trimmed[0, :],
    # #                                    terrain_xygrid_trimmed[1, :],
    # #                                    presence_prob, cmap='Reds', vmin=0., vmax=1.)
    # #                 plot_turbines(ax, False)
    # #                 cb, lg = create_gis_axis(fig, ax, cm, 10)
    # #                 cb.set_label('Relative eagle presence map')
    # #                 plt.title(timestamp.strftime('%I %p, %d %b %Y'))
    # #                 save_fig(fig, terrain_fig_dir, tc_run_id + '_oro_presence.png')
    # #         plt.close('all')
    # #     print('done')

    # # if not tc_run_ids:
    # #     print('No track data found! \nRun run_tcmodel.py')
    # # else:
    # #     print('Plotting {0:d} model outputs .. '.format(len(tc_run_ids)),
    # #           end='', flush=True)
    # #     for i, tc_run_id in enumerate(tc_run_ids):
    # #         try:
    # #             _, year, month, day, hour = extract_from_wtk_run_id(tc_run_id)
    # #             timestamp = datetime(year, month, day, hour)
    # #             tracks = load_data(config.data_dir, tc_run_id + '_both_tracks.pkl')
    # #             potential = load_data(
    # #                 config.data_dir, tc_run_id + '_both_potential.npy')
    # #             counts = load_data(config.data_dir, tc_run_id + '_both_counts.npy')
    # #             cumulative_counts += counts
    # #         except:
    # #             print(tc_run_id, ' .. missing full output!')
    # #         else:
    # #             if not os.path.exists(terrain_fig_dir + tc_run_id + '_tracks.png'):
    # #                 fig, ax = plt.subplots(figsize=config.fig_size)
    # #                 cm = ax.pcolormesh(terrain_xygrid[0, :], terrain_xygrid[1, :],
    # #                                    tr_features[0, :, :], cmap='terrain')
    # #                 lwidth = 0.25 if len(tracks) > 251 else 0.95
    # #                 for track in tracks:
    # #                     ax.plot(terrain_xygrid[0, track[:, 1]],
    # #                             terrain_xygrid[1, track[:, 0]],
    # #                             '-r', linewidth=lwidth, alpha=0.4)
    # #                 plot_turbines(ax, False)
    # #                 cb, lg = create_gis_axis(fig, ax, cm, 10)
    # #                 cb.set_label('Terrain elevation (Km)')
    # #                 plt.title(timestamp.strftime('%I %p, %d %b %Y'))
    # #                 save_fig(fig, terrain_fig_dir, tc_run_id + '_both_tracks.png')

    # #             if not os.path.exists(terrain_fig_dir + tc_run_id + '_both_potential.png'):
    # #                 lvls = np.linspace(0, np.amax(potential), 6)
    # #                 fig, ax = plt.subplots(figsize=config.fig_size)
    # #                 cm = ax.contourf(terrain_xygrid[0, :], terrain_xygrid[1, :],
    # #                                  potential, lvls, cmap='YlOrBr')
    # #                 plot_turbines(ax, False)
    # #                 cb, lg = create_gis_axis(fig, ax, cm, 10)
    # #                 cb.set_label('Migration potential')
    # #                 cb.set_ticks(lvls)
    # #                 plt.title(timestamp.strftime('%I %p, %d %b %Y'))
    # #                 save_fig(fig, terrain_fig_dir, tc_run_id +
    # #                          '_both_potential.png')

    # #             if not os.path.exists(terrain_fig_dir + tc_run_id + '_both_presence.png'):
    # #                 presence_prob = compute_presence_probability(
    # #                     counts, config.kernel_radius, config.pmap_trim_edges)
    # #                 fig, ax = plt.subplots(figsize=config.fig_size)
    # #                 cm = ax.pcolormesh(terrain_xygrid_trimmed[0, :],
    # #                                    terrain_xygrid_trimmed[1, :],
    # #                                    presence_prob, cmap='Reds', vmin=0., vmax=1.)
    # #                 plot_turbines(ax, False)
    # #                 cb, lg = create_gis_axis(fig, ax, cm, 10)
    # #                 cb.set_label('Relative eagle presence map')
    # #                 plt.title(timestamp.strftime('%I %p, %d %b %Y'))
    # #                 save_fig(fig, terrain_fig_dir, tc_run_id + '_both_presence.png')
    # #         plt.close('all')
    # #     print('done')

    # #     # %% Plotting summary features
    # #     print('Plotting cumulative presence map .. ', end='', flush=True)
    # #     #cumulative_counts /= cumulative_counts.sum(axis=1)[:,None]
    # #     cumulative_counts[cumulative_counts <= 1] = 0.
    # #     #cumulative_counts[cumulative_counts > 2] = 1.
    # #     cumulative_prob = compute_presence_probability(
    # #         cumulative_counts, config.kernel_radius, config.pmap_trim_edges)
    # #     #cumulative_prob /= np.amax(cumulative_prob)
    # #     #cumulative_prob /= cumulative_prob.sum(axis=1)[:,None]
    # #     fig, ax = plt.subplots(figsize=config.fig_size)
    # #     cm = ax.pcolormesh(terrain_xygrid_trimmed[0, :], terrain_xygrid_trimmed[1, :],
    # #                        cumulative_prob, cmap='Reds', vmin=0.)
    # #     plot_turbines(ax)
    # #     cb, lg = create_gis_axis(fig, ax, cm, 10)
    # #     cb.remove()
    # #     cb.set_label('Relative eagle presence map')
    # #     save_fig(fig, terrain_fig_dir, 'summary_presence.png')
    # #     save_data(config.data_dir, 'summary_presence.npy', cumulative_prob)
    # #     print('done')

    # # # %% Plotting summary stat
    # # print('Plotting summary stat of orographic updrafts .. ', end='', flush=True)
    # # updraft_stats = []
    # # label_stats = []
    # # updraft_stats.append(
    # #     np.sum(updrafts > config.threshold_updraft, axis=0, dtype=np.float32)
    # #     / len(wtk_run_ids))
    # # label_stats.append('Precentage time orographic updraft above threshold')

    # # updraft_stats.append(np.median(updrafts, axis=0))
    # # label_stats.append('Median (mps)')

    # # updraft_stats.append(np.std(updrafts, axis=0).astype(np.float32))
    # # label_stats.append('Standard devation')

    # # for istat, ilbl in zip(updraft_stats, label_stats):
    # #     fig, ax = plt.subplots(figsize=config.fig_size)
    # #     cm = ax.pcolormesh(terrain_xygrid[0, :], terrain_xygrid[1, :], istat,
    # #                        cmap='YlOrBr')
    # #     cb, lg = create_gis_axis(fig, ax, cm, 10)
    # #     cb.set_label(ilbl)
    # #     plot_turbines(ax)
    # #     save_fig(fig, terrain_fig_dir, ('summary_orographic_' +
    # #                                    ilbl.replace(" ", "_")).lower() + '.png')
    # #     save_data(config.data_dir, ('summary_orographic_' +
    # #                                 ilbl.replace(" ", "_")).lower() + '.npy',
    # #               updraft_stats[2])
    # # print('done')

    # # # %% finish
    # # plt.close('all')
    # # print_elapsed_time(start_time)
