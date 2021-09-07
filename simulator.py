import pathos.multiprocessing as mp
from datetime import datetime
import numpy as np

from config import *
from tools.common import *
from tools.tracks import *


# Intialize
config = setup_config()


print('\n--- Simulating eagle tracks')
try:
    extent_km = load_data(config.terrain_data_dir, 'extent_km.txt')
    terrain_alt = load_data(config.terrain_data_dir, 'terrain_altitude.npy')
    res_km = config.terrain_res / 1000.
    M, N = terrain_alt.shape
    max_eagle_moves = 10000
except:
    print('No saved terrain data found!')
    exit('Run preprocess.py in {0:s} mode.'.format(config.mode))


# figure out what times to run the model for
if config.mode == 'seasonal':
    dtime_ids = get_dirs_in(config.mode_data_dir)
    if len(dtime_ids) == 0:
        exit('No saved data found! First run preprocess.py in seasonal mode')
    print('seasonal: {0:d} instance(s)'.format(len(dtime_ids)))
elif config.mode == 'snapshot':
    dtime = datetime(*config.wtk_snapshot_datetime)
    dtime_ids = [dtime.strftime(config.dtime_format)]
    print('snapshot: {0:s}'.format(dtime_ids[0]))
elif config.mode == 'predefined':
    dtime_ids = [get_predefined_mode_id(config.predefined_windspeed,
                                        config.predefined_winddirn)]
    print('predefined: {0:s}'.format(dtime_ids[0]))
else:
    print('Incorrect mode:', config.mode)
    print('Options: seasonal, snapshot, predefined')


# load orograph and thermal updrafts for all the times
orographs = []
oro_energys = []
dtime_ids_to_solve = []
for dtime_id in dtime_ids:
    data_dir = config.mode_data_dir + dtime_id + '/'
    if file_exists(data_dir, dtime_id + '_orograph.npy'):
        orographs.append(load_data(data_dir, dtime_id + '_orograph.npy'))
        run_id = dtime_id + '_' + config.track_direction
        if not file_exists(data_dir, run_id + '_tracks.pkl'):
            dtime_ids_to_solve.append(dtime_id)
    else:
        print(dtime_id + '_orograph.npy not found!')
        exit('Run preprocess.py in {0:s} mode'.format(config.mode))
if config.track_use_saved_data and len(dtime_ids) - len(dtime_ids_to_solve) > 0:
    print('Found saved tracks for {0:d} instance(s) (track_use_saved_data=True)'
          .format(len(dtime_ids) - len(dtime_ids_to_solve)))
    print('Set track_use_saved_data=False to ignore saved tracks')
else:
    dtime_ids_to_solve = dtime_ids


if dtime_ids_to_solve:
    print('Setting boundary vals for {0:s} movement...'
          .format(config.track_direction))
    bndry_nodes = get_boundary_nodes(config.track_direction, (M,N))
    bndry_energy = np.zeros((bndry_nodes.size))
    bndry_energy[bndry_nodes.size // 2:] = 1000.

    print('Getting starting locations...')
    sim_start_inds = get_starting_indices(
        config.track_count,
        config.track_start_region,
        config.track_start_type,
        config.terrain_width,
        config.terrain_res
    )

    print('Assembling the sparse linear system...')
    sls_row_inds, sls_col_inds, sls_facs = assemble_sparse_linear_system(M, N)

    print('Solving linear system and generating {0:d} tracks...'
          .format(config.track_count))
    n_cpu = min(sim_start_inds.shape[1], config.max_cores)
    for i, dtime_id in enumerate(dtime_ids_to_solve):
        start = initiate_timer('{1:3d}. Solve...{0:s}'.format(dtime_id, i + 1))
        new_energy = solve_sparse_linear_system(
            orographs[dtime_ids.index(dtime_id)],
            bndry_nodes,
            bndry_energy,
            sls_row_inds,
            sls_col_inds,
            sls_facs
        )
        print_elapsed_time(start)
        start = initiate_timer('{1:3d}. Tracks..{0:s}'.format(dtime_id, i + 1))
        with mp.Pool(n_cpu) as pool:
            new_tracks = pool.map(lambda k: generate_eagle_track(
                orographs[dtime_ids.index(dtime_id)],
                new_energy,
                sim_start_inds[:, k],
                config.track_dirn_restrict,
                config.track_stochastic_nu,
                max_eagle_moves
            ), range(sim_start_inds.shape[1]))
        print_elapsed_time(start)
        data_dir = config.mode_data_dir + dtime_id + '/'
        run_id = dtime_id + '_' + config.track_direction
        save_data(data_dir, run_id + '_orograph_energy.npy', new_energy)
        save_data(data_dir, run_id + '_tracks.pkl', new_tracks)
