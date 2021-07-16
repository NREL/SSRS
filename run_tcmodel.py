import pathos.multiprocessing as mp
import sys
from config import *
from tcfuncs.common import *
from tcfuncs.config_tools import initialize_code
from tcfuncs.tcmodel import *

# Intialize
#config = initialize_code(os.path.basename(__file__), eval(sys.argv[1]))
config = initialize_code('config.py', eval(sys.argv[1]))

# load updraft data
datetimes = get_saved_datetimes(
    config.data_dir, config.datetime_format, '_orograph.npy')
if not datetimes:
    raise Exception('\nRun compute_updrafts.py first!')
datetime_ids = [t.strftime(config.datetime_format) for t in datetimes]
data_dirnames = [config.data_dir + datetime_ids[i] +
                 '/' for i in range(len(datetimes))]
done_datetimes = get_saved_datetimes(
    config.data_dir, config.datetime_format, '_tracks.pkl')
todo_indices = [i for i, x in enumerate(datetimes) if x not in done_datetimes]
if not todo_indices:
    sys.exit('Requested TCModel output already exists!')


def run_tcmodel(dt_id, data_dir):
    run_id = dt_id + '_' + config.bndry_condition + '_'
    orograph = load_data(data_dir, dt_id + '_orograph.npy')
    # thermal = load_data(data_dir, dt_id + '_thermal.npy')
    # net = np.add(orograph, thermal)
    # prob = load_data(data_dir, dt_id + '_prob.npy')
    coeffs = [orograph]
    #   standardize_matrix(orograph), standardize_matrix(thermal),
    #   standardize_matrix(net), standardize_matrix(prob)]
    # usables = []
    # for updraft in coeffs[:3]:
    #     updraft_usable = np.copy(updraft)
    #     updraft_usable[updraft_usable < config.updraft_threshold] = 1e-10
    #     usables.append(updraft_usable)
    #     usables.append(standardize_matrix(updraft_usable))
    # coeffs += usables
    # print(len(for_tcmodel))
    # print(np.min(for_tcmodel[3]), np.max(for_tcmodel[3]))
    # cases = ('orograph', 'prob')
    # cases = ('orograph', 'thermal', 'net', 'prob',
    #         'uorograph', 'uthermal', 'unet')
    cases = ('orograph',)
    #         'Sorograph', 'Sthermal', 'Snet', 'Sprob')
    # 'Uorograph', 'USorograph', 'Uthermal', 'USthermal', 'Unet', 'USnet')
    tracks_all = []
    pot_all = []
    # _ = initiate_timer('\n')
    for j, updraft in enumerate(coeffs):
        tracks, potential = terrain_conductance_model(
            run_id + cases[j],
            updraft,
            config.bndry_condition,
            config.number_of_eagles,
            config.region_of_eagle_entry,
            config.type_of_eagle_entry,
            config.dirn_restrict,
            config.nu,
            config.terrain_res
        )
        #     tracks_all.append(tracks)
        #     pot_all.append(potential)
        # for j, _ in enumerate(coeffs):
        save_data(data_dir, run_id + cases[j] + '_tracks.pkl', tracks)
        save_data(data_dir, run_id + cases[j] + '_potential.npy', potential)


n_cpu = min(len(todo_indices), config.max_cores)
start_time = initiate_timer(
    '\n--- Running TCmodel: {0:d} requested, {1:d} exist, {2:d} cores\n'
        .format(len(datetimes), len(done_datetimes), n_cpu))
with mp.Pool(n_cpu) as pool:
    out = pool.map(lambda i: run_tcmodel(
        datetime_ids[i],
        data_dirnames[i]
    ), todo_indices)
print_elapsed_time(start_time)

# run the model
# for i, dt in enumerate(datetimes):
#     fext = dt.strftime(config.datetime_format)
#     run_id = fext + '_' + config.bndry_condition + '_'
#     if not os.path.exists(config.data_dir + run_id + '_orograph_tracks.pkl'):
#         orograph = load_data(config.data_dir, fext + '_orograph.npy')
#         deardoff = load_data(config.data_dir, fext + '_deardoff.npy')
#         blheight = load_data(config.data_dir, fext + '_blheight.npy')
#         thermal = compute_thermals_at_height(config.thermal_altitude,
#                                              deardoff, blheight)
#         save_data(config.data_dir, fext + '_thermal.npy', thermal)

#         updrafts_for_tcmodel = [orograph, thermal, np.add(orograph, thermal)]
#         usable_for_tcmodel = []
#         for updraft in updrafts_for_tcmodel:
#             updraft_usable = updraft.copy()
#             updraft_usable[updraft_usable < config.updraft_threshold] = 1e-04
#             usable_for_tcmodel.append(updraft_usable)
#         updrafts_for_tcmodel += usable_for_tcmodel

#         cases = ('orograph', 'thermal', 'combined',
#                 'uorograph', 'uthermal', 'ucombined')

#         # parallel
#         cpu_tcmodel = min(len(cases), config.max_cores)
#         with mp.Pool(cpu_tcmodel) as pool:
#             out_tcmodel = pool.map(lambda j: terrain_conductance_model_serial(
#                 run_id + cases[j],
#                 updrafts_for_tcmodel[j],
#                 config.bndry_condition,
#                 track_parameters
#             ), range(len(updrafts_for_tcmodel)))
#         for j, lbl in enumerate(cases):
#             save_data(config.data_dir, run_id + lbl + '_tracks.pkl',
#                       out_tcmodel[j][0])
#             save_data(config.data_dir, run_id + cases[j] + '_potential.npy',
#                       out_tcmodel[j][1])

# # serial
# for j, updraft in enumerate(updrafts_for_tcmodel):
#     tracks, potential = terrain_conductance_model(
#         run_id + cases[j],
#         updraft,
#         config.bndry_condition,
#         track_parameters,
#         config.max_cores
#     )
#     save_data(config.data_dir, run_id + cases[j] + '_tracks.pkl',
#               tracks)
#     save_data(config.data_dir, run_id + cases[j] + '_potential.npy',
#               potential.astype(np.float16))


# %% Parallel implemetation
# with multiprocessing.ThreadPool(config.max_cpu_usage) as pool:
#     tc_output = pool.map(
#         lambda i: terrain_conductance_model(
#             load_data(config.data_dir, i + '_orographic.npy'),
#             config.bndry_condition,
#             track_parameters,
#             grid_spacing,
#             config.max_cpu_usage_tracks),
#         wtk_run_ids)

# for i, out in enumerate(tc_output):
#     tracks, potential = out
#     tc_run_id = generate_tcmodel_run_id(wtk_run_ids[i], config.bndry_condition)
#     save_data(config.data_dir, tc_run_id + '_tracks.pkl', tracks)
#     save_data(config.data_dir, tc_run_id + '_potential.npy', potential)
#     count_mat = compute_count_matrix(grid_size, tracks)
#     save_data(config.data_dir, tc_run_id + '_counts.npy', count_mat)

# %%
