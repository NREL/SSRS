import os
import pickle
import time
import errno
import json
import re
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Tuple
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from copy import deepcopy


def parse_config_from_args(args: [str], default_config: {}) -> {}:
    """
    Makes a configuration map given a list of (command line) override args 
    and a default configuration
    """
    config = deepcopy(default_config)
    arg = ''.join(args).strip()
    # print('c1: ', arg)
    if not arg.startswith('{'):  # add enclosing brackets if they are missing
        arg = '{' + arg + '}'
    # print('c2: ', arg)
    # convert bare true, false, and null's to lowercase
    arg = re.sub(r'(?i)(?<=[\t :{},])["\']?(true|false|null)["\']?(?=[\t :{},])',
                 lambda match: match.group(0).lower(), arg)
    # print('c4: ', arg)
    # replace bare or single quoted strings with quoted ones
    arg = re.sub(
        r'(?<=[\t :{},])["\']?(((?<=")((?!(?<!\\)").)*(?="))|(?<=\')((?!\').)*(?=\')|(?!(true|false|null).)(['
        r'a-zA-Z_][a-zA-Z_0-9]*))["\']?(?=[\t :{},])',
        r'"\1"',
        arg)
    # print('c5: ', arg)
    overrides = json.loads(arg)
    config = merge_configs(config, overrides)
    return config


def merge_configs(target, overrides):
    """ Merging two dictionaries/configurations """
    if isinstance(overrides, dict):
        for key, value in overrides.items():
            if key in target:
                target[key] = merge_configs(target[key], value)
            else:
                target[key] = value
        return target
    else:
        return overrides


def create_gis_axis(
        cur_fig,
        cur_ax,
        cur_cm=None,
        b_kms=10,
):
    """ Creates GIS axes """

    plt.tick_params(axis='both', which='both', bottom=False, top=False,
                    labelbottom=False, right=False, left=False, labelleft=False)
    b_txt = str(int(b_kms)) + ' km'
    my_arrow = AnchoredSizeBar(cur_ax.transData, b_kms, b_txt, 3,
                               pad=0.1, size_vertical=0.1, frameon=False)
    cur_ax.add_artist(my_arrow)
    arrowpr = dict(fc="k", ec="k", alpha=0.9, lw=2.1,
                   arrowstyle="<-,head_length=1.0")
    cur_ax.annotate('N', xy=(0.03, 0.925), xycoords='axes fraction',
                    xytext=(0.03, 0.99), textcoords='axes fraction',
                    arrowprops=arrowpr,
                    bbox=dict(pad=-4, facecolor="none", edgecolor="none"),
                    ha='center', va='top', alpha=0.9)
    if cur_cm:
        cur_cb = cur_fig.colorbar(cur_cm, ax=cur_ax, pad=0.01,
                                  shrink=0.8, aspect=40)
        cur_cb.outline.set_visible(False)
        cur_cb.ax.tick_params(size=0)
    else:
        cur_cb = None
    _, labels = cur_ax.get_legend_handles_labels()
    if labels:
        w = cur_fig.get_size_inches()[0]
        cur_lg = cur_ax.legend(bbox_to_anchor=(0, 1.005), ncol=int(w // 2),
                               loc='lower left', markerscale=2,
                               columnspacing=1.0, handletextpad=0.0,
                               borderaxespad=0., fontsize='small')
    else:
        cur_lg = None
    cur_ax.set_aspect('equal', adjustable='box')
    return cur_cb, cur_lg


def save_data(dname: str, fname: str, data, **kwrgs) -> None:
    """ saved the data in a format based on fname extension"""
    if fname.endswith('.npy'):
        np.save(dname + fname, data, **kwrgs)
    elif fname.endswith('.pkl'):
        pickle.dump(data, open(dname + fname, "wb"))
    elif fname.endswith('.csv'):
        data.to_csv(dname + fname, **kwrgs)
    else:
        try:
            np.savetxt(dname + fname, data, **kwrgs)
        except:
            with open(dname + fname, "w") as f:
                f.writelines('%s\n' % str(s) for s in data)


def load_data(dname: str, fname: str, **kwrgs):
    """ loads the data saved as fname """
    if fname.endswith('.npy'):
        data = np.load(dname + fname, **kwrgs)
    elif fname.endswith('.csv'):
        data = pd.read_csv(dname + fname, index_col=0, **kwrgs)
    elif fname.endswith('.pkl'):
        data = pickle.load(open(dname + fname, "rb"))
    else:
        data = np.loadtxt(dname + fname, **kwrgs)
    return data


def save_fig(fig: Figure,
             dname: str,
             fname: str,
             fig_dpi: int = 200,
             fig_bbox: str = 'tight',
             **kwrgs) -> None:
    """ saves the figure """
    fig.savefig(dname + fname, bbox_inches=fig_bbox, dpi=fig_dpi, **kwrgs)
    plt.close(fig)


def makedir_if_not_exists(filename: str) -> None:
    """ Create the directory if it does not exists"""
    try:
        os.makedirs(filename)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def initiate_timer(in_str: str):
    """ Returns the currne time and prints the string """
    print(in_str, end="", flush=True)
    return time.time()


def print_elapsed_time(start):
    """ prints elapsed time since start"""
    hours, rem = divmod(time.time() - start, 3600)
    mins, secs = divmod(rem, 60)
    if hours == 0:
        if mins == 0:
            print("..took {0:d} sec".format(int(secs) + 1))
        else:
            print("..took {0:d} min {1:d} sec".format(int(mins), int(secs)))
    else:
        print("..took {0:d} hr {1:d} min".format(int(hours), int(mins)))


def get_elapsed_time(start) -> str:
    "returns the elapsed time as string"
    hours, rem = divmod(time.time() - start, 3600)
    mins, secs = divmod(rem, 60)
    if hours == 0:
        if mins == 0:
            xstr = "..took {0:d} sec".format(int(secs) + 1)
        else:
            xstr = "..took {0:d} min {1:d} sec".format(int(mins), int(secs))
    else:
        xstr = "..took {0:d} hr {1:d} min".format(int(hours), int(mins))
    return xstr


def file_exists(dirname: str, fname: str) -> bool:
    """ returns true if file exists """
    return os.path.exists(dirname + fname)


def remove_dirs_in(dname: str) -> None:
    """ remove all the subdirectories in the given directory"""
    dirnames = [f for f in os.scandir(dname) if f.is_dir()]
    for dirname in dirnames:
        shutil.rmtree(dirname)


def count_dirs_in(dname: str) -> int:
    """ counts all the subdirectories in the given directory"""
    blist = [os.path.isdir(os.path.join(dname, i)) for i in os.listdir(dname)]
    return sum(blist)


def get_dirs_in(dname: str) -> List[str]:
    """ return the names of all the subdirectories in the given directory"""
    blist = [i for i in os.listdir(dname)]
    blist = [x for x in blist if x.startswith('y')]
    return blist


def get_predefined_mode_id(wspeed: float, wdirn: float) -> str:
    """ sets up mode ID for predefined mode """
    return 's' + str(int(wspeed)) + 'd' + str(int(wdirn))







############ Extra ##########

# n_cpu = min(len(dtime_ids_to_solve), config.max_cores)
# with mp.Pool(n_cpu) as pool:
#     new_energys = pool.map(lambda dtime_id: solve_sparse_linear_system(
#         dtime_id,
#         orographs[dtime_ids.index(dtime_id)],
#         bndry_nodes,
#         bndry_energy,
#         sls_row_inds,
#         sls_col_inds,
#         sls_facs
#     ), dtime_ids_to_solve)

# def create_pdf_axis(cur_fig, cur_ax, xmin, xmid, xmax):
#     cur_ax.set_yticks([])
#     cur_ax.set_xticks([xmin, xmid, xmax])
#     cur_ax.set_xlim([xmin, xmax])
#     cur_ax.grid(True)
#     _, ymax = cur_ax.get_ylim()
#     # print(ymin, ymax)
#     cur_ax.set_ylim([0, 1.1 * ymax])
#     cur_fig.tight_layout()

# def standardize_matrix(
#         inA: np.ndarray
# ) -> np.ndarray:
#     tol = 1e-5
#     if np.amax(inA) - np.amin(inA) < 1e-5:
#         outA = np.multiply(np.ones(inA.shape), 1.)
#     else:
#         outA = np.divide(np.subtract(inA, np.amin(inA)),
#                          np.amax(inA) - np.amin(inA))
#         outA = outA.clip(min=tol)
#     return outA

# def get_transparent_cmap(cmp: str, cmin: float, bnds: List[float]):
#     icmp = cm.get_cmap(cmp)
#     newcolors = icmp(np.arange(icmp.N))
#     ind = int(icmp.N * cmin / (bnds[1] - bnds[0]))
#     newcolors[:ind, :] = (1, 0, 0, 0)
#     newcolors[ind:, -1] = np.linspace(0.6, 0.6, icmp.N - ind)
#     newcmp = ListedColormap(newcolors)
#     return newcmp

# def delete_files_of_type(dirname: str, fstring: str):
# for path, _, files in os.walk(dirname):
#     for name in files:
#         if name.endswith(fstring):
#             os.remove(os.path.join(path, name))

# def compute_count_matrix(
#         mat_shape: Tuple[int, int],
#         tracks: List[np.ndarray]
# ) -> np.ndarray:
#     """Computes count matrix that contains the number of
#     times eagles were present at each grid point"""

#     A = np.zeros(mat_shape, dtype=np.int16)
#     for track in tracks:
#         for move in track:
#             A[move[0], move[1]] += 1
#     return A.astype(np.int16)


# def compute_presence_probability_old(
#     A: np.ndarray,
#     radius: int
# ) -> np.ndarray:
#     """ Smothens a matrix using 2D covolution of the circular kernel matrix
#     with the given matrix """

#     kernel = np.zeros((2 * radius + 1, 2 * radius + 1))
#     y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
#     mask2 = x**2 + y**2 <= radius**2
#     kernel[mask2] = 1
#     Asmooth = ssg.convolve2d(A, kernel, mode='same')
#     Asmooth /= np.amax(Asmooth)
#     return Asmooth.astype(np.float32)

# def compute_energy_at_all_nodes(
#         run_id: str,
#         cond_coeff: np.ndarray,
#         bndry_nodes: np.ndarray,
#         bndry_energy: np.ndarray,
#         inner_nodes: np.ndarray
# ) -> np.ndarray:
#     """ Create and solve a linear system for unknown energy values
#     at inner nodes, returns the energy at all nodes"""

#     # print('Assembling global conductivity matrix')
#     start_time = time.time()
#     G_row_index = []
#     G_column_index = []
#     G_value = []
#     k = 0
#     tshape = cond_coeff.shape
#     node_rindex, node_cindex, nodes_nearby = gen_node_numbering(tshape)
#     for i in range(0, tshape[0] * tshape[1]):
#         # nearby_nodes_for_Kij = [x for r,x in enumerate(nodes_nearby[i,:])
#         # if r%2==0 and x>=0]
#         nearby_nodes_for_Kij = [x for r, x in enumerate(nodes_nearby[i, :])
#                                 if x >= 0]
#         for inearby in nearby_nodes_for_Kij:
#             coeff_a = cond_coeff[node_rindex[i], node_cindex[i]]
#             coeff_b = cond_coeff[node_rindex[inearby],
#                                  node_cindex[inearby]]
#             if coeff_a != 0 and coeff_b != 0:
#                 mean_cond = harmonic_mean(coeff_a, coeff_b)
#             else:
#                 mean_cond = 1e-10
#             if abs(tshape[0] + i - inearby) % 2 == 1:
#                 mean_cond /= sqrt(2.)
#             G_row_index.append(i)
#             G_column_index.append(inearby)
#             G_value.append(mean_cond)
#             k += 1
#         z = len(nearby_nodes_for_Kij)
#         if sum(G_value[k - z:k]) != 0:
#             G_value[k - z:k] = [x / sum(G_value[k - z:k])
#                                 for x in G_value[k - z:k]]
#     G_coo = ss.coo_matrix((np.array(G_value),
#                            (np.array(G_row_index),
#                             np.array(G_column_index))),
#                           shape=(tshape[0] * tshape[1],
#                                  tshape[0] * tshape[1]))
#     G_csr = G_coo.tocsr()
#     print('{0:s}: assembling global matrix {1:s}'.format(
#         run_id, get_elapsed_time(start_time)), flush=True)
#     # print('Sparsity level:', len(G_value),'/',tshape[0]**2*tshape[1]**2)
#     start_time = time.time()
#     G_inner_csr = G_csr[inner_nodes, :]
#     G_inner_csc = G_inner_csr.tocoo().tocsc()
#     G_inner_inner_csc = G_inner_csc[:, inner_nodes]
#     G_inner_bndry_csc = G_inner_csc[:, bndry_nodes]
#     b_vec = G_inner_bndry_csc.dot(bndry_energy)
#     # inner_energy = ss.linalg.spsolve(G_inner_inner_csc, -b_vec)
#     A_matrix = ss.eye(np.size(inner_nodes)).tocsc() - G_inner_inner_csc
#     inner_energy = ss.linalg.spsolve(A_matrix, b_vec)
#     # inner_energy,  istop, itn, r1nor = ss.linalg.lsqr(
#     #     G_inner_inner_csc, -b_vec, damp=0.0, atol=1e-5, btol=1e-5)[:4]
#     # print('{0:d}, {1:d}, {2:f}'.format(istop, itn, r1nor), flush=True)
#     print('{0:s}: solving linear system {1:s}'.format(
#         run_id, get_elapsed_time(start_time)), flush=True)
#     global_energy = np.empty(tshape[0] * tshape[1])
#     global_energy[inner_nodes] = inner_energy
#     global_energy[bndry_nodes] = bndry_energy
#     pot_energy = np.empty(tshape)
#     for i, (cur_xindex, cur_yindex) in enumerate(zip(node_rindex, node_cindex)):
#         pot_energy[cur_xindex, cur_yindex] = global_energy[i]
#     return pot_energy


# # %% define static constants for eagle track generation
# neighbour_deltas = []
# neighbour_delta_norms_inv = np.empty((3, 3), dtype=np.float32)
# center = (np.array(neighbour_delta_norms_inv.shape, dtype=np.int) - 1) // 2
# for r in range(neighbour_delta_norms_inv.shape[0]):
#     for c in range(neighbour_delta_norms_inv.shape[1]):
#         delta = np.array([r, c], dtype=np.int) - center
#         neighbour_deltas.append(delta)
#         distance = np.linalg.norm(delta)
#         neighbour_delta_norms_inv[r, c] = 1.0 / distance if distance > 0 else 0

# neighbour_deltas_alt = neighbour_deltas[0:4] + neighbour_deltas[5:]
# flat_neighbour_delta_norms_inv = list(neighbour_delta_norms_inv.flatten())
# neighbour_delta_norms_inv_alt = np.array(
#     flat_neighbour_delta_norms_inv[0:4] + flat_neighbour_delta_norms_inv[5:],
#     dtype=np.float32)
# delta_rows_alt = np.array([delta[0] for delta in neighbour_deltas_alt])
# delta_cols_alt = np.array([delta[1] for delta in neighbour_deltas_alt])


# def get_track_restrictions(dr: int, dc: int):
#     Amat = np.zeros((3, 3), dtype=int)
#     dr_mat = np.zeros((3, 3), dtype=int)
#     dc_mat = np.zeros((3, 3), dtype=int)
#     if abs(dr + dc % 2) == 1:
#         if dr == 0:
#             Amat[:, dc + 1] = 1
#         else:
#             Amat[dr + 1, :] = 1
#     else:
#         dr_mat[(dr + 1, 1), :] = 1
#         dc_mat[:, (1, dc + 1)] = 1
#         Amat = np.logical_and(dr_mat, dc_mat).astype(int)
#     if dr == 0 and dc == 0:
#         Amat[:, :] = 1
#     Amat[1, 1] = 0
#     return Amat.flatten()


# def generate_eagle_track(
#         i: int,
#         coeff: np.ndarray,
#         penergy: np.ndarray,
#         start_loc: List[int],
#         dirn_restrict: int,
#         nu: float
# ):
#     """ Generate an eagle track """

#     num_rows, num_cols = coeff.shape
#     scaled_neighbour_delta_norms_inv = 1. * neighbour_delta_norms_inv
#     scaled_neighbour_delta_norms_inv = scaled_neighbour_delta_norms_inv.astype(
#         np.float32)
#     burnin = 5
#     max_moves = 10 * max(num_rows, num_cols)
#     dirn = [0, 0]
#     previous_dirn = [0, 0]
#     np.random.seed(i)
#     position = start_loc.copy()
#     trajectory = []
#     trajectory.append(position)
#     for k in range(max_moves):
#         r, c = position
#         if k > max_moves - 2:
#             print('Maximum steps reached!', i, k, r, c)
#         if k > burnin:
#             if r <= 0 or r >= num_rows - 1 or c <= 0 or c >= num_cols - 1:
#                 # print('hit boundary!', k, ': ', r, c)
#                 break  # absorb if we hit a boundary
#         else:
#             if r == 0 or r == 1:
#                 r += 1
#             elif r == num_rows - 1 or r == num_rows:
#                 r -= 1
#             if c == 0 or c == 1:
#                 c += 1
#             elif c == num_cols - 1 or c == num_cols:
#                 c -= 1
#         position = [r, c]
#         previous_dirn = np.copy(dirn)
#         local_conductance = coeff[r - 1:r + 2, c - 1:c + 2]
#         local_potential_energy = penergy[r - 1:r + 2, c - 1:c + 2]
#         local_conductance = local_conductance.clip(min=1e-10)
#         mc = 2.0 / (1.0 / local_conductance[1, 1] + 1.0 / local_conductance)
#         q_diff = local_potential_energy[1, 1] - local_potential_energy
#         if np.count_nonzero(q_diff) == 0:
#             q_diff = 1. + np.random.rand(*q_diff.shape) * 1e-1
#             # print('All potentials same!', i, k, r, c, mc)
#         q_diff = np.multiply(q_diff, neighbour_delta_norms_inv)
#         q = np.multiply(mc, q_diff)
#         q = q.flatten()
#         q -= np.min(q)
#         q[4] = 0.
#         if dirn_restrict > 0 and k > burnin:
#             z = get_track_restrictions(*dirn)
#             if dirn_restrict == 2:
#                 z = np.logical_and(z, get_track_restrictions(*previous_dirn))
#             if sum(z) != 0:
#                 q_new = [x * float(y) for x, y in zip(q, z)]
#                 if np.sum(q_new) != 0:
#                     q = q_new.copy()
#         if np.sum(q) != 0:
#             q /= np.sum(q)
#             q = np.power(q, nu)
#             q /= np.sum(q)
#             chosen_index = np.random.choice(range(len(q)), p=q)
#         else:
#             # print('Sum(q) is zero!', i, k, r, c, q)
#             chosen_index = np.random.choice(range(len(q)))
#         dirn = neighbour_deltas[chosen_index]
#         position = [x + y for x, y in zip(position, dirn)]
#         trajectory.append(position)
#     return np.array(trajectory, dtype=np.int16)


# def generate_eagle_track_drw(
#         i: int,
#         coeff: np.ndarray,
#         penergy: np.ndarray,
#         start_loc: List[int],
#         dirn_restrict: int,
#         nu: float
# ):
#     """ Generate an eagle track """

#     num_rows, num_cols = coeff.shape
#     scaled_neighbour_delta_norms_inv = 1. * neighbour_delta_norms_inv
#     scaled_neighbour_delta_norms_inv = scaled_neighbour_delta_norms_inv.astype(
#         np.float32)
#     burnin = 5
#     max_moves = 10 * max(num_rows, num_cols)
#     dirn = [0, 0]
#     previous_dirn = [0, 0]
#     np.random.seed(i)
#     position = start_loc.copy()
#     trajectory = []
#     trajectory.append(position)
#     for k in range(max_moves):
#         r, c = position
#         if k > max_moves - 2:
#             print('Maximum steps reached!', i, k, r, c)
#         if k > burnin:
#             if r <= 0 or r >= num_rows - 1 or c <= 0 or c >= num_cols - 1:
#                 # print('hit boundary!', k, ': ', r, c)
#                 break  # absorb if we hit a boundary
#         else:
#             if r == 0 or r == 1:
#                 r += 1
#             elif r == num_rows - 1 or r == num_rows:
#                 r -= 1
#             if c == 0 or c == 1:
#                 c += 1
#             elif c == num_cols - 1 or c == num_cols:
#                 c -= 1
#         position = [r, c]
#         previous_dirn = np.copy(dirn)
#         # local_conductance = coeff[r - 1:r + 2, c - 1:c + 2]
#         # local_potential_energy = penergy[r - 1:r + 2, c - 1:c + 2]
#         # local_conductance = local_conductance.clip(min=1e-10)
#         # mc = 2.0 / (1.0 / local_conductance[1, 1] + 1.0 / local_conductance)
#         # q_diff = local_potential_energy[1, 1] - local_potential_energy
#         # if np.count_nonzero(q_diff) == 0:
#         #     q_diff = 1. + np.random.rand(*q_diff.shape) * 1e-1
#         #     #print('All potentials same!', i, k, r, c, mc)
#         # q_diff = np.multiply(q_diff, neighbour_delta_norms_inv)
#         # q = np.multiply(mc, q_diff)
#         # q = q.flatten()
#         # q -= np.min(q)
#         # q[4] = 0.
#         # if dirn_restrict > 0 and k > burnin:
#         #     z = get_track_restrictions(*dirn)
#         #     if dirn_restrict == 2:
#         #         z = np.logical_and(z, get_track_restrictions(*previous_dirn))
#         #     if sum(z) != 0:
#         #         q_new = [x * float(y) for x, y in zip(q, z)]
#         #         if np.sum(q_new) != 0:
#         #             q = q_new.copy()
#         q = [1., 1., 0., 1., 0., 0., 1., 1., 0.]
#         if np.sum(q) != 0:
#             q /= np.sum(q)
#             q = np.power(q, nu)
#             q /= np.sum(q)
#             chosen_index = np.random.choice(range(len(q)), p=q)
#         else:
#             # print('Sum(q) is zero!', i, k, r, c, q)
#             chosen_index = np.random.choice(range(len(q)))
#         dirn = neighbour_deltas[chosen_index]
#         position = [x + y for x, y in zip(position, dirn)]
#         trajectory.append(position)
#     return np.array(trajectory, dtype=np.int16)


# def get_position_from_index(index: int, tshape: tuple) -> List[int]:
#     return [int(index % tshape[1]), int(index // tshape[1])]


# def get_starting_indices_old(ntracks, bounds, res, entry_type='uniform'):
#     base_ind = np.arange(int(bounds[0] / res), int(bounds[1] / res))
#     if entry_type == 'uniform':
#         idx = np.round(np.linspace(0, len(base_ind) - 1,
#                                    ntracks % len(base_ind))).astype(int)
#         target_ind = base_ind[idx]
#         for _ in range(ntracks // len(base_ind)):
#             target_ind = np.append(target_ind, base_ind)
#         return target_ind
#     elif entry_type == 'random':
#         return np.random.randint(base_ind[0], base_ind[-1], ntracks)


# def terrain_conductance_model_parallel(
#         run_id: str,
#         updraft: np.ndarray,
#         bndry_cond: str,
#         total_tracks: int,
#         bounds: Tuple[float, float],
#         entry_type: str,
#         dirn_restrict: int,
#         nu: float,
#         res: float,
#         n_cpu: int
# ):
#     """ Runs terrain conductance model"""

#     # unpack eagle track initiation settings
#     start_time = time.time()
#     t_start, t_end, t_dist, t_per, nu, dirn_restrict = track_pars

#     # set up boundary conditions
#     tshape = np.shape(updraft)
#     bndry_nodes, inner_nodes, bndry_energy, starting_nodes \
#         = get_boundary_values(bndry_cond, tshape)

#     # computed potential energy
#     penergy = compute_energy_at_all_nodes(
#         run_id,
#         updraft,
#         bndry_nodes,
#         bndry_energy,
#         inner_nodes)

#     # %% Generating eagle tracks
#     start_index = np.repeat(
#         np.arange(
#             t_start,
#             t_end,
#             t_dist,
#             dtype='int16'),
#         t_per)
#     total_tracks = np.size(start_index)
#     # print('Generating {:d} eagle tracks .. '.format(total_tracks), end='')
#     eagle_tracks = []
#     updraft = updraft.astype(np.float32)
#     penergy = penergy.astype(np.float32)
#     with multiprocessing.Pool(n_cpu) as pool:
#         eagle_tracks = pool.map(
#             lambda i: generate_eagle_track(
#                 i,
#                 updraft,
#                 penergy,
#                 get_position_from_index(
#                     starting_nodes[start_index[i]], tshape),
#                 dirn_restrict,
#                 nu
#             ),
#             range(total_tracks))
#     print('\nTCmodel: {0:s} .. took {1:.1f} secs'.format(
#         run_id, get_elapsed_time(start_time)))
#     return eagle_tracks, penergy.astype(np.float16)


# def terrain_conductance_model_parallel_old(
#         run_id: str,
#         updraft: np.ndarray,
#         bndry_cond: str,
#         track_pars: List,
#         n_cpu: int
# ):
#     """ Runs terrain conductance model"""

#     # unpack eagle track initiation settings
#     start_time = time.time()
#     t_start, t_end, t_dist, t_per, nu, dirn_restrict = track_pars

#     # set up boundary conditions
#     tshape = np.shape(updraft)
#     bndry_nodes, inner_nodes, bndry_energy, starting_nodes \
#         = get_boundary_values(bndry_cond, tshape)

#     # computed potential energy
#     penergy = compute_energy_at_all_nodes(
#         run_id,
#         updraft,
#         bndry_nodes,
#         bndry_energy,
#         inner_nodes)

#     # %% Generating eagle tracks
#     start_index = np.repeat(
#         np.arange(
#             t_start,
#             t_end,
#             t_dist,
#             dtype='int16'),
#         t_per)
#     total_tracks = np.size(start_index)
#     # print('Generating {:d} eagle tracks .. '.format(total_tracks), end='')
#     eagle_tracks = []
#     updraft = updraft.astype(np.float32)
#     penergy = penergy.astype(np.float32)
#     with multiprocessing.Pool(n_cpu) as pool:
#         eagle_tracks = pool.map(
#             lambda i: generate_eagle_track(
#                 i,
#                 updraft,
#                 penergy,
#                 get_position_from_index(
#                     starting_nodes[start_index[i]], tshape),
#                 dirn_restrict,
#                 nu
#             ),
#             range(total_tracks))
#     print('\nTCmodel: {0:s} .. took {1:.1f} secs'.format(
#         run_id, get_elapsed_time(start_time)))
#     return eagle_tracks, penergy.astype(np.float16)


# def terrain_conductance_model_parallel_drw(
#         run_id: str,
#         updraft: np.ndarray,
#         bndry_cond: str,
#         track_pars: List,
#         n_cpu: int
# ):
#     """ Runs terrain conductance model"""

#     # unpack eagle track initiation settings
#     start_time = time.time()
#     t_start, t_end, t_dist, t_per, nu, dirn_restrict = track_pars

#     # set up boundary conditions
#     tshape = np.shape(updraft)
#     bndry_nodes, inner_nodes, bndry_energy, starting_nodes \
#         = get_boundary_values(bndry_cond, tshape)

#     penergy = np.zeros(updraft.shape)
#     # computed potential energy
#     # penergy = compute_energy_at_all_nodes(
#     #     run_id,
#     #     updraft,
#     #     bndry_nodes,
#     #     bndry_energy,
#     #     inner_nodes)

#     # %% Generating eagle tracks
#     start_index = np.repeat(
#         np.arange(
#             t_start,
#             t_end,
#             t_dist,
#             dtype='int16'),
#         t_per)
#     total_tracks = np.size(start_index)
#     # print('Generating {:d} eagle tracks .. '.format(total_tracks), end='')
#     eagle_tracks = []
#     updraft = updraft.astype(np.float32)
#     penergy = penergy.astype(np.float32)
#     with multiprocessing.Pool(n_cpu) as pool:
#         eagle_tracks = pool.map(
#             lambda i: generate_eagle_track_drw(
#                 i,
#                 updraft,
#                 penergy,
#                 get_position_from_index(
#                     starting_nodes[start_index[i]], tshape),
#                 dirn_restrict,
#                 nu
#             ),
#             range(total_tracks))
#     print('\nTCmodel: {0:s} .. took {1:.1f} secs'.format(
#         run_id, get_elapsed_time(start_time)))
#     return eagle_tracks, penergy.astype(np.float16)


# def terrain_conductance_model_serial(
#         run_id: str,
#         updraft: np.ndarray,
#         bndry_cond: str,
#         track_pars: List
# ):
#     """ Runs terrain conductance model"""

#     # unpack eagle track initiation settings
#     start_time = time.time()
#     t_start, t_end, t_dist, t_per, nu, dirn_restrict = track_pars
#     print('TCmodel: {0:s}'.format(run_id), flush=True)

#     # set up boundary conditions
#     tshape = np.shape(updraft)
#     bndry_nodes, inner_nodes, bndry_energy, starting_nodes \
#         = get_boundary_values(bndry_cond, tshape)

#     # computed potential energy
#     penergy = compute_energy_at_all_nodes(
#         run_id,
#         updraft,
#         bndry_nodes,
#         bndry_energy,
#         inner_nodes)

#     # %% Generating eagle tracks
#     start_index = np.repeat(
#         np.arange(
#             t_start,
#             t_end,
#             t_dist,
#             dtype='int16'),
#         t_per)
#     total_tracks = np.size(start_index)
#     # print('Generating {:d} eagle tracks .. '.format(total_tracks), end='',
#     #      flush=True)
#     eagle_tracks = []
#     # updraft = updraft.astype(np.float32)
#     # penergy = penergy.astype(np.float32)
#     for i in range(total_tracks):
#         eagle_tracks.append(generate_eagle_track(
#             i,
#             updraft,
#             penergy,
#             get_position_from_index(starting_nodes[start_index[i]], tshape),
#             dirn_restrict,
#             nu
#         ))
#     print('TCmodel: {0:s} .. took {1:.1f} secs'.format(
#         run_id, get_elapsed_time(start_time)))
#     return eagle_tracks, penergy.astype(np.float32)


# def terrain_conductance_model(
#         run_id: str,
#         coeff: np.ndarray,
#         bndry_cond: str,
#         total_tracks: int,
#         bounds: Tuple[float, float],
#         entry_type: str,
#         dirn_restrict: int,
#         nu: float,
#         res: float
# ):
#     """ Runs terrain conductance model"""

#     # unpack eagle track initiation settings
#     start_time = time.time()
#     # print('TCmodel: {0:s}'.format(run_id), flush=True)

#     # set up boundary conditions
#     tshape = np.shape(coeff)
#     bndry_nodes, inner_nodes, bndry_energy, starting_nodes \
#         = get_boundary_values(bndry_cond, tshape)

#     # computed potential energy
#     penergy = compute_energy_at_all_nodes(
#         run_id,
#         coeff,
#         bndry_nodes,
#         bndry_energy,
#         inner_nodes)

#     # Generating eagle tracks
#     start_time = time.time()
#     starting_indices = get_starting_indices(
#         total_tracks, bounds, res, entry_type)
#     eagle_tracks = []
#     coeff = coeff.astype(np.float32)
#     penergy = penergy.astype(np.float32)
#     for i in range(total_tracks):
#         eagle_tracks.append(generate_eagle_track(
#             i,
#             coeff,
#             penergy,
#             get_position_from_index(
#                 starting_nodes[starting_indices[i]], tshape),
#             dirn_restrict,
#             nu
#         ))
#     print('{0:s}: generating eagle tracks {1:s}'.format(
#         run_id, get_elapsed_time(start_time)), flush=True)
#     return eagle_tracks, penergy.astype(np.float32)


# def compute_count_matrix(
#         mat_shape: Tuple[int, int],
#         tracks: List[np.ndarray]
# ) -> np.ndarray:
#     """Computes count matrix that contains the number of
#     times eagles were present at each grid point"""

#     # print('Computing counts for {0:d} tracks'.format(len(tracks)), end="")
#     # start_time = time.time()
#     A = np.zeros(mat_shape, dtype=np.int16)
#     for track in tracks:
#         for move in track:
#             A[move[0], move[1]] += 1
#     # print_elapsed_time(start_time)
#     return A.astype(np.int16)


# # def compute_presence_probability(
# #     A: np.ndarray,
# #     radius: int
# # ) -> np.ndarray:
# #     """ Smothens a matrix using 2D covolution of the circular kernel matrix
# #     with the givem atrix """

# #     kernel = np.zeros((2 * radius + 1, 2 * radius + 1))
# #     y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
# #     mask2 = x**2 + y**2 <= radius**2
# #     kernel[mask2] = 1
# #     Asmooth = ssg.convolve2d(A, kernel, mode='same')
# #     Asmooth /= np.amax(Asmooth)
# #     return Asmooth.astype(np.float32)

# def get_minmax_indices(
#         xgrid: np.ndarray,
#         ygrid: np.ndarray,
#         extent: Tuple[float, float, float, float]
# ):
#     idxmin = (np.abs(xgrid - extent[0])).argmin()
#     idymin = (np.abs(ygrid - extent[2])).argmin()
#     idxmax = (np.abs(xgrid - extent[1])).argmin()
#     idymax = (np.abs(ygrid - extent[3])).argmin()
#     return [idxmin, idxmax, idymin, idymax]


# def compute_presence(
#         counts: np.ndarray,
#         extent: Tuple[float, float, float, float],
#         radius: int
# ) -> np.ndarray:
#     """ Smothens a matrix using 2D covolution of the circular kernel matrix
#     with the givem matrix """

#     trimmed = counts[extent[2]:extent[3], extent[0]:extent[1]]
#     kernel = np.zeros((2 * radius + 1, 2 * radius + 1))
#     y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
#     mask2 = x ** 2 + y ** 2 <= radius ** 2
#     kernel[mask2] = 1
#     Asmooth = ssg.convolve2d(trimmed, kernel, mode='same')
#     # Asmooth /= np.amax(Asmooth)
#     return Asmooth.astype(np.float32)


# # %% Junk
# # print(A[0,:])

# # # %% Generate eagle track
# # def generate_eagle_track(
# #         conductance_map,
# #         potential_energy_map,
# #         start,  # (row, col) np array
# #         grid_spacing,
# #         nu,
# #         max_moves=15000
# # ):
# #     """ Generate an eagle track """

# #     position = np.copy(start)
# #     trajectory = [position]
# #     num_rows, num_cols = conductance_map.shape

# #     scaled_neighbour_delta_norms_inv = grid_spacing * neighbour_delta_norms_inv
# #     scaled_neighbour_delta_norms_inv = scaled_neighbour_delta_norms_inv.astype(
# #         np.float32)

# #     for step_number in range(max_moves):
# #         position_tuple = tuple(position)
# #         r = position_tuple[0]
# #         c = position_tuple[1]

# #         if r <= 0 or r >= num_rows - 1 or c <= 0 or c >= num_cols - 1:
# #             break  # absorb if we hit a boundary

# #         local_conductance = conductance_map[r - 1:r + 2, c - 1:c + 2]
# #         local_potential_energy = potential_energy_map[r - 1:r + 2, c - 1:c + 2]

# #         q = 2.0 / (1.0 / local_conductance[1, 1] + 1.0 / local_conductance)
# #         q *= (local_potential_energy[1, 1] -
# #               local_potential_energy) * scaled_neighbour_delta_norms_inv

# #         if np.max(q) < 0:
# #             print('hit well')
# #             break  # terminate if we're in a well

# #         # possibly smooth eagle tracks by penalizing changing velocities
# #         # test out different action probability schemes

# #         # q = q.flatten()
# #         # q -= np.max(q)  # prevents overflow when exp()'ing
# #         # q /= 1e-1 #boltzmann_temperature
# #         # p = np.exp(q)
# #         # q[4] = 0.0
# #         # p /= np.sum(p)

# #         q = q.flatten()
# #         q -= np.min(q)
# #         q = np.power(q, nu)
# #         q[4] = 0.0
# #         q /= np.sum(q)

# #         chosen_index = np.random.choice(range(len(q)), p=q)
# #         position = position + neighbour_deltas[chosen_index]
# #         trajectory.append(position)
# #     return np.array(trajectory)


# # # %% Generate eagle track
# # def generate_eagle_track_alternate(
# #         conductance_map,
# #         potential_energy_map,
# #         start,  # (row, col) np array
# #         grid_spacing,
# #         nu,
# #         max_moves=15000
# # ):
# #     """ Generate an eagle track """
# #     position = np.copy(start)
# #     trajectory = [position]
# #     delta_norms_inv_scaled = grid_spacing * neighbour_delta_norms_inv_alt

# #     num_rows, num_cols = conductance_map.shape

# #     for step_number in range(max_moves):
# #         position_tuple = tuple(position)

# #         if position_tuple[0] <= 0 or position_tuple[0] >= num_rows - 1 or \
# #                 position_tuple[1] <= 0 or position_tuple[1] >= num_cols - 1:
# #             break  # terminate if we hit a boundary

# #         # current_conductance = conductance_map[position_tuple]
# #         current_conductance_inv = 1.0 / conductance_map[position_tuple]
# #         current_potential_energy = potential_energy_map[position_tuple]

# #         # candidate_positions = [position + delta for delta in neighbour_deltas]
# #         candidate_rows = delta_rows_alt + position_tuple[0]
# #         candidate_cols = delta_cols_alt + position_tuple[1]
# #         candidate_conductance = conductance_map[candidate_rows, candidate_cols]
# #         candidate_potential_energy = potential_energy_map[candidate_rows, candidate_cols]

# #         q = 2.0 / (current_conductance_inv + 1.0 / candidate_conductance)
# #         q *= (current_potential_energy - candidate_potential_energy) * \
# #             delta_norms_inv_scaled

# #         if np.max(q) < 0:
# #             print('hit well')
# #             break  # terminate if we're in a well

# #         # q -= np.max(q)  # prevents overflow when exp()'ing
# #         # q /= boltzmann_temperature
# #         # p = np.exp(q)
# #         # p /= np.sum(p)

# #         q -= np.min(q)
# #         q = np.power(q, nu)
# #         q /= np.sum(q)

# #         chosen_index = np.random.choice(range(len(q)), p=q)
# #         position = position + neighbour_deltas_alt[chosen_index]
# #         trajectory.append(position)

# #     return np.array(trajectory)


# # # %% Generate eagle track
# # def generate_eagle_track_old(
# #         cond_coeff,  # 1000x1000 float array
# #         penergy,  # 1000x1000 float array
# #         start_node,  # 50999
# #         grid_spacing,  # .05
# #         smart_fac,  # 1.0
# #         max_moves,  # 15000
# #         node_rindex,  # 1M int array
# #         node_cindex,  # 1M int array
# #         nodes_nearby,  # 1Mx8 int array
# # ):
# #     """ Generate an eagle track """

# #     kk = 0
# #     curn = start_node
# #     track_not_converged = True
# #     eagle_track = []
# #     while track_not_converged:
# #         eagle_track.append([node_rindex[curn], node_cindex[curn]])
# #         try_dirs, tryns = zip(*[(j, x) for j, x in enumerate(nodes_nearby[curn, :])
# #                                 if x >= 0])
# #         try_qij = np.zeros(len(tryns))
# #         for k, (try_dir, tryn) in enumerate(zip(try_dirs, tryns)):
# #             try_Kij = 2 / (1 / cond_coeff[node_rindex[curn], node_cindex[curn]]
# #                            + 1 / cond_coeff[node_rindex[tryn], node_cindex[tryn]])
# #             try_qij[k] = try_Kij * (penergy[node_rindex[curn], node_cindex[curn]]
# #                                     - penergy[node_rindex[tryn],
# #                                                     node_cindex[tryn]]) / grid_spacing
# #             if try_dir % 2 == 1:
# #                 try_qij[k] = try_qij[k] / sqrt(2.0)
# #                 # print(k,try_dir,tryn,try_Kij,try_qij[k])
# #         if np.all(try_qij <= 0.):
# #             break
# #         try_qij = (try_qij - min(try_qij)) / sum(try_qij - min(try_qij))
# #         try_qij = try_qij ** smart_fac / sum(try_qij ** smart_fac)
# #         curn = tryns[np.random.choice(len(tryns), 1, p=try_qij).item()]
# #         kk += 1
# #         if kk > max_moves:
# #             track_not_converged = False
# #     return np.array(eagle_track)


# # # %% computes presence map (kde approximated)
# # def compute_presence_map(
# #         eagle_tracks, xy_bnd, xgrid, ygrid
# # ):
# #     """ Computes KDE approximation of eagle presence probability"""

# #     tshape = (xgrid.size, ygrid.size)
# #     node_rindex, node_cindex, _ = gen_node_numbering(tshape)
# #     xmesh, ymesh = np.mgrid[xy_bnd[0]:xy_bnd[1]:50j, xy_bnd[2]:xy_bnd[3]:50j]
# #     positions = np.vstack([xmesh.ravel(), ymesh.ravel()])
# #     x_values = np.array([])
# #     y_values = np.array([])
# #     for eagle_track in eagle_tracks:
# #         x_values = np.concatenate((x_values, xgrid[eagle_track[:, 1]]))
# #         y_values = np.concatenate((y_values, ygrid[eagle_track[:, 0]]))
# #     values = np.vstack([x_values, y_values])
# #     # kernel = st.gaussian_kde(values,bw_method=0.06)
# #     kernel = st.gaussian_kde(values)
# #     # print(kernel.factor)
# #     Z = np.reshape(kernel(positions), xmesh.shape)
# #     return Z.T


# def construct_sparse_mat_for_spsolve(ind, cond, nodes_nearby,
#                                      node_rindex, node_cindex):
#     rindex = []
#     cindex = []
#     value = []
#     # nearby_nodes_for_Kij = [x for r,x in enumerate(nodes_nearby[i,:])
#     # if r%2==0 and x>=0]
#     k = 0
#     for i in range(*ind):
#         rel_i = i - ind[0]
#         nearby_nodes_for_Kij = [x for r, x in enumerate(nodes_nearby[rel_i, :])
#                                 if x >= 0]
#         for inearby in nearby_nodes_for_Kij:
#             rindex.append(i)
#             cindex.append(inearby)
#             value.append(2 / (1 / cond[node_rindex[i], node_cindex[i]]
#                               + 1 / cond[node_rindex[inearby],
#                                          node_cindex[inearby]]))
#             k += 1
#         ntemp = len(nearby_nodes_for_Kij)
#         value[k - ntemp:k] = [x / sum(value[k - ntemp:k])
#                               for x in value[k - ntemp:k]]
#     return rindex, cindex, value


# # # %% Create and solve a linear system for unknown energy values at inner nodes
# def compute_energy_at_all_nodes_parallel(
#         coeff,
#         bndry_nodes,
#         bndry_energy,
#         inner_nodes,
#         n_cpu
# ):
#     """ Create and solve a linear system for unknown energy values
#     at inner nodes, returns the energy at all nodes"""

#     # print('Assembling global conductivity matrix', end=" ")
#     # start_time = time.time()

#     # print('Assembling global conductivity matrix .. ', end=" ")
#     tshape = coeff.shape
#     node_rindex, node_cindex, nodes_nearby = gen_node_numbering(tshape)
#     G_value = []
#     G_rindex = []
#     G_cindex = []
#     # with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
#     #     eagle_tracks = pool.map(
#     #         lambda i:
#     # coeff = coeff.astype(np.float32)
#     # node_rindex = node_rindex.astype(np.int32)
#     # node_cindex = node_cindex.astype(np.int32)
#     # print(node_rindex.dtype, node_cindex.dtype)
#     # print(coeff.nbytes / 1024**2, nodes_nearby.nbytes / 1024**2,
#     #      node_rindex.nbytes / 1024**2, node_cindex.nbytes / 1024**2)

#     n_chunks = tshape[0] * tshape[1] // n_cpu
#     increments = list(range(0, tshape[0] * tshape[1], n_chunks))
#     increments.append(tshape[0] * tshape[1])
#     chunk_list = [(increments[i], increments[i + 1])
#                   for i in range(len(increments) - 1)]
#     # print(n_chunks, increments, chunk_list)

#     with multiprocessing.Pool() as pool:
#         out = pool.map(
#             lambda x: construct_sparse_mat_for_spsolve(x,
#                                                        coeff,
#                                                        nodes_nearby[x[0]
#                                                            :x[1], :],
#                                                        node_rindex,
#                                                        node_cindex),
#             chunk_list)
#     # print(out[-1])
#     for (rindex, cindex, value) in out:
#         G_rindex.extend(rindex)
#         G_cindex.extend(cindex)
#         G_value.extend(value)
#     # for i in range(0, tshape[0] * tshape[1]):
#     #     rindex, cindex, value = construct_sparse_mat_for_spsolve(i, coeff,
#     #                                                              nodes_nearby,
#     #                                                              node_rindex,
#     #                                                              node_cindex)
#     #     # G_rindex = np.concatenate((G_rindex, rindex))
#     #     # G_cindex = np.concatenate((G_cindex, cindex))
#     #     # G_value = np.concatenate((G_value, value))
#     #     G_rindex.extend(rindex)
#     #     G_cindex.extend(cindex)
#     #     G_value.extend(value)
#     G_coo = ss.coo_matrix((np.array(G_value),
#                            (np.array(G_rindex), np.array(G_cindex))),
#                           shape=(tshape[0] * tshape[1],
#                                  tshape[0] * tshape[1]))
#     # print(G_coo.dtype)
#     # print_elapsed_time(start_time)
#     # print('Solving for unknown energy values', end=' ')
#     G_csr = G_coo.tocsr()
#     # print('Sparsity level:', len(G_value),'/',tshape[0]**2*tshape[1]**2)
#     G_inner_csr = G_csr[inner_nodes, :]
#     G_inner_csc = G_inner_csr.tocoo().tocsc()
#     G_inner_inner_csc = G_inner_csc[:, inner_nodes]
#     G_inner_bndry_csc = G_inner_csc[:, bndry_nodes]
#     b_vec = G_inner_bndry_csc.dot(bndry_energy)
#     A_matrix = ss.eye(np.size(inner_nodes)).tocsc() - G_inner_inner_csc
#     # print_elapsed_time(start_time)
#     # print(A_matrix.dtype, b_vec.dtype)
#     inner_energy = ss.linalg.spsolve(A_matrix, b_vec)
#     # print_elapsed_time(start_time)
#     # Reassemble the global solution for energy
#     global_energy = np.empty(tshape[0] * tshape[1])
#     global_energy[inner_nodes] = inner_energy
#     global_energy[bndry_nodes] = bndry_energy
#     penergy = np.empty(tshape)
#     for i, (cur_xindex, cur_yindex) in enumerate(zip(node_rindex, node_cindex)):
#         penergy[cur_xindex, cur_yindex] = global_energy[i]
#     # print_elapsed_time(start_time)
#     # print(penergy.dtype)
#     return penergy


# fig, ax = plt.subplots(figsize=fig_size)
# cm = ax.imshow(terrain_alt, cmap='terrain', origin='lower')
# ax.plot(start_indices[0, :], start_indices[1, :], 'r*')
# cb, lg = create_gis_axis(fig, ax, cm, 'both', 1.)
# for i in range(M):
#     for j in range(N):
#         ax.text(j, i, "%d" % (j * M + i), ha="center", fontsize=3)
# save_fig(fig, config.terrain_fig_dir, 'test.png', fig_dpi)

# get_boundary_node_ids(terrain_alt.shape)
# tracks, potential = terrain_conductance_model_parallel(
#     run_id,
#     orograph,
#     config.track_direction,
#     config.track_count,
#     config.track_start_region,
#     config.track_start_type,
#     config.track_dirn_restrict,
#     config.track_stochastic_nu,
#     config.terrain_res,
#     n_cpu
# )
# #     tracks_all.append(tracks)
# #     pot_all.append(potential)
# # for j, _ in enumerate(coeffs):
# save_data(data_dir, run_id + cases[j] + '_tracks.pkl', tracks)
# save_data(data_dir, run_id + cases[j] + '_potential.npy', potential)


# config = setup_config()
# # load updraft data
# datetimes = get_saved_datetimes(
#     config.data_dir, config.datetime_format, '_orograph.npy')
# if not datetimes:
#     raise Exception('\nRun compute_updrafts.py first!')
# datetime_ids = [t.strftime(config.datetime_format) for t in datetimes]
# data_dirnames = [config.data_dir + datetime_ids[i] +
#                  '/' for i in range(len(datetimes))]
# done_datetimes = get_saved_datetimes(
#     config.data_dir, config.datetime_format, '_tracks.pkl')
# todo_indices = [i for i, x in enumerate(datetimes) if x not in done_datetimes]
# if not todo_indices:
#     sys.exit('Requested TCModel output already exists!')


# def run_tcmodel(dt_id, data_dir):
#     run_id = dt_id + '_' + config.config.track_direction + '_'
#     orograph = load_data(data_dir, dt_id + '_orograph.npy')
#     # thermal = load_data(data_dir, dt_id + '_thermal.npy')
#     # net = np.add(orograph, thermal)
#     # prob = load_data(data_dir, dt_id + '_prob.npy')
#     coeffs = [orograph]
#     #   standardize_matrix(orograph), standardize_matrix(thermal),
#     #   standardize_matrix(net), standardize_matrix(prob)]
#     # usables = []
#     # for updraft in coeffs[:3]:
#     #     updraft_usable = np.copy(updraft)
#     #     updraft_usable[updraft_usable < config.updraft_threshold] = 1e-10
#     #     usables.append(updraft_usable)
#     #     usables.append(standardize_matrix(updraft_usable))
#     # coeffs += usables
#     # print(len(for_tcmodel))
#     # print(np.min(for_tcmodel[3]), np.max(for_tcmodel[3]))
#     # cases = ('orograph', 'prob')
#     # cases = ('orograph', 'thermal', 'net', 'prob',
#     #         'uorograph', 'uthermal', 'unet')
#     cases = ('orograph',)
#     #         'Sorograph', 'Sthermal', 'Snet', 'Sprob')
#     # 'Uorograph', 'USorograph', 'Uthermal', 'USthermal', 'Unet', 'USnet')
#     tracks_all = []
#     pot_all = []
#     # _ = initiate_timer('\n')
#     for j, updraft in enumerate(coeffs):
#         tracks, potential = terrain_conductance_model(
#             run_id + cases[j],
#             updraft,
#             config.config.track_direction,
#             config.number_of_eagles,
#             config.region_of_eagle_entry,
#             config.type_of_eagle_entry,
#             config.dirn_restrict,
#             config.nu,
#             config.terrain_res
#         )
#         #     tracks_all.append(tracks)
#         #     pot_all.append(potential)
#         # for j, _ in enumerate(coeffs):
#         save_data(data_dir, run_id + cases[j] + '_tracks.pkl', tracks)
#         save_data(data_dir, run_id + cases[j] + '_potential.npy', potential)


# n_cpu = min(len(todo_indices), config.max_cores)
# start_time = initiate_timer(
#     '\n--- Running TCmodel: {0:d} requested, {1:d} exist, {2:d} cores\n'
#     .format(len(datetimes), len(done_datetimes), n_cpu))
# with mp.Pool(n_cpu) as pool:
#     out = pool.map(lambda i: run_tcmodel(
#         datetime_ids[i],
#         data_dirnames[i]
#     ), todo_indices)
# print_elapsed_time(start_time)

# # run the model
# # for i, dt in enumerate(datetimes):
# #     fext = dt.strftime(config.datetime_format)
# #     run_id = fext + '_' + config.config.track_direction + '_'
# #     if not os.path.exists(config.data_dir + run_id + '_orograph_tracks.pkl'):
# #         orograph = load_data(config.data_dir, fext + '_orograph.npy')
# #         deardoff = load_data(config.data_dir, fext + '_deardoff.npy')
# #         blheight = load_data(config.data_dir, fext + '_blheight.npy')
# #         thermal = compute_thermals_at_height(config.thermal_altitude,
# #                                              deardoff, blheight)
# #         save_data(config.data_dir, fext + '_thermal.npy', thermal)

# #         updrafts_for_tcmodel = [orograph, thermal, np.add(orograph, thermal)]
# #         usable_for_tcmodel = []
# #         for updraft in updrafts_for_tcmodel:
# #             updraft_usable = updraft.copy()
# #             updraft_usable[updraft_usable < config.updraft_threshold] = 1e-04
# #             usable_for_tcmodel.append(updraft_usable)
# #         updrafts_for_tcmodel += usable_for_tcmodel

# #         cases = ('orograph', 'thermal', 'combined',
# #                 'uorograph', 'uthermal', 'ucombined')

# #         # parallel
# #         cpu_tcmodel = min(len(cases), config.max_cores)
# #         with mp.Pool(cpu_tcmodel) as pool:
# #             out_tcmodel = pool.map(lambda j: terrain_conductance_model_serial(
# #                 run_id + cases[j],
# #                 updrafts_for_tcmodel[j],
# #                 config.config.track_direction,
# #                 track_parameters
# #             ), range(len(updrafts_for_tcmodel)))
# #         for j, lbl in enumerate(cases):
# #             save_data(config.data_dir, run_id + lbl + '_tracks.pkl',
# #                       out_tcmodel[j][0])
# #             save_data(config.data_dir, run_id + cases[j] + '_potential.npy',
# #                       out_tcmodel[j][1])

# # # serial
# # for j, updraft in enumerate(updrafts_for_tcmodel):
# #     tracks, potential = terrain_conductance_model(
# #         run_id + cases[j],
# #         updraft,
# #         config.config.track_direction,
# #         track_parameters,
# #         config.max_cores
# #     )
# #     save_data(config.data_dir, run_id + cases[j] + '_tracks.pkl',
# #               tracks)
# #     save_data(config.data_dir, run_id + cases[j] + '_potential.npy',
# #               potential.astype(np.float16))


# # %% Parallel implemetation
# # with multiprocessing.ThreadPool(config.max_cpu_usage) as pool:
# #     tc_output = pool.map(
# #         lambda i: terrain_conductance_model(
# #             load_data(config.data_dir, i + '_orographic.npy'),
# #             config.config.track_direction,
# #             track_parameters,
# #             grid_spacing,
# #             config.max_cpu_usage_tracks),
# #         wtk_run_ids)

# # for i, out in enumerate(tc_output):
# #     tracks, potential = out
# #     tc_run_id = generate_tcmodel_run_id(wtk_run_ids[i], config.config.track_direction)
# #     save_data(config.data_dir, tc_run_id + '_tracks.pkl', tracks)
# #     save_data(config.data_dir, tc_run_id + '_potential.npy', potential)
# #     count_mat = compute_count_matrix(grid_size, tracks)
# #     save_data(config.data_dir, tc_run_id + '_counts.npy', count_mat)

# # %%
# west_nodes = all_nodes - N
# west_nodes[np.in1d(all_nodes, west_bndry_nodes, invert=True)] = -1
# east_nodes = all_nodes + N
# east_nodes[np.in1d(all_nodes, east_bndry_nodes, invert=True)] = -1
# north_nodes = all_nodes + 1
# north_nodes[np.in1d(all_nodes, north_bndry_nodes, invert=True)] = -1
# south_nodes = all_nodes - 1
# south_nodes[np.in1d(all_nodes, south_bndry_nodes, invert=True)] = -1
# southwest_nodes = all_nodes - N - 1
# southwest_nodes[np.in1d(all_nodes, south_bndry_nodes, invert=True) or
#                    np.in1d(all_nodes, west_bndry_nodes, invert=True)] = -1
# northwest_nodes = all_nodes - N + 1
# northwest_nodes[np.in1d(all_nodes, north_bndry_nodes, invert=True) or
#                    np.in1d(all_nodes, west_bndry_nodes, invert=True)] = -1
# southeast_nodes = all_nodes + N - 1
# southeast_nodes[np.in1d(all_nodes, south_bndry_nodes, invert=True) or
#                    np.in1d(all_nodes, east_bndry_nodes, invert=True)] = -1
# northeast_nodes = all_nodes + N + 1
# northeast_nodes[np.in1d(all_nodes, north_bndry_nodes, invert=True) or
#                    np.in1d(all_nodes, east_bndry_nodes, invert=True)] = -1
# nearby_nodes = np.concatenate(
#     (north_nodes, west_nodes, south_nodes, east_nodes,
#     northwest_nodes, northeast_nodes, southwest_nodes, southeast_nodes))
