import time
from math import (sqrt)
from typing import List, Tuple

import numpy as np
import pathos.multiprocessing as multiprocessing
import scipy.signal as ssg
import scipy.sparse as ss


def get_elapsed_time(start):
    hours, rem = divmod(time.time() - start, 3600)
    mins, secs = divmod(rem, 60)
    if hours == 0:
        if mins == 0:
            xstr = " .. took {0:d} sec".format(int(secs) + 1)
        else:
            xstr = " .. took {0:d} min {1:d} sec".format(int(mins), int(secs))
    else:
        xstr = " .. took {0:d} hr {1:d} min".format(int(hours), int(mins))
    return xstr


def gen_node_numbering(grid_size: Tuple[int, int]):
    """Generate node numbering and outputs row index, column index and nearby
    nodes for a structured terrain elevation data"""

    # numbering starts from southwest, ends at north-east corner
    # print('Generating node numbering and identifying nearby nodes .. ',end="")
    node_rindex = np.empty(grid_size[0] * grid_size[1], dtype='int')
    node_cindex = np.empty(grid_size[0] * grid_size[1], dtype='int')
    nodes_nearby = -1 * np.ones((grid_size[0] * grid_size[1], 8), dtype='int')
    for j in range(grid_size[1]):
        for k, i in enumerate(range(grid_size[0]), start=grid_size[0] * j):
            node_rindex[k] = i
            node_cindex[k] = j
            if (k + 1) % grid_size[0] != 0:
                nodes_nearby[k, 0] = k + 1  # north
                if k < grid_size[0] * (grid_size[1] - 1):
                    nodes_nearby[k, 1] = k + grid_size[0] + 1  # north east
                if k >= grid_size[0]:
                    nodes_nearby[k, 7] = k - grid_size[0] + 1  # north west
            if k % grid_size[0] != 0:
                nodes_nearby[k, 4] = k - 1  # south
                if k < grid_size[0] * (grid_size[1] - 1):
                    nodes_nearby[k, 3] = k + grid_size[0] - 1  # southeast
                if k >= grid_size[0]:
                    nodes_nearby[k, 5] = k - grid_size[0] - 1  # southwest
            if k < grid_size[0] * (grid_size[1] - 1):
                nodes_nearby[k, 2] = k + grid_size[0]  # east
            if k >= grid_size[0]:
                nodes_nearby[k, 6] = k - grid_size[0]  # west
        # print(k,node_cindex[k],node_rindex[k],nodes_nearby[k,:])
    # print('done')
    return node_rindex, node_cindex, nodes_nearby


def get_boundary_values(
        bndry_condition: str,
        grid_size: Tuple[int, int]
):
    """Returns boundary nodes, energy values at those nodes, and starting nodes
    for initiating eagle migration"""

    # start_time = time.time()
    # print('Assigning migratory bndry conditions of {:s}'.format(
    #    bndry_condition), end=' ')
    bval = 100.
    north_bndry = np.array(
        [grid_size[0] * (x + 1) - 1 for x in range(grid_size[1])])
    south_bndry = np.array([grid_size[0] * x for x in range(grid_size[1])])
    west_bndry = np.array([x for x in range(grid_size[0])])
    east_bndry = np.array([(grid_size[1] - 1) * grid_size[0] + x
                           for x in range(grid_size[0])])
    if bndry_condition == 'mnorth':
        bndry_nodes = np.concatenate((north_bndry, south_bndry))
        bndry_energy = np.zeros((bndry_nodes.shape[0]))
        bndry_energy[np.size(bndry_nodes) // 2:] = bval
        starting_nodes = south_bndry + 1
    elif bndry_condition == 'msouth':
        bndry_nodes = np.concatenate((south_bndry, north_bndry))
        bndry_energy = np.zeros((bndry_nodes.shape[0]))
        bndry_energy[np.size(bndry_nodes) // 2:] = bval
        starting_nodes = north_bndry - 1
    elif bndry_condition == 'meast':
        bndry_nodes = np.concatenate((east_bndry, west_bndry))
        bndry_energy = np.zeros((bndry_nodes.shape[0]))
        bndry_energy[np.size(bndry_nodes) // 2:] = bval
        starting_nodes = west_bndry + grid_size[0]
    elif bndry_condition == 'mwest':
        bndry_nodes = np.concatenate((west_bndry, east_bndry))
        bndry_energy = np.zeros((bndry_nodes.shape[0]))
        bndry_energy[np.size(bndry_nodes) // 2:] = bval
        starting_nodes = east_bndry - grid_size[0]
    # elif bndry_condition == 'a2b_1':
    #     bndry_nodes = np.concatenate((west_bndry, east_bndry,
    #                                   south_bndry, north_bndry))
    #     bndry_energy = np.zeros((bndry_nodes.shape[0]))
    #     bndry_energy[np.size(bndry_nodes) // 2:] = bval
    #     starting_nodes = east_bndry - grid_size[0]
    else:
        raise NameError('Invalid boundary condition type!')
    # print(bndry_condition)
    inner_nodes = np.array([x for x in list(range(grid_size[0] * grid_size[1]))
                            if x not in bndry_nodes])
    # print_elapsed_time(start_time)
    return bndry_nodes, inner_nodes, bndry_energy, starting_nodes


def harmonic_mean(a: float, b: float) -> float:
    return 2. / (1. / a + 1 / b)


def compute_energy_at_all_nodes(
        run_id: str,
        cond_coeff: np.ndarray,
        bndry_nodes: np.ndarray,
        bndry_energy: np.ndarray,
        inner_nodes: np.ndarray
) -> np.ndarray:
    """ Create and solve a linear system for unknown energy values
    at inner nodes, returns the energy at all nodes"""

    # print('Assembling global conductivity matrix')
    start_time = time.time()
    Kij_row_index = []
    Kij_column_index = []
    Kij_value = []
    k = 0
    grid_size = cond_coeff.shape
    node_rindex, node_cindex, nodes_nearby = gen_node_numbering(grid_size)
    for i in range(0, grid_size[0] * grid_size[1]):
        # nearby_nodes_for_Kij = [x for r,x in enumerate(nodes_nearby[i,:])
        # if r%2==0 and x>=0]
        nearby_nodes_for_Kij = [x for r, x in enumerate(nodes_nearby[i, :])
                                if x >= 0]
        for nearby_node_for_Kij in nearby_nodes_for_Kij:
            coeff_a = cond_coeff[node_rindex[i], node_cindex[i]]
            coeff_b = cond_coeff[node_rindex[nearby_node_for_Kij],
                                 node_cindex[nearby_node_for_Kij]]
            if coeff_a != 0 and coeff_b != 0:
                mean_cond = harmonic_mean(coeff_a, coeff_b)
            else:
                mean_cond = 1e-10
            if abs(grid_size[0] + i - nearby_node_for_Kij) % 2 == 1:
                mean_cond /= sqrt(2.)
            Kij_row_index.append(i)
            Kij_column_index.append(nearby_node_for_Kij)
            Kij_value.append(mean_cond)
            k += 1
        z = len(nearby_nodes_for_Kij)
        if sum(Kij_value[k - z:k]) != 0:
            Kij_value[k - z:k] = [x / sum(Kij_value[k - z:k])
                                  for x in Kij_value[k - z:k]]
    Kij_global_coo = ss.coo_matrix((np.array(Kij_value),
                                    (np.array(Kij_row_index),
                                     np.array(Kij_column_index))),
                                   shape=(grid_size[0] * grid_size[1],
                                          grid_size[0] * grid_size[1]))
    Kij_global_csr = Kij_global_coo.tocsr()
    print('{0:s}: assembling global matrix {1:s}'.format(
        run_id, get_elapsed_time(start_time)), flush=True)
    # print('Sparsity level:', len(Kij_value),'/',grid_size[0]**2*grid_size[1]**2)
    start_time = time.time()
    Kij_inner_csr = Kij_global_csr[inner_nodes, :]
    Kij_inner_csc = Kij_inner_csr.tocoo().tocsc()
    Kij_inner_inner_csc = Kij_inner_csc[:, inner_nodes]
    Kij_inner_bndry_csc = Kij_inner_csc[:, bndry_nodes]
    b_vec = Kij_inner_bndry_csc.dot(bndry_energy)
    # inner_energy = ss.linalg.spsolve(Kij_inner_inner_csc, -b_vec)
    A_matrix = ss.eye(np.size(inner_nodes)).tocsc() - Kij_inner_inner_csc
    inner_energy = ss.linalg.spsolve(A_matrix, b_vec)
    # inner_energy,  istop, itn, r1nor = ss.linalg.lsqr(
    #     Kij_inner_inner_csc, -b_vec, damp=0.0, atol=1e-5, btol=1e-5)[:4]
    # print('{0:d}, {1:d}, {2:f}'.format(istop, itn, r1nor), flush=True)
    print('{0:s}: solving linear system {1:s}'.format(
        run_id, get_elapsed_time(start_time)), flush=True)
    global_energy = np.empty(grid_size[0] * grid_size[1])
    global_energy[inner_nodes] = inner_energy
    global_energy[bndry_nodes] = bndry_energy
    pot_energy = np.empty(grid_size)
    for i, (cur_xindex, cur_yindex) in enumerate(zip(node_rindex, node_cindex)):
        pot_energy[cur_xindex, cur_yindex] = global_energy[i]
    return pot_energy


# %% define static constants for eagle track generation
neighbour_deltas = []
neighbour_delta_norms_inv = np.empty((3, 3), dtype=np.float32)
center = (np.array(neighbour_delta_norms_inv.shape, dtype=np.int) - 1) // 2
for r in range(neighbour_delta_norms_inv.shape[0]):
    for c in range(neighbour_delta_norms_inv.shape[1]):
        delta = np.array([r, c], dtype=np.int) - center
        neighbour_deltas.append(delta)
        distance = np.linalg.norm(delta)
        neighbour_delta_norms_inv[r, c] = 1.0 / distance if distance > 0 else 0

neighbour_deltas_alt = neighbour_deltas[0:4] + neighbour_deltas[5:]
flat_neighbour_delta_norms_inv = list(neighbour_delta_norms_inv.flatten())
neighbour_delta_norms_inv_alt = np.array(
    flat_neighbour_delta_norms_inv[0:4] + flat_neighbour_delta_norms_inv[5:],
    dtype=np.float32)
delta_rows_alt = np.array([delta[0] for delta in neighbour_deltas_alt])
delta_cols_alt = np.array([delta[1] for delta in neighbour_deltas_alt])


def get_track_restrictions(dr: int, dc: int):
    Amat = np.zeros((3, 3), dtype=int)
    dr_mat = np.zeros((3, 3), dtype=int)
    dc_mat = np.zeros((3, 3), dtype=int)
    if abs(dr + dc % 2) == 1:
        if dr == 0:
            Amat[:, dc + 1] = 1
        else:
            Amat[dr + 1, :] = 1
    else:
        dr_mat[(dr + 1, 1), :] = 1
        dc_mat[:, (1, dc + 1)] = 1
        Amat = np.logical_and(dr_mat, dc_mat).astype(int)
    if dr == 0 and dc == 0:
        Amat[:, :] = 1
    Amat[1, 1] = 0
    return Amat.flatten()


def generate_eagle_track(
        i: int,
        cond_mat: np.ndarray,
        potential_mat: np.ndarray,
        start_loc: List[int],
        dirn_restrict: int,
        nu: float
):
    """ Generate an eagle track """

    num_rows, num_cols = cond_mat.shape
    scaled_neighbour_delta_norms_inv = 1. * neighbour_delta_norms_inv
    scaled_neighbour_delta_norms_inv = scaled_neighbour_delta_norms_inv.astype(
        np.float32)
    burnin = 5
    max_steps = 10 * max(num_rows, num_cols)
    dirn = [0, 0]
    previous_dirn = [0, 0]
    np.random.seed(i)
    position = start_loc.copy()
    trajectory = []
    trajectory.append(position)
    for k in range(max_steps):
        r, c = position
        if k > max_steps - 2:
            print('Maximum steps reached!', i, k, r, c)
        if k > burnin:
            if r <= 0 or r >= num_rows - 1 or c <= 0 or c >= num_cols - 1:
                # print('hit boundary!', k, ': ', r, c)
                break  # absorb if we hit a boundary
        else:
            if r == 0 or r == 1:
                r += 1
            elif r == num_rows - 1 or r == num_rows:
                r -= 1
            if c == 0 or c == 1:
                c += 1
            elif c == num_cols - 1 or c == num_cols:
                c -= 1
        position = [r, c]
        previous_dirn = np.copy(dirn)
        local_conductance = cond_mat[r - 1:r + 2, c - 1:c + 2]
        local_potential_energy = potential_mat[r - 1:r + 2, c - 1:c + 2]
        local_conductance = local_conductance.clip(min=1e-10)
        mc = 2.0 / (1.0 / local_conductance[1, 1] + 1.0 / local_conductance)
        q_diff = local_potential_energy[1, 1] - local_potential_energy
        if np.count_nonzero(q_diff) == 0:
            q_diff = 1. + np.random.rand(*q_diff.shape) * 1e-1
            # print('All potentials same!', i, k, r, c, mc)
        q_diff = np.multiply(q_diff, neighbour_delta_norms_inv)
        q = np.multiply(mc, q_diff)
        q = q.flatten()
        q -= np.min(q)
        q[4] = 0.
        if dirn_restrict > 0 and k > burnin:
            z = get_track_restrictions(*dirn)
            if dirn_restrict == 2:
                z = np.logical_and(z, get_track_restrictions(*previous_dirn))
            if sum(z) != 0:
                q_new = [x * float(y) for x, y in zip(q, z)]
                if np.sum(q_new) != 0:
                    q = q_new.copy()
        if np.sum(q) != 0:
            q /= np.sum(q)
            q = np.power(q, nu)
            q /= np.sum(q)
            chosen_index = np.random.choice(range(len(q)), p=q)
        else:
            # print('Sum(q) is zero!', i, k, r, c, q)
            chosen_index = np.random.choice(range(len(q)))
        dirn = neighbour_deltas[chosen_index]
        position = [x + y for x, y in zip(position, dirn)]
        trajectory.append(position)
    return np.array(trajectory, dtype=np.int16)


def generate_eagle_track_drw(
        i: int,
        cond_mat: np.ndarray,
        potential_mat: np.ndarray,
        start_loc: List[int],
        dirn_restrict: int,
        nu: float
):
    """ Generate an eagle track """

    num_rows, num_cols = cond_mat.shape
    scaled_neighbour_delta_norms_inv = 1. * neighbour_delta_norms_inv
    scaled_neighbour_delta_norms_inv = scaled_neighbour_delta_norms_inv.astype(
        np.float32)
    burnin = 5
    max_steps = 10 * max(num_rows, num_cols)
    dirn = [0, 0]
    previous_dirn = [0, 0]
    np.random.seed(i)
    position = start_loc.copy()
    trajectory = []
    trajectory.append(position)
    for k in range(max_steps):
        r, c = position
        if k > max_steps - 2:
            print('Maximum steps reached!', i, k, r, c)
        if k > burnin:
            if r <= 0 or r >= num_rows - 1 or c <= 0 or c >= num_cols - 1:
                # print('hit boundary!', k, ': ', r, c)
                break  # absorb if we hit a boundary
        else:
            if r == 0 or r == 1:
                r += 1
            elif r == num_rows - 1 or r == num_rows:
                r -= 1
            if c == 0 or c == 1:
                c += 1
            elif c == num_cols - 1 or c == num_cols:
                c -= 1
        position = [r, c]
        previous_dirn = np.copy(dirn)
        # local_conductance = cond_mat[r - 1:r + 2, c - 1:c + 2]
        # local_potential_energy = potential_mat[r - 1:r + 2, c - 1:c + 2]
        # local_conductance = local_conductance.clip(min=1e-10)
        # mc = 2.0 / (1.0 / local_conductance[1, 1] + 1.0 / local_conductance)
        # q_diff = local_potential_energy[1, 1] - local_potential_energy
        # if np.count_nonzero(q_diff) == 0:
        #     q_diff = 1. + np.random.rand(*q_diff.shape) * 1e-1
        #     #print('All potentials same!', i, k, r, c, mc)
        # q_diff = np.multiply(q_diff, neighbour_delta_norms_inv)
        # q = np.multiply(mc, q_diff)
        # q = q.flatten()
        # q -= np.min(q)
        # q[4] = 0.
        # if dirn_restrict > 0 and k > burnin:
        #     z = get_track_restrictions(*dirn)
        #     if dirn_restrict == 2:
        #         z = np.logical_and(z, get_track_restrictions(*previous_dirn))
        #     if sum(z) != 0:
        #         q_new = [x * float(y) for x, y in zip(q, z)]
        #         if np.sum(q_new) != 0:
        #             q = q_new.copy()
        q = [1., 1., 0., 1., 0., 0., 1., 1., 0.]
        if np.sum(q) != 0:
            q /= np.sum(q)
            q = np.power(q, nu)
            q /= np.sum(q)
            chosen_index = np.random.choice(range(len(q)), p=q)
        else:
            # print('Sum(q) is zero!', i, k, r, c, q)
            chosen_index = np.random.choice(range(len(q)))
        dirn = neighbour_deltas[chosen_index]
        position = [x + y for x, y in zip(position, dirn)]
        trajectory.append(position)
    return np.array(trajectory, dtype=np.int16)


def get_position_from_index(index: int, grid_size: tuple) -> List[int]:
    return [int(index % grid_size[1]), int(index // grid_size[1])]


def get_starting_indices(ntracks, bounds, res, entry_type='uniform'):
    base_ind = np.arange(int(bounds[0] / res), int(bounds[1] / res))
    if entry_type == 'uniform':
        idx = np.round(np.linspace(0, len(base_ind) - 1,
                                   ntracks % len(base_ind))).astype(int)
        target_ind = base_ind[idx]
        for _ in range(ntracks // len(base_ind)):
            target_ind = np.append(target_ind, base_ind)
        return target_ind
    elif entry_type == 'random':
        return np.random.randint(base_ind[0], base_ind[-1], ntracks)


def terrain_conductance_model_parallel(
        run_id: str,
        updraft: np.ndarray,
        bndry_cond: str,
        track_pars: List,
        n_cpu: int
):
    """ Runs terrain conductance model"""

    # unpack eagle track initiation settings
    start_time = time.time()
    t_start, t_end, t_dist, t_per, nu, dirn_restrict = track_pars

    # set up boundary conditions
    grid_size = np.shape(updraft)
    bndry_nodes, inner_nodes, bndry_energy, starting_nodes \
        = get_boundary_values(bndry_cond, grid_size)

    # computed potential energy
    potential_mat = compute_energy_at_all_nodes(
        run_id,
        updraft,
        bndry_nodes,
        bndry_energy,
        inner_nodes)

    # %% Generating eagle tracks
    start_index = np.repeat(
        np.arange(
            t_start,
            t_end,
            t_dist,
            dtype='int16'),
        t_per)
    total_tracks = np.size(start_index)
    # print('Generating {:d} eagle tracks .. '.format(total_tracks), end='')
    eagle_tracks = []
    updraft = updraft.astype(np.float32)
    potential_mat = potential_mat.astype(np.float32)
    with multiprocessing.Pool(n_cpu) as pool:
        eagle_tracks = pool.map(
            lambda i: generate_eagle_track(
                i,
                updraft,
                potential_mat,
                get_position_from_index(
                    starting_nodes[start_index[i]], grid_size),
                dirn_restrict,
                nu
            ),
            range(total_tracks))
    print('\nTCmodel: {0:s} .. took {1:.1f} secs'.format(
        run_id, get_elapsed_time(start_time)))
    return eagle_tracks, potential_mat.astype(np.float16)


def terrain_conductance_model_parallel_drw(
        run_id: str,
        updraft: np.ndarray,
        bndry_cond: str,
        track_pars: List,
        n_cpu: int
):
    """ Runs terrain conductance model"""

    # unpack eagle track initiation settings
    start_time = time.time()
    t_start, t_end, t_dist, t_per, nu, dirn_restrict = track_pars

    # set up boundary conditions
    grid_size = np.shape(updraft)
    bndry_nodes, inner_nodes, bndry_energy, starting_nodes \
        = get_boundary_values(bndry_cond, grid_size)

    potential_mat = np.zeros(updraft.shape)
    # computed potential energy
    # potential_mat = compute_energy_at_all_nodes(
    #     run_id,
    #     updraft,
    #     bndry_nodes,
    #     bndry_energy,
    #     inner_nodes)

    # %% Generating eagle tracks
    start_index = np.repeat(
        np.arange(
            t_start,
            t_end,
            t_dist,
            dtype='int16'),
        t_per)
    total_tracks = np.size(start_index)
    # print('Generating {:d} eagle tracks .. '.format(total_tracks), end='')
    eagle_tracks = []
    updraft = updraft.astype(np.float32)
    potential_mat = potential_mat.astype(np.float32)
    with multiprocessing.Pool(n_cpu) as pool:
        eagle_tracks = pool.map(
            lambda i: generate_eagle_track_drw(
                i,
                updraft,
                potential_mat,
                get_position_from_index(
                    starting_nodes[start_index[i]], grid_size),
                dirn_restrict,
                nu
            ),
            range(total_tracks))
    print('\nTCmodel: {0:s} .. took {1:.1f} secs'.format(
        run_id, get_elapsed_time(start_time)))
    return eagle_tracks, potential_mat.astype(np.float16)


def terrain_conductance_model_serial(
        run_id: str,
        updraft: np.ndarray,
        bndry_cond: str,
        track_pars: List
):
    """ Runs terrain conductance model"""

    # unpack eagle track initiation settings
    start_time = time.time()
    t_start, t_end, t_dist, t_per, nu, dirn_restrict = track_pars
    print('TCmodel: {0:s}'.format(run_id), flush=True)

    # set up boundary conditions
    grid_size = np.shape(updraft)
    bndry_nodes, inner_nodes, bndry_energy, starting_nodes \
        = get_boundary_values(bndry_cond, grid_size)

    # computed potential energy
    potential_mat = compute_energy_at_all_nodes(
        run_id,
        updraft,
        bndry_nodes,
        bndry_energy,
        inner_nodes)

    # %% Generating eagle tracks
    start_index = np.repeat(
        np.arange(
            t_start,
            t_end,
            t_dist,
            dtype='int16'),
        t_per)
    total_tracks = np.size(start_index)
    # print('Generating {:d} eagle tracks .. '.format(total_tracks), end='',
    #      flush=True)
    eagle_tracks = []
    # updraft = updraft.astype(np.float32)
    # potential_mat = potential_mat.astype(np.float32)
    for i in range(total_tracks):
        eagle_tracks.append(generate_eagle_track(
            i,
            updraft,
            potential_mat,
            get_position_from_index(starting_nodes[start_index[i]], grid_size),
            dirn_restrict,
            nu
        ))
    print('TCmodel: {0:s} .. took {1:.1f} secs'.format(
        run_id, get_elapsed_time(start_time)))
    return eagle_tracks, potential_mat.astype(np.float32)


def terrain_conductance_model(
        run_id: str,
        coeff: np.ndarray,
        bndry_cond: str,
        total_tracks: int,
        bounds: Tuple[float, float],
        entry_type: str,
        dirn_restrict: int,
        nu: float,
        res: float
):
    """ Runs terrain conductance model"""

    # unpack eagle track initiation settings
    start_time = time.time()
    # print('TCmodel: {0:s}'.format(run_id), flush=True)

    # set up boundary conditions
    grid_size = np.shape(coeff)
    bndry_nodes, inner_nodes, bndry_energy, starting_nodes \
        = get_boundary_values(bndry_cond, grid_size)

    # computed potential energy
    potential_mat = compute_energy_at_all_nodes(
        run_id,
        coeff,
        bndry_nodes,
        bndry_energy,
        inner_nodes)

    # Generating eagle tracks
    start_time = time.time()
    starting_indices = get_starting_indices(
        total_tracks, bounds, res, entry_type)
    eagle_tracks = []
    coeff = coeff.astype(np.float32)
    potential_mat = potential_mat.astype(np.float32)
    for i in range(total_tracks):
        eagle_tracks.append(generate_eagle_track(
            i,
            coeff,
            potential_mat,
            get_position_from_index(
                starting_nodes[starting_indices[i]], grid_size),
            dirn_restrict,
            nu
        ))
    print('{0:s}: generating eagle tracks {1:s}'.format(
        run_id, get_elapsed_time(start_time)), flush=True)
    return eagle_tracks, potential_mat.astype(np.float32)


def compute_count_matrix(
        mat_shape: Tuple[int, int],
        tracks: List[np.ndarray]
) -> np.ndarray:
    """Computes count matrix that contains the number of
    times eagles were present at each grid point"""

    # print('Computing counts for {0:d} tracks'.format(len(tracks)), end="")
    # start_time = time.time()
    A = np.zeros(mat_shape, dtype=np.int16)
    for track in tracks:
        for move in track:
            A[move[0], move[1]] += 1
    # print_elapsed_time(start_time)
    return A.astype(np.int16)


# def compute_presence_probability(
#     A: np.ndarray,
#     radius: int
# ) -> np.ndarray:
#     """ Smothens a matrix using 2D covolution of the circular kernel matrix
#     with the givem atrix """

#     kernel = np.zeros((2 * radius + 1, 2 * radius + 1))
#     y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
#     mask2 = x**2 + y**2 <= radius**2
#     kernel[mask2] = 1
#     Asmooth = ssg.convolve2d(A, kernel, mode='same')
#     Asmooth /= np.amax(Asmooth)
#     return Asmooth.astype(np.float32)

def get_minmax_indices(
        xgrid: np.ndarray,
        ygrid: np.ndarray,
        extent: Tuple[float, float, float, float]
):
    idxmin = (np.abs(xgrid - extent[0])).argmin()
    idymin = (np.abs(ygrid - extent[2])).argmin()
    idxmax = (np.abs(xgrid - extent[1])).argmin()
    idymax = (np.abs(ygrid - extent[3])).argmin()
    return [idxmin, idxmax, idymin, idymax]


def compute_presence(
        counts: np.ndarray,
        extent: Tuple[float, float, float, float],
        radius: int
) -> np.ndarray:
    """ Smothens a matrix using 2D covolution of the circular kernel matrix
    with the givem matrix """

    trimmed = counts[extent[2]:extent[3], extent[0]:extent[1]]
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1))
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask2 = x ** 2 + y ** 2 <= radius ** 2
    kernel[mask2] = 1
    Asmooth = ssg.convolve2d(trimmed, kernel, mode='same')
    # Asmooth /= np.amax(Asmooth)
    return Asmooth.astype(np.float32)


# %% Junk
# print(A[0,:])

# # %% Generate eagle track
# def generate_eagle_track(
#         conductance_map,
#         potential_energy_map,
#         start,  # (row, col) np array
#         grid_spacing,
#         nu,
#         max_steps=15000
# ):
#     """ Generate an eagle track """

#     position = np.copy(start)
#     trajectory = [position]
#     num_rows, num_cols = conductance_map.shape

#     scaled_neighbour_delta_norms_inv = grid_spacing * neighbour_delta_norms_inv
#     scaled_neighbour_delta_norms_inv = scaled_neighbour_delta_norms_inv.astype(
#         np.float32)

#     for step_number in range(max_steps):
#         position_tuple = tuple(position)
#         r = position_tuple[0]
#         c = position_tuple[1]

#         if r <= 0 or r >= num_rows - 1 or c <= 0 or c >= num_cols - 1:
#             break  # absorb if we hit a boundary

#         local_conductance = conductance_map[r - 1:r + 2, c - 1:c + 2]
#         local_potential_energy = potential_energy_map[r - 1:r + 2, c - 1:c + 2]

#         q = 2.0 / (1.0 / local_conductance[1, 1] + 1.0 / local_conductance)
#         q *= (local_potential_energy[1, 1] -
#               local_potential_energy) * scaled_neighbour_delta_norms_inv

#         if np.max(q) < 0:
#             print('hit well')
#             break  # terminate if we're in a well

#         # possibly smooth eagle tracks by penalizing changing velocities
#         # test out different action probability schemes

#         # q = q.flatten()
#         # q -= np.max(q)  # prevents overflow when exp()'ing
#         # q /= 1e-1 #boltzmann_temperature
#         # p = np.exp(q)
#         # q[4] = 0.0
#         # p /= np.sum(p)

#         q = q.flatten()
#         q -= np.min(q)
#         q = np.power(q, nu)
#         q[4] = 0.0
#         q /= np.sum(q)

#         chosen_index = np.random.choice(range(len(q)), p=q)
#         position = position + neighbour_deltas[chosen_index]
#         trajectory.append(position)
#     return np.array(trajectory)


# # %% Generate eagle track
# def generate_eagle_track_alternate(
#         conductance_map,
#         potential_energy_map,
#         start,  # (row, col) np array
#         grid_spacing,
#         nu,
#         max_steps=15000
# ):
#     """ Generate an eagle track """
#     position = np.copy(start)
#     trajectory = [position]
#     delta_norms_inv_scaled = grid_spacing * neighbour_delta_norms_inv_alt

#     num_rows, num_cols = conductance_map.shape

#     for step_number in range(max_steps):
#         position_tuple = tuple(position)

#         if position_tuple[0] <= 0 or position_tuple[0] >= num_rows - 1 or \
#                 position_tuple[1] <= 0 or position_tuple[1] >= num_cols - 1:
#             break  # terminate if we hit a boundary

#         # current_conductance = conductance_map[position_tuple]
#         current_conductance_inv = 1.0 / conductance_map[position_tuple]
#         current_potential_energy = potential_energy_map[position_tuple]

#         # candidate_positions = [position + delta for delta in neighbour_deltas]
#         candidate_rows = delta_rows_alt + position_tuple[0]
#         candidate_cols = delta_cols_alt + position_tuple[1]
#         candidate_conductance = conductance_map[candidate_rows, candidate_cols]
#         candidate_potential_energy = potential_energy_map[candidate_rows, candidate_cols]

#         q = 2.0 / (current_conductance_inv + 1.0 / candidate_conductance)
#         q *= (current_potential_energy - candidate_potential_energy) * \
#             delta_norms_inv_scaled

#         if np.max(q) < 0:
#             print('hit well')
#             break  # terminate if we're in a well

#         # q -= np.max(q)  # prevents overflow when exp()'ing
#         # q /= boltzmann_temperature
#         # p = np.exp(q)
#         # p /= np.sum(p)

#         q -= np.min(q)
#         q = np.power(q, nu)
#         q /= np.sum(q)

#         chosen_index = np.random.choice(range(len(q)), p=q)
#         position = position + neighbour_deltas_alt[chosen_index]
#         trajectory.append(position)

#     return np.array(trajectory)


# # %% Generate eagle track
# def generate_eagle_track_old(
#         cond_coeff,  # 1000x1000 float array
#         potential_mat,  # 1000x1000 float array
#         start_node,  # 50999
#         grid_spacing,  # .05
#         smart_fac,  # 1.0
#         max_steps,  # 15000
#         node_rindex,  # 1M int array
#         node_cindex,  # 1M int array
#         nodes_nearby,  # 1Mx8 int array
# ):
#     """ Generate an eagle track """

#     kk = 0
#     curn = start_node
#     track_not_converged = True
#     eagle_track = []
#     while track_not_converged:
#         eagle_track.append([node_rindex[curn], node_cindex[curn]])
#         try_dirs, tryns = zip(*[(j, x) for j, x in enumerate(nodes_nearby[curn, :])
#                                 if x >= 0])
#         try_qij = np.zeros(len(tryns))
#         for k, (try_dir, tryn) in enumerate(zip(try_dirs, tryns)):
#             try_Kij = 2 / (1 / cond_coeff[node_rindex[curn], node_cindex[curn]]
#                            + 1 / cond_coeff[node_rindex[tryn], node_cindex[tryn]])
#             try_qij[k] = try_Kij * (potential_mat[node_rindex[curn], node_cindex[curn]]
#                                     - potential_mat[node_rindex[tryn],
#                                                     node_cindex[tryn]]) / grid_spacing
#             if try_dir % 2 == 1:
#                 try_qij[k] = try_qij[k] / sqrt(2.0)
#                 # print(k,try_dir,tryn,try_Kij,try_qij[k])
#         if np.all(try_qij <= 0.):
#             break
#         try_qij = (try_qij - min(try_qij)) / sum(try_qij - min(try_qij))
#         try_qij = try_qij ** smart_fac / sum(try_qij ** smart_fac)
#         curn = tryns[np.random.choice(len(tryns), 1, p=try_qij).item()]
#         kk += 1
#         if kk > max_steps:
#             track_not_converged = False
#     return np.array(eagle_track)


# # %% computes presence map (kde approximated)
# def compute_presence_map(
#         eagle_tracks, xy_bnd, xgrid, ygrid
# ):
#     """ Computes KDE approximation of eagle presence probability"""

#     grid_size = (xgrid.size, ygrid.size)
#     node_rindex, node_cindex, _ = gen_node_numbering(grid_size)
#     xmesh, ymesh = np.mgrid[xy_bnd[0]:xy_bnd[1]:50j, xy_bnd[2]:xy_bnd[3]:50j]
#     positions = np.vstack([xmesh.ravel(), ymesh.ravel()])
#     x_values = np.array([])
#     y_values = np.array([])
#     for eagle_track in eagle_tracks:
#         x_values = np.concatenate((x_values, xgrid[eagle_track[:, 1]]))
#         y_values = np.concatenate((y_values, ygrid[eagle_track[:, 0]]))
#     values = np.vstack([x_values, y_values])
#     # kernel = st.gaussian_kde(values,bw_method=0.06)
#     kernel = st.gaussian_kde(values)
#     # print(kernel.factor)
#     Z = np.reshape(kernel(positions), xmesh.shape)
#     return Z.T


def construct_sparse_mat_for_spsolve(ind, cond, nodes_nearby,
                                     node_rindex, node_cindex):
    rindex = []
    cindex = []
    value = []
    # nearby_nodes_for_Kij = [x for r,x in enumerate(nodes_nearby[i,:])
    # if r%2==0 and x>=0]
    k = 0
    for i in range(*ind):
        rel_i = i - ind[0]
        nearby_nodes_for_Kij = [x for r, x in enumerate(nodes_nearby[rel_i, :])
                                if x >= 0]
        for nearby_node_for_Kij in nearby_nodes_for_Kij:
            rindex.append(i)
            cindex.append(nearby_node_for_Kij)
            value.append(2 / (1 / cond[node_rindex[i], node_cindex[i]]
                              + 1 / cond[node_rindex[nearby_node_for_Kij],
                                         node_cindex[nearby_node_for_Kij]]))
            k += 1
        ntemp = len(nearby_nodes_for_Kij)
        value[k - ntemp:k] = [x / sum(value[k - ntemp:k])
                              for x in value[k - ntemp:k]]
    return rindex, cindex, value


# # %% Create and solve a linear system for unknown energy values at inner nodes
def compute_energy_at_all_nodes_parallel(
        cond_mat,
        bndry_nodes,
        bndry_energy,
        inner_nodes,
        n_cpu
):
    """ Create and solve a linear system for unknown energy values
    at inner nodes, returns the energy at all nodes"""

    # print('Assembling global conductivity matrix', end=" ")
    # start_time = time.time()

    # print('Assembling global conductivity matrix .. ', end=" ")
    grid_size = cond_mat.shape
    node_rindex, node_cindex, nodes_nearby = gen_node_numbering(grid_size)
    Kij_value = []
    Kij_rindex = []
    Kij_cindex = []
    # with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
    #     eagle_tracks = pool.map(
    #         lambda i:
    # cond_mat = cond_mat.astype(np.float32)
    # node_rindex = node_rindex.astype(np.int32)
    # node_cindex = node_cindex.astype(np.int32)
    # print(node_rindex.dtype, node_cindex.dtype)
    # print(cond_mat.nbytes / 1024**2, nodes_nearby.nbytes / 1024**2,
    #      node_rindex.nbytes / 1024**2, node_cindex.nbytes / 1024**2)

    n_chunks = grid_size[0] * grid_size[1] // n_cpu
    increments = list(range(0, grid_size[0] * grid_size[1], n_chunks))
    increments.append(grid_size[0] * grid_size[1])
    chunk_list = [(increments[i], increments[i + 1])
                  for i in range(len(increments) - 1)]
    # print(n_chunks, increments, chunk_list)

    with multiprocessing.Pool() as pool:
        out = pool.map(
            lambda x: construct_sparse_mat_for_spsolve(x,
                                                       cond_mat,
                                                       nodes_nearby[x[0]:x[1], :],
                                                       node_rindex,
                                                       node_cindex),
            chunk_list)
    # print(out[-1])
    for (rindex, cindex, value) in out:
        Kij_rindex.extend(rindex)
        Kij_cindex.extend(cindex)
        Kij_value.extend(value)
    # for i in range(0, grid_size[0] * grid_size[1]):
    #     rindex, cindex, value = construct_sparse_mat_for_spsolve(i, cond_mat,
    #                                                              nodes_nearby,
    #                                                              node_rindex,
    #                                                              node_cindex)
    #     # Kij_rindex = np.concatenate((Kij_rindex, rindex))
    #     # Kij_cindex = np.concatenate((Kij_cindex, cindex))
    #     # Kij_value = np.concatenate((Kij_value, value))
    #     Kij_rindex.extend(rindex)
    #     Kij_cindex.extend(cindex)
    #     Kij_value.extend(value)
    Kij_global_coo = ss.coo_matrix((np.array(Kij_value),
                                    (np.array(Kij_rindex), np.array(Kij_cindex))),
                                   shape=(grid_size[0] * grid_size[1],
                                          grid_size[0] * grid_size[1]))
    # print(Kij_global_coo.dtype)
    # print_elapsed_time(start_time)
    # print('Solving for unknown energy values', end=' ')
    Kij_global_csr = Kij_global_coo.tocsr()
    # print('Sparsity level:', len(Kij_value),'/',grid_size[0]**2*grid_size[1]**2)
    Kij_inner_csr = Kij_global_csr[inner_nodes, :]
    Kij_inner_csc = Kij_inner_csr.tocoo().tocsc()
    Kij_inner_inner_csc = Kij_inner_csc[:, inner_nodes]
    Kij_inner_bndry_csc = Kij_inner_csc[:, bndry_nodes]
    b_vec = Kij_inner_bndry_csc.dot(bndry_energy)
    A_matrix = ss.eye(np.size(inner_nodes)).tocsc() - Kij_inner_inner_csc
    # print_elapsed_time(start_time)
    # print(A_matrix.dtype, b_vec.dtype)
    inner_energy = ss.linalg.spsolve(A_matrix, b_vec)
    # print_elapsed_time(start_time)
    # Reassemble the global solution for energy
    global_energy = np.empty(grid_size[0] * grid_size[1])
    global_energy[inner_nodes] = inner_energy
    global_energy[bndry_nodes] = bndry_energy
    potential_mat = np.empty(grid_size)
    for i, (cur_xindex, cur_yindex) in enumerate(zip(node_rindex, node_cindex)):
        potential_mat[cur_xindex, cur_yindex] = global_energy[i]
    # print_elapsed_time(start_time)
    # print(potential_mat.dtype)
    return potential_mat
