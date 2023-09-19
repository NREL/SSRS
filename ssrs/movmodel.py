""" Module for implementing fluid-flow based movement model """

from math import (floor, ceil, sqrt)
from typing import List, Tuple, Optional
import numpy as np
import scipy.signal as ssg
import scipy.sparse as ss


class MovModel:
    """ Class for fluid-flow based model """

    def __init__(
        self,
        move_dirn: float,
        grid_shape: Tuple[int, int]
    ):
        self.move_dirn = move_dirn
        self.grid_shape = grid_shape

    def get_boundary_nodes(self):
        """ returns boundary nodes and potential for given direction of
        movement """
        nrow, ncol = self.grid_shape
        north_bnodes = np.array([nrow * (x + 1) - 1 for x in range(ncol)])
        south_bnodes = np.array([nrow * x for x in range(ncol)])
        west_bnodes = np.array(list(range(1, nrow - 1)))
        east_bnodes = np.array(
            [(ncol - 1) * nrow + x for x in range(1, nrow - 1)])
        mov_angle = self.move_dirn % 90.
        mov_quad = (self.move_dirn % 360) // 90.
        col_len = round(ncol * mov_angle / 90.)
        row_len = round(nrow * mov_angle / 90.)
        if mov_quad == 0:
            low_nodes = np.concatenate(
                (north_bnodes[col_len:], east_bnodes[nrow - row_len:]))
            high_nodes = np.concatenate(
                (south_bnodes[:ncol - col_len], west_bnodes[:row_len]))
        elif mov_quad == 1:
            low_nodes = np.concatenate(
                (south_bnodes[ncol - col_len:], east_bnodes[:nrow - row_len]))
            high_nodes = np.concatenate(
                (north_bnodes[:col_len], west_bnodes[row_len:]))
        elif mov_quad == 2:
            low_nodes = np.concatenate(
                (south_bnodes[:ncol - col_len], west_bnodes[:row_len]))
            high_nodes = np.concatenate(
                (north_bnodes[col_len:], east_bnodes[nrow - row_len:]))
        elif mov_quad == 3:
            high_nodes = np.concatenate(
                (south_bnodes[ncol - col_len:], east_bnodes[:nrow - row_len]))
            low_nodes = np.concatenate(
                (north_bnodes[:col_len], west_bnodes[row_len:]))
        bndry_nodes = np.concatenate((low_nodes, high_nodes))
        bndry_potential = np.zeros((bndry_nodes.size))
        bndry_potential[bndry_nodes.size // 2:] = 1000.
        return bndry_nodes, bndry_potential

    def assemble_sparse_linear_system(self):
        """ returns the row and column index of sparse linear system"""
        nrow, ncol = self.grid_shape
        row_index = []
        col_index = []
        facs = []
        for i in range(0, nrow * ncol):
            if (i + 1) % nrow == 0:  # north bndry
                nearby = [i + nrow, i + nrow - 1,
                          i - 1, i - nrow - 1, i - nrow]
            elif i % nrow == 0:  # south bndry
                nearby = [i - nrow, i - nrow + 1,
                          i + 1, i + nrow + 1, i + nrow]
            else:  # otherwise
                nearby = [i - nrow, i - nrow + 1, i + 1, i + nrow + 1, i + nrow,
                          i + nrow - 1, i - 1, i - nrow - 1]
            nearby = [x for x in nearby if nrow * ncol > x >= 0]
            row_index.extend([i for _ in range(0, len(nearby))])
            col_index.extend(nearby)
            facs.extend([sqrt(2.) if i %
                        2 else 1. for i in range(len(nearby))])
        row_index = np.array(row_index, dtype='u4')
        col_index = np.array(col_index, dtype='u4')
        facs = np.array(facs, dtype='f4')
        # print('Sparsity: {0:.3f} %'.format(100 * (1 - facs.size / (nrow * ncol)**2)))
        return row_index, col_index, facs

    @ classmethod
    def solve_sparse_linear_system(
        cls,
        conductivity: np.ndarray,
        bnodes: np.ndarray,
        benergy: np.ndarray,
        row_inds: np.ndarray,
        col_inds: np.ndarray,
        facs: np.ndarray
    ) -> np.ndarray:
        """ Solves the linear system for unknown energy values at inner nodes,
        returns the energy at all nodes (inner + bndry)"""

        nrow, ncol = conductivity.shape
        vals = []
        for r, c, fac in zip(row_inds, col_inds, facs):
            conductivity_a = conductivity[r % nrow, r // nrow]
            conductivity_b = conductivity[c % nrow, c // nrow]
            vals.append(harmonic_mean(conductivity_a,
                        conductivity_b, 1e-08) / fac)
        g_coo = ss.coo_matrix((np.array(vals),
                               (np.array(row_inds),
                              np.array(col_inds))), shape=(nrow * ncol, nrow * ncol))
        g_csr = g_coo.tocsr()
        g_csr.data = g_csr.data / np.repeat(np.add.reduceat(g_csr.data,
                                                            g_csr.indptr[:-1]),
                                            np.diff(g_csr.indptr))
        inodes = np.setdiff1d(np.arange(0, nrow * ncol),
                              bnodes, assume_unique=True)
        g_inner_csr = g_csr[inodes, :]
        g_inner_csc = g_inner_csr.tocoo().tocsc()
        g_inner_inner_csc = g_inner_csc[:, inodes]
        g_inner_bndry_csc = g_inner_csc[:, bnodes]
        b_vec = g_inner_bndry_csc.dot(benergy)
        a_matrix = ss.eye(np.size(inodes)).tocsc() - g_inner_inner_csc
        ienergy = ss.linalg.spsolve(a_matrix, b_vec)
        global_energy = np.empty(nrow * ncol)
        global_energy[inodes] = ienergy
        global_energy[bnodes] = benergy
        pot_energy = np.empty((nrow, ncol))
        for i in range(nrow * ncol):
            pot_energy[i % nrow, i // nrow] = global_energy[i]
        return pot_energy.astype(np.float32)


# define static constants for eagle track generation
neighbour_deltas = []
neighbour_delta_norms_inv = np.empty((3, 3), dtype=np.float32)
center = (np.array(neighbour_delta_norms_inv.shape, dtype=int) - 1) // 2
for r in range(neighbour_delta_norms_inv.shape[0]):
    for c in range(neighbour_delta_norms_inv.shape[1]):
        delta = np.array([r, c], dtype=int) - center
        neighbour_deltas.append(delta)
        distance = np.linalg.norm(delta)
        neighbour_delta_norms_inv[r, c] = 1.0 / \
            distance if distance > 0 else 0


def get_starting_indices(
    ntracks: int,
    sbounds: List[float],
    stype: str,
    twidth: Tuple[float, float],
    tres: float
) -> List[int]:
    """ get starting indices of eagle tracks """

    if (sbounds[1] < sbounds[0] or sbounds[3] < sbounds[2] or
        sbounds[0] < 0. or sbounds[2] < 0. or sbounds[1] > twidth[0] or
            sbounds[3] > twidth[1]):
        raise ValueError('track_start_region incompatible with terrain_width!')
    res_km = tres / 1000.
    xind_max = ceil(twidth[0] / res_km)
    yind_max = ceil(twidth[1] / res_km)
    xind_low = min(max(floor(sbounds[0] / res_km) - 1, 1), xind_max - 2)
    xind_upp = max(min(ceil(sbounds[1] / res_km), xind_max - 1), 2)
    yind_low = min(max(floor(sbounds[2] / res_km) - 1, 1), yind_max - 2)
    yind_upp = max(min(ceil(sbounds[3] / res_km), yind_max - 1), 2)
    xmesh, ymesh = np.mgrid[xind_low:xind_upp, yind_low:yind_upp]
    base_inds = np.vstack((np.ravel(ymesh), np.ravel(xmesh)))
    base_count = base_inds.shape[1]
    if stype == 'structured':
        idx = np.round(np.linspace(0, base_count - 1, ntracks % base_count))
        if ntracks > base_count:
            start_inds = np.tile(base_inds, (1, ntracks // base_count))
            start_inds = np.hstack(
                (start_inds, start_inds[:, idx.astype(int)]))
        else:
            start_inds = base_inds[:, idx.astype(int)]
    elif stype == 'random':
        idx = np.random.randint(0, base_count, ntracks)
        start_inds = base_inds[:, idx]
    else:
        raise ValueError((f'Model:Invalid sim_start_type of {stype}\n'
                          'Options: structured, random'))
    start_inds = start_inds.astype(int)
    return start_inds[0, :], start_inds[1, :]


def get_track_restrictions(dr: int, dc: int):
    """ Returns updated movement probabilities based on previous moves"""
    a_mat = np.zeros((3, 3), dtype=int)
    dr_mat = np.zeros((3, 3), dtype=int)
    dc_mat = np.zeros((3, 3), dtype=int)
    if abs(dr + dc % 2) == 1:
        if dr == 0:
            a_mat[:, dc + 1] = 1
        else:
            a_mat[dr + 1, :] = 1
    else:
        dr_mat[(dr + 1, 1), :] = 1
        dc_mat[:, (1, dc + 1)] = 1
        a_mat = np.logical_and(dr_mat, dc_mat).astype(int)
    if dr == 0 and dc == 0:
        a_mat[:, :] = 1
    a_mat[1, 1] = 0
    return a_mat.flatten()


def move_away_from_boundary(row, col, num_rows, num_cols):
    """ move simulated eagle away from edges """
    new_col = col
    new_row = row
    if row <= 1:
        new_row = row + 2
    elif row >= num_rows - 2:
        new_row = row - 2
    if col <= 0:
        new_col = col + 2
    elif col >= num_cols - 2:
        new_col = col - 2
    return new_row, new_col


def generate_move_probabilities(
    in_probs: np.ndarray,
    move_dirn: float,
    nu_par: float,
    dir_bool: np.ndarray
):
    """ create move probabilities from a 1d array of values"""
    out_probs = np.asarray(in_probs.copy())
    if np.isnan(out_probs).any():
        print('NANs in move probabilities!')
        out_probs = get_directional_probs(move_dirn * np.pi / 180.)
    out_probs = out_probs.clip(min=0.)
    out_probs[4] = 0.
    out_probs = [ix * float(iy) for ix, iy in zip(out_probs, dir_bool)]
    if np.count_nonzero(out_probs) == 0:
        out_probs = get_directional_probs(move_dirn * np.pi / 180.)
        #out_probs = np.random.rand(len(out_probs))
    out_probs[4] = 0.
    out_probs = [ix * float(iy) for ix, iy in zip(out_probs, dir_bool)]
    if np.count_nonzero(out_probs) == 0:
        out_probs = get_directional_probs(move_dirn * np.pi / 180.)
    out_probs /= np.sum(out_probs)
    out_probs = np.power(out_probs, nu_par)
    out_probs /= np.sum(out_probs)
    return out_probs


def get_directional_probs(theta: float) -> np.ndarray:
    """ Returns a dirction array based on angle"""
    dir_mat = np.zeros((3, 3))
    dir_mat[0, :] = [np.cos(np.pi / 4 + theta), np.cos(theta),
                     np.cos(7 * np.pi / 4 + theta)]
    dir_mat[1, :] = [np.cos(np.pi / 2 + theta), 0,
                     np.cos(3 * np.pi / 2 + theta)]
    dir_mat[2, :] = [np.cos(3 * np.pi / 4 + theta), np.cos(np.pi + theta),
                     np.cos(5 * np.pi / 4 + theta)]
    dir_mat[dir_mat < 0.01] = 0.
    return np.flipud(dir_mat.clip(min=0.)).flatten()


def get_harmonic_mean(in_first, in_second):
    return 2.0 / (1.0 / in_first + 1.0 / in_second)


def generate_simulated_tracks(
        move_dirn: float,
        start_location: Tuple[int, int],
        grid_shape: Tuple[int, int],
        memory_parameter: int = 1,
        scaling_parameter: float = 1.,
        updraft_field: Optional[np.ndarray] = None,
        potential_field: Optional[np.ndarray] = None
):
    """ Generate an eagle track """

    num_rows, num_cols = grid_shape
    burnin_length = int(min(num_rows, num_cols) / 10)
    max_moves = num_rows / 2 * num_cols / 2
    direction = [0, 0]
    directions = []
    directions.append(direction)
    position = start_location.copy()
    trajectory = []
    trajectory.append(position)
    k = 0
    while k < max_moves:
        row, col = position
        if k > burnin_length:  # move away from boundary in initial steps
            if not (0 < row < num_rows - 1 and 0 < col < num_cols - 1):
                break
        else:
            row, col = move_away_from_boundary(row, col, num_rows, num_cols)
        move_probs = np.ones_like(neighbour_delta_norms_inv)
        if updraft_field is not None:
            local_updraft = updraft_field[row - 1:row + 2, col - 1:col + 2]
            local_updraft = local_updraft.clip(min=1e-06)
            mean_change = get_harmonic_mean(local_updraft[1, 1], local_updraft)
            move_probs = np.multiply(move_probs, mean_change)
        else:
            move_probs = get_directional_probs(move_dirn * np.pi / 180.)
        if potential_field is not None:
            local_potential = potential_field[row - 1:row + 2, col - 1:col + 2]
            potential_diff = local_potential[1, 1] - local_potential
            potential_diff = np.multiply(potential_diff,
                                         neighbour_delta_norms_inv)
            move_probs = np.multiply(move_probs, potential_diff)
        move_probs = move_probs.flatten()
        dir_bool = get_track_restrictions(0, 0)
        for idirn in directions[-memory_parameter:]:
            dir_bool = np.logical_and(get_track_restrictions(*idirn), dir_bool)
        move_probs = generate_move_probabilities(move_probs, move_dirn,
                                                 scaling_parameter, dir_bool)
        chosen_index = np.random.choice(range(len(move_probs)), p=move_probs)
        dirn = neighbour_deltas[chosen_index]
        position = [x + y for x, y in zip([row, col], dirn)]
        trajectory.append(position)
        directions.append(dirn)
        k += 1
    return np.array(trajectory, dtype=np.int16)


# def generate_eagle_track(
#         conductivity: np.ndarray,
#         potential: np.ndarray,
#         move_dirn: float,
#         start_loc: List[int],
#         dirn_restrict: int,
#         nu_par: float
# ):
#     """ Generate an eagle track """

#     num_rows, num_cols = conductivity.shape
#     burnin = int(min(num_rows, num_cols) / 10)
#     max_moves = num_rows / 2 * num_cols / 2
#     dirn = [0, 0]
#     directions = []
#     directions.append(dirn)
#     position = start_loc.copy()
#     trajectory = []
#     trajectory.append(position)
#     k = 0
#     while k < max_moves:
#         row, col = position
#         if k > burnin:
#             if not (0 < row < num_rows - 1 and 0 < col < num_cols - 1):
#                 break
#         else:
#             row, col = move_away_from_boundary(row, col, num_rows, num_cols)
#         local_cond = conductivity[row - 1:row + 2, col - 1:col + 2]
#         local_potential_energy = potential[row - 1:row + 2, col - 1:col + 2]
#         local_cond = local_cond.clip(min=1e-06)
#         mean_cond = 2.0 / (1.0 / local_cond[1, 1] + 1.0 / local_cond)
#         q_diff = local_potential_energy[1, 1] - local_potential_energy
#         q_diff = np.multiply(q_diff, neighbour_delta_norms_inv)
#         mov_probs = np.multiply(mean_cond, q_diff)
#         mov_probs = mov_probs.flatten()
#         dir_bool = get_track_restrictions(0, 0)
#         for idirn in directions[-dirn_restrict:]:
#             dir_bool = np.logical_and(get_track_restrictions(*idirn), dir_bool)
#         mov_probs = generate_move_probabilities(mov_probs, move_dirn,
#                                                 nu_par, dir_bool)
#         chosen_index = np.random.choice(range(len(mov_probs)), p=mov_probs)
#         dirn = neighbour_deltas[chosen_index]
#         position = [x + y for x, y in zip([row, col], dirn)]
#         trajectory.append(position)
#         directions.append(dirn)
#         k += 1
#     return np.array(trajectory, dtype=np.int16)


# def generate_eagle_track_drw(
#     grid_shape: Tuple[int, int],
#     start_loc: List[int],
#     move_dirn: float,
#     dirn_restrict: int,
#     nu_par: float
# ):
#     """ Generate an eagle track using directed random walk """

#     num_rows, num_cols = grid_shape
#     burnin = int(min(num_rows, num_cols) / 10)
#     max_moves = num_rows / 2 * num_cols / 2
#     dirn = [0, 0]
#     directions = []
#     directions.append(dirn)
#     position = start_loc.copy()
#     trajectory = []
#     trajectory.append(position)
#     k = 0
#     while k < max_moves:
#         row, col = position
#         if k > burnin:
#             if not (0 < row < num_rows - 1 and 0 < col < num_cols - 1):
#                 break
#         else:
#             row, col = move_away_from_boundary(row, col, num_rows, num_cols)
#         dir_bool = get_track_restrictions(0, 0)
#         mov_probs = np.ones_like(dir_bool)
#         for idirn in directions[-dirn_restrict:]:
#             dir_bool = np.logical_and(get_track_restrictions(*idirn), dir_bool)
#         mov_probs = generate_move_probabilities(mov_probs, nu_par, dir_bool)
#         chosen_index = np.random.choice(range(len(mov_probs)), p=mov_probs)
#         dirn = neighbour_deltas[chosen_index]
#         position = [x + y for x, y in zip([row, col], dirn)]
#         trajectory.append(position)
#         directions.append(dirn)
#         k += 1
#     return np.array(trajectory, dtype=np.int16)


def compute_presence_counts(
    tracks: List[np.ndarray],
    gridshape: Tuple[int, int]
):
    """ Count the number of eagles detected at each grid point """
    count_mat = np.zeros(gridshape, dtype=np.int16)
    for track in tracks:
        for move in track:
            count_mat[move[0], move[1]] += 1
    return count_mat


def compute_smooth_presence_counts(
    tracks: List[np.ndarray],
    gridshape: Tuple[int, int],
    radius: float
) -> np.ndarray:
    """ Smothens a matrix using 2D covolution of the circular kernel matrix
    with the given matrix """

    count_mat = compute_presence_counts(tracks, gridshape)
    krad = int(radius)
    kernel = np.zeros((2 * krad + 1, 2 * krad + 1))
    y, x = np.ogrid[-krad:krad + 1, -krad:krad + 1]
    mask2 = x**2 + y**2 <= krad**2
    kernel[mask2] = 1
    kernel /= np.sum(kernel)
    presence_prob = ssg.convolve2d(count_mat, kernel, mode='same')
    # presence_prob /= np.amax(presence_prob)
    return presence_prob.astype(np.float32)


def harmonic_mean(aval: float, bval: float, minval: float = 1e-10) -> float:
    """ returns harmonic mean of a and b """
    val = minval
    if aval != 0 and bval != 0:
        val = 2. / (1. / aval + 1 / bval)
    return val
