""" Module for implementing fluid-flow based movement model """

from math import (floor, ceil, sqrt)
from typing import List, Tuple
import numpy as np
import scipy.signal as ssg
import scipy.sparse as ss


class MovModel:
    """ Class for fluid-flow based model """

    def __init__(
        self,
        move_dirn: str,
        grid_shape: Tuple[int, int]
    ):
        self.move_dirn = move_dirn
        self.grid_shape = grid_shape

    def get_boundary_nodes_old(self):
        """ returns boundary nodes for given direction of movement"""
        nrow, ncol = self.grid_shape
        north_bnodes = np.array([nrow * (x + 1) - 1 for x in range(ncol)])
        south_bnodes = np.array([nrow * x for x in range(ncol)])
        west_bnodes = np.array(list(range(1, nrow - 1)))
        east_bnodes = np.array(
            [(ncol - 1) * nrow + x for x in range(1, nrow - 1)])
        fac = 2  # determine portion of each boundary is used for diagonal moves
        if self.move_dirn == 'north':
            bnodes = np.concatenate((north_bnodes, south_bnodes))
        elif self.move_dirn == 'south':
            bnodes = np.concatenate((south_bnodes, north_bnodes))
        elif self.move_dirn == 'east':
            bnodes = np.concatenate((east_bnodes, west_bnodes))
        elif self.move_dirn == 'west':
            bnodes = np.concatenate((west_bnodes, east_bnodes))
        elif self.move_dirn == 'northwest':
            bnodes = np.concatenate(
                (north_bnodes[:north_bnodes.size // fac],
                 west_bnodes[west_bnodes.size // fac:],
                 south_bnodes[south_bnodes.size // fac:],
                 east_bnodes[:east_bnodes.size // fac],
                 ))
        elif self.move_dirn == 'southeast':
            bnodes = np.concatenate(
                (south_bnodes[south_bnodes.size // fac:],
                 east_bnodes[:east_bnodes.size // fac],
                 north_bnodes[:north_bnodes.size // fac],
                 west_bnodes[west_bnodes.size // fac:],
                 ))
        elif self.move_dirn == 'southwest':
            bnodes = np.concatenate(
                (south_bnodes[:south_bnodes.size // fac],
                 west_bnodes[:west_bnodes.size // fac],
                 north_bnodes[north_bnodes.size // fac:],
                 east_bnodes[east_bnodes.size // fac:],
                 ))
        elif self.move_dirn == 'northeast':
            bnodes = np.concatenate(
                (north_bnodes[north_bnodes.size // fac:],
                 east_bnodes[east_bnodes.size // fac:],
                 south_bnodes[:south_bnodes.size // fac],
                 west_bnodes[:west_bnodes.size // fac],
                 ))
        else:
            raise ValueError(f'ModelSSRS: Invalid direction {self.move_dirn}')
        bndry_energy = np.zeros((bnodes.size))
        bndry_energy[bnodes.size // 2:] = 1000.

        return bnodes, bndry_energy

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
                        conductivity_b, 1e-10) / fac)
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


def generate_eagle_track(
        conductivity: np.ndarray,
        potential: np.ndarray,
        start_loc: List[int],
        dirn_restrict: int,
        nu_par: float
):
    """ Generate an eagle track """

    num_rows, num_cols = conductivity.shape
    burnin = 10
    max_moves = num_rows * num_cols
    dirn = [0, 0]
    directions = []
    directions.append(dirn)
    position = start_loc.copy()
    trajectory = []
    trajectory.append(position)
    #print('start track at',position,flush=True)
    k = 0
    while k < max_moves:
        row, col = position
        if k > max_moves - 2:
            print(f'Maximum steps reached at {row},{col}')
        if k > burnin:
            if row <= 0 or row >= num_rows - 1 or col <= 0 or col >= num_cols - 1:
                break  # absorb if we hit a boundary
        local_cond = conductivity[row - 1:row + 2, col - 1:col + 2]
        local_potential_energy = potential[row - 1:row + 2, col - 1:col + 2]
        local_cond = local_cond.clip(min=1e-5)
        try:
            mean_cond = 2.0 / (1.0 / local_cond[1, 1] + 1.0 / local_cond)
        except IndexError:
            print(f'point at ({row:d},{col:d}) is at boundary after {k+1:d} moves? local_cond:{local_cond}')
            break
        q_diff = local_potential_energy[1, 1] - local_potential_energy
        if np.count_nonzero(q_diff) == 0:
            q_diff = 1. + np.random.rand(*q_diff.shape) * 1e-1
            # print('All potentials same!')
        q_diff = np.multiply(q_diff, neighbour_delta_norms_inv)
        mov_probs = np.multiply(mean_cond, q_diff)
        mov_probs = mov_probs.flatten()
        mov_probs -= np.min(mov_probs)
        mov_probs[4] = 0.
        vec_bool = get_track_restrictions(0, 0)
        for idirn in directions[-dirn_restrict:]:
            vec_bool = np.logical_and(get_track_restrictions(*idirn), vec_bool)
        rmov_probs = [x * float(y) for x, y in zip(mov_probs, vec_bool)]
        if np.sum(rmov_probs) != 0:
            rmov_probs /= np.sum(rmov_probs)
            rmov_probs = np.power(rmov_probs, nu_par)
            rmov_probs /= np.sum(rmov_probs)
            chosen_index = np.random.choice(
                range(len(rmov_probs)), p=rmov_probs)
        else:
            # print('Sum(mov_probs) is zero!')
            if np.sum(mov_probs) != 0:
                mov_probs /= np.sum(mov_probs)
                mov_probs = np.power(mov_probs, nu_par)
                mov_probs /= np.sum(mov_probs)
                chosen_index = np.random.choice(
                    range(len(mov_probs)), p=mov_probs)
            else:
                chosen_index = np.random.choice(range(len(mov_probs)))
        # print(neighbour_deltas)
        dirn = neighbour_deltas[chosen_index]
        position = [x + y for x, y in zip(position, dirn)]
        trajectory.append(position)
        directions.append(dirn)
        k += 1
    return np.array(trajectory, dtype=np.int16)


def generate_eagle_track_old(
        conductivity: np.ndarray,
        potential: np.ndarray,
        start_loc: List[int],
        dirn_restrict: int,
        nu_par: float
):
    """ Generate an eagle track """

    num_rows, num_cols = conductivity.shape
    burnin = 5
    max_moves = num_rows * num_cols
    dirn = [0, 0]
    previous_dirn = [0, 0]
    position = start_loc.copy()
    trajectory = []
    trajectory.append(position)
    k = 0
    while k < max_moves:
        row, col = position
        if k > max_moves - 2:
            print(f'Maximum steps reached at {row},{col}')
        if k > burnin:
            if row <= 0 or row >= num_rows - 1 or col <= 0 or col >= num_cols - 1:
                break  # absorb if we hit a boundary
        else:
            if row in (0, 1):
                row += 1
            elif row in (num_rows - 1, num_rows):
                row -= 1
            if col in (0, 1):
                col += 1
            elif col in (num_cols - 1, num_cols):
                col -= 1
        position = [row, col]
        previous_dirn = np.copy(dirn)
        local_cond = conductivity[row - 1:row + 2, col - 1:col + 2]
        local_potential_energy = potential[row - 1:row + 2, col - 1:col + 2]
        local_cond = local_cond.clip(min=1e-5)
        mc = 2.0 / (1.0 / local_cond[1, 1] + 1.0 / local_cond)
        q_diff = local_potential_energy[1, 1] - local_potential_energy
        if np.count_nonzero(q_diff) == 0:
            q_diff = 1. + np.random.rand(*q_diff.shape) * 1e-1
            # print('All potentials same!')
        q_diff = np.multiply(q_diff, neighbour_delta_norms_inv)
        mov_probs = np.multiply(mc, q_diff)
        mov_probs = mov_probs.flatten()
        mov_probs -= np.min(mov_probs)
        mov_probs[4] = 0.
        if dirn_restrict > 0 and k > burnin:
            zvec = get_track_restrictions(*dirn)
            if dirn_restrict == 2:
                zvec = np.logical_and(
                    zvec, get_track_restrictions(*previous_dirn))
            if sum(zvec) != 0:
                q_new = [x * float(y) for x, y in zip(mov_probs, zvec)]
                if np.sum(q_new) != 0:
                    mov_probs = q_new.copy()
        if np.sum(mov_probs) != 0:
            mov_probs /= np.sum(mov_probs)
            mov_probs = np.power(mov_probs, nu_par)
            mov_probs /= np.sum(mov_probs)
            chosen_index = np.random.choice(range(len(mov_probs)), p=mov_probs)
            # print([round(x, 1) for x in mov_probs], chosen_index)
        else:
            # print('Sum(q) is zero!')
            chosen_index = np.random.choice(range(len(mov_probs)))
        # print(neighbour_deltas)
        dirn = neighbour_deltas[chosen_index]
        position = [x + y for x, y in zip(position, dirn)]
        trajectory.append(position)
        k += 1
    return np.array(trajectory, dtype=np.int16)


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
    xind_low = min(max(floor(sbounds[0] / res_km) - 1, 0), xind_max)
    xind_upp = max(min(ceil(sbounds[1] / res_km), xind_max), 1)
    yind_low = min(max(floor(sbounds[2] / res_km) - 1, 0), yind_max)
    yind_upp = max(min(ceil(sbounds[3] / res_km), yind_max), 1)
    xmesh, ymesh = np.mgrid[xind_low:xind_upp, yind_low:yind_upp]
    base_inds = np.vstack((np.ravel(ymesh), np.ravel(xmesh)))
    base_count = base_inds.shape[1]
    if stype == 'uniform':
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
                          'Options: uniform, random'))
    start_inds = start_inds.astype(int)
    return start_inds[0, :], start_inds[1, :]


def compute_presence_count(
    tracks: List[np.ndarray],
    gridshape: Tuple[int, int]
):
    """ Count the number of eagles detected at each grid point """
    count_mat = np.zeros(gridshape, dtype=np.int16)
    for track in tracks:
        for move in track:
            count_mat[move[0], move[1]] += 1
    return count_mat


def compute_presence_probability(
    tracks: List[np.ndarray],
    gridshape: Tuple[int, int],
    radius: float
) -> np.ndarray:
    """ Smothens a matrix using 2D covolution of the circular kernel matrix
    with the given matrix """

    count_mat = compute_presence_count(tracks, gridshape)
    krad = int(radius)
    kernel = np.zeros((2 * krad + 1, 2 * krad + 1))
    y, x = np.ogrid[-krad:krad + 1, -krad:krad + 1]
    mask2 = x**2 + y**2 <= krad**2
    kernel[mask2] = 1
    presence_prob = ssg.convolve2d(count_mat, kernel, mode='same')
    presence_prob /= np.amax(presence_prob)
    return presence_prob.astype(np.float32)


def harmonic_mean(aval: float, bval: float, minval: float = 1e-10) -> float:
    """ returns harmonic mean of a and b """
    val = minval
    if aval != 0 and bval != 0:
        val = 2. / (1. / aval + 1 / bval)
    return val
