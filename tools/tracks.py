from math import (floor, ceil, sqrt)
from typing import List, Tuple

import numpy as np
import pathos.multiprocessing as multiprocessing
import scipy.signal as ssg
import scipy.sparse as ss


def get_boundary_nodes(mdirn: str, tshape):
    """ returns boundary nodes for given direction of movement"""
    M, N = tshape
    north_bnodes = np.array([M * (x + 1) - 1 for x in range(N)])
    south_bnodes = np.array([M * x for x in range(N)])
    west_bnodes = np.array([x for x in range(1, M - 1)])
    east_bnodes = np.array([(N - 1) * M + x for x in range(1, M - 1)])
    fac = 2  # determine portion of each boundary is used for diagonal moves
    if mdirn == 'north':
        bnodes = np.concatenate((north_bnodes, south_bnodes))
    elif mdirn == 'south':
        bnodes = np.concatenate((south_bnodes, north_bnodes))
    elif mdirn == 'east':
        bnodes = np.concatenate((east_bnodes, west_bnodes))
    elif mdirn == 'west':
        bnodes = np.concatenate((west_bnodes, east_bnodes))
    elif mdirn == 'northwest':
        bnodes = np.concatenate(
            (north_bnodes[:north_bnodes.size // fac],
             west_bnodes[west_bnodes.size // fac:],
             south_bnodes[south_bnodes.size // fac:],
             east_bnodes[:east_bnodes.size // fac],
             ))
    elif mdirn == 'southeast':
        bnodes = np.concatenate(
            (south_bnodes[south_bnodes.size // fac:],
             east_bnodes[:east_bnodes.size // fac],
             north_bnodes[:north_bnodes.size // fac],
             west_bnodes[west_bnodes.size // fac:],
             ))
    elif mdirn == 'southwest':
        bnodes = np.concatenate(
            (north_bnodes[:south_bnodes.size // fac],
             west_bnodes[:west_bnodes.size // fac],
             south_bnodes[north_bnodes.size // fac:],
             east_bnodes[east_bnodes.size // fac:],
             ))
    elif mdirn == 'northeast':
        bnodes = np.concatenate(
            (south_bnodes[north_bnodes.size // fac:],
             east_bnodes[east_bnodes.size // fac:],
             north_bnodes[:south_bnodes.size // fac],
             west_bnodes[:west_bnodes.size // fac],
             ))
    else:
        raise ValueError('Invalid track_direction')
    return bnodes


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
        return start_inds.astype(int)
    elif stype == 'random':
        idx = np.random.randint(0, base_count, ntracks)
        start_inds = base_inds[:, idx]
        return start_inds.astype(int)
    else:
        raise ValueError('Invalid sim_start_type. Options: uniform,random')


def assemble_sparse_linear_system(rM: int, cN: int):
    """ returns the row and column index of sparse linear system"""

    row_index = []
    col_index = []
    facs = []
    for i in range(0, rM * cN):
        if (i + 1) % rM == 0:  # north bndry
            nearby = [i + rM, i + rM - 1, i - 1, i - rM - 1, i - rM]
        elif i % rM == 0:  # south bndry
            nearby = [i - rM, i - rM + 1, i + 1, i + rM + 1, i + rM]
        else:  # otherwise
            nearby = [i - rM, i - rM + 1, i + 1, i + rM + 1, i + rM,
                      i + rM - 1, i - 1, i - rM - 1]
        nearby = [x for x in nearby if x >= 0 and x < rM * cN]
        row_index.extend([i for _ in range(0, len(nearby))])
        col_index.extend(nearby)
        facs.extend([sqrt(2.) if i % 2 else 1. for i in range(len(nearby))])
    row_index = np.array(row_index, dtype='u4')
    col_index = np.array(col_index, dtype='u4')
    facs = np.array(facs, dtype='f4')
    #print('Sparsity: {0:.3f} %'.format(100 * (1 - facs.size / (rM * cN)**2)))
    return row_index, col_index, facs


def harmonic_mean(a: float, b: float, minval: float = 1e-10) -> float:
    """ returns harmonic mean of a and b """
    if a != 0 and b != 0:
        return 2. / (1. / a + 1 / b)
    else:
        return minval


def solve_sparse_linear_system(
        coeff: np.ndarray,
        bnodes: np.ndarray,
        benergy: np.ndarray,
        row_inds: np.ndarray,
        col_inds: np.ndarray,
        facs: np.ndarray
) -> np.ndarray:
    """ Solves the linear system for unknown energy values at inner nodes, 
    returns the energy at all nodes (inner + bndry)"""

    rM, cN = coeff.shape
    vals = []
    for r, c, fac in zip(row_inds, col_inds, facs):
        coeff_a = coeff[r % rM, r // rM]
        coeff_b = coeff[c % rM, c // rM]
        vals.append(harmonic_mean(coeff_a, coeff_b, 1e-10) / fac)
    G_coo = ss.coo_matrix((np.array(vals),
                           (np.array(row_inds),
                            np.array(col_inds))), shape=(rM * cN, rM * cN))
    G_csr = G_coo.tocsr()
    G_csr.data = G_csr.data / np.repeat(np.add.reduceat(G_csr.data,
                                                        G_csr.indptr[:-1]),
                                        np.diff(G_csr.indptr))
    inodes = np.setdiff1d(np.arange(0, rM * cN), bnodes, assume_unique=True)
    G_inner_csr = G_csr[inodes, :]
    G_inner_csc = G_inner_csr.tocoo().tocsc()
    G_inner_inner_csc = G_inner_csc[:, inodes]
    G_inner_bndry_csc = G_inner_csc[:, bnodes]
    b_vec = G_inner_bndry_csc.dot(benergy)
    A_matrix = ss.eye(np.size(inodes)).tocsc() - G_inner_inner_csc
    ienergy = ss.linalg.spsolve(A_matrix, b_vec)
    global_energy = np.empty(rM * cN)
    global_energy[inodes] = ienergy
    global_energy[bnodes] = benergy
    pot_energy = np.empty((rM, cN))
    for i in range(rM * cN):
        pot_energy[i % rM, i // rM] = global_energy[i]
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
        coeff: np.ndarray,
        penergy: np.ndarray,
        start_loc: List[int],
        dirn_restrict: int,
        nu: float,
        max_moves: int
):
    """ Generate an eagle track """

    num_rows, num_cols = coeff.shape
    burnin = 5
    dirn = [0, 0]
    previous_dirn = [0, 0]
    position = start_loc.copy()
    trajectory = []
    trajectory.append(position)
    k = 0
    while k < max_moves:
        r, c = position
        if k > max_moves - 2:
            print('Maximum steps reached!', k, r, c)
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
        local_conductance = coeff[r - 1:r + 2, c - 1:c + 2]
        local_potential_energy = penergy[r - 1:r + 2, c - 1:c + 2]
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
        k += 1
    return np.array(trajectory, dtype=np.int16)


def compute_presence_probability(
    tracks: List[np.ndarray],
    tshape: Tuple[int, int],
    radius: float
) -> np.ndarray:
    """ Smothens a matrix using 2D covolution of the circular kernel matrix
    with the given matrix """

    A = np.zeros(tshape, dtype=np.int16)
    krad = int(radius)
    for track in tracks:
        for move in track:
            A[move[0], move[1]] += 1
    kernel = np.zeros((2 * krad + 1, 2 * krad + 1))
    y, x = np.ogrid[-krad:krad + 1, -krad:krad + 1]
    mask2 = x**2 + y**2 <= krad**2
    kernel[mask2] = 1
    Asmooth = ssg.convolve2d(A, kernel, mode='same')
    Asmooth /= np.amax(Asmooth)
    return Asmooth.astype(np.float32)
