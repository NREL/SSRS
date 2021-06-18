import os
from math import atan2
from typing import Tuple

import numpy as np
import pandas as pd
import utm
from mmctools.coupling.terrain import SRTM
from scipy.interpolate import RectBivariateSpline

pd.set_option('display.float_format', lambda x: '%.1f' % x)


def get_terrain_bounds(
        tr_southwest: Tuple[float, float],
        tr_sw_pad: Tuple[float, float],
        tr_width: Tuple[float, float]
) -> pd.DataFrame:
    """Computes terrain domain bounds in km and lat/lon"""

    print('\n--- Setting terrain bounds for data extraction')
    xref, yref, zone_n, zone_l = utm.from_latlon(*list(reversed(tr_southwest)))
    xmin, ymin = xref / 1000. - tr_sw_pad[0], yref / 1000. - tr_sw_pad[1]
    xmax, ymax = xmin + tr_width[0], ymin + tr_width[1]
    wtk_pad = 2.  # extract 2 Km extra on all sides for good interpolation
    latmin, lonmin = utm.to_latlon(
        (xmin - wtk_pad) * 1000., (ymin - wtk_pad) * 1000., zone_n, zone_l)
    latmax, lonmax = utm.to_latlon(
        (xmax + wtk_pad) * 1000., (ymax + wtk_pad) * 1000., zone_n, zone_l)
    df_bounds = pd.DataFrame({'xlong': [lonmin, lonmax],
                              'ylat': [latmin, latmax],
                              'xkm': [xmin, xmax],
                              'ykm': [ymin, ymax],
                              }, index=['min', 'max'])
    print(df_bounds.head())
    return df_bounds


def extract_terrain_elevation_srtm(
        xlong_bnd: Tuple[float, float],
        ylat_bnd: Tuple[float, float],
        xkm_bnd: Tuple[float, float],
        ykm_bnd: Tuple[float, float],
        tr_res: float,
) -> np.ndarray:
    """Extracts terrain altitude using SRTM package"""

    print('\n--- Extracting terrain info from SRTM dataset ')
    srtm_bounds = (xlong_bnd[0], ylat_bnd[0],
                   xlong_bnd[1], ylat_bnd[1])
    fpathname = os.getcwd() + '/physics_site.tif'
    # SRTM1 = 30m resolution, SRTM3 = 90m
    srtm_prod = 'SRTM1' if tr_res * 1000. >= 90. else 'SRTM1'
    srtm_obj = SRTM(srtm_bounds, fpath=fpathname, product=srtm_prod)
    srtm_obj.download()
    srtm_x, srtm_y, srtm_z = srtm_obj.to_terrain()  # x,y are in meters
    # os.remove(fpathname)
    srtm_x /= 1000.
    srtm_y /= 1000.
    srtm_z /= 1000.

    # Interpolate terrain at desired resolution
    print('Interpolating terrain at {:.2f} m resolution'.format(tr_res * 1000.))
    xgrid = np.arange(xkm_bnd[0], xkm_bnd[1], tr_res)
    ygrid = np.arange(ykm_bnd[0], ykm_bnd[1], tr_res)
    interpfun = RectBivariateSpline(srtm_x[:, 0], srtm_y[0, :], srtm_z)
    tr_elev = np.transpose(interpfun(xgrid, ygrid, grid=True))
    print('Terrain grid shape:', np.shape(tr_elev))
    return tr_elev


def compute_slope_and_aspect(
        tr_elev: np.ndarray,
        tr_res: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes terrain slope and aspect using terrain elevation with a
    given grid spacing"""

    print('Computing terrain slope and aspect ', end="", flush=True)
    grid_size = tr_elev.shape
    mask_reach = 1
    mask_center = np.array((mask_reach, mask_reach))
    mask_width = mask_reach * 2 + 1
    mask_size = np.array((mask_width, mask_width))
    mask_inverse_distances = np.empty(mask_size)
    mask_aspects = np.empty(mask_size)
    for r in range(mask_size[0]):
        for c in range(mask_size[1]):
            pos = np.array((r, c)) - mask_center
            if r == mask_reach and c == mask_reach:
                mask_inverse_distances[r, c] = 0.0
                mask_aspects[r, c] = 0.0
            else:
                mask_inverse_distances[r, c] = 1.0 / \
                                               (tr_res * np.linalg.norm(pos))
                # CCW angle relative to north
                mask_aspects[r, c] = atan2(pos[1], pos[0])
    mask_aspects_modified = np.flip(mask_aspects, axis=0) + np.pi
    # print(mask_inverse_distances,'\n',mask_aspects_modified*180/np.pi)
    tr_hgt_padded = np.pad(
        tr_elev, (mask_center, mask_center), mode='edge')
    terrain_slope = np.empty(grid_size)
    terrain_aspect = np.empty(grid_size)
    for r in range(grid_size[0]):
        for c in range(grid_size[1]):
            pos = np.array((r, c))
            center = pos + mask_center
            end = pos + mask_size
            source_region = np.flip(tr_hgt_padded[r:end[0], c:end[1]], axis=0)
            center_alt = tr_hgt_padded[center[0], center[1]]
            slopes = np.arctan((source_region - center_alt)
                               * mask_inverse_distances)
            max_index = np.unravel_index(np.argmax(slopes), slopes.shape)
            terrain_slope[r, c] = slopes[max_index]
            terrain_aspect[r, c] = mask_aspects_modified[max_index]
    print('done.')
    return terrain_slope, terrain_aspect
