""" Module for manipulating raster data in different coordinate reference
systems (crs) """

import os
from typing import List, Tuple, Union
import numpy as np
import rasterio as rs
from rasterio.warp import transform, reproject
from rasterio.crs import CRS
from rasterio._err import CPLE_AppDefinedError


def get_raster_in_projected_crs(
    fpath: str,
    proj_bounds: Tuple[float, float, float, float],
    proj_gridsize: Tuple[int, int],
    proj_res: Union[float, Tuple[float, float]],
    proj_crs_string: str
):
    """ Get raster data from fpath in projected crs """

    # get the CRS object from projected crs
    proj_crs = get_rasterio_crs_object(proj_crs_string)
    assert proj_crs.is_projected, f'{proj_crs_string} is not a projected crs!'
    check_if_raster_file_exists(fpath)

    # Compute the Affine transform in projected crs
    proj_dx = proj_res if isinstance(proj_res, float) else proj_res[0]
    proj_dy = proj_res if isinstance(proj_res, float) else proj_res[1]
    assert (proj_dx > 0. and proj_dy > 0.), f'{proj_res} invalid resolution!'
    proj_transform = rs.transform.from_origin(
        proj_bounds[0], proj_bounds[3], proj_dx, proj_dy)

    # reproject the data into projeted crs
    resampling_method = rs.warp.Resampling.bilinear
    band = 1
    proj_dem = np.empty(proj_gridsize)
    try:
        with rs.open(fpath) as src_img:
            reproject(source=src_img.read(band),
                      destination=proj_dem,
                      src_transform=src_img.transform,
                      src_crs=src_img.crs,
                      dst_transform=proj_transform,
                      dst_crs=proj_crs,
                      resampling=resampling_method,
                      num_threads=1)
    except FileNotFoundError as _:
        raise FileNotFoundError(f'Need to download {fpath} first!') from None
    return np.flipud(proj_dem)


def transform_bounds(
        src_bounds: Tuple[float, float, float, float],
        src_crs_string: str,
        dest_crs_string: str,
        pad: float = 0.
) -> Tuple[float, float, float, float]:
    """ Compute bounds of the rectangular region in destination coordinate
    reference system (crs) that is contained within the bounds in source crs

    Parameters:
    ------
    src_bounds: Tuple[float, float, float, float]
        Defines the bounds = (min_lon, min_lat, max_lon, max_lat) of the
        rectangular region in source crs
    src_crs_string: str
        EPSG, ESRI, PROJ4, or WKT string that defines the source crs
    dest_crs_string: str
        EPSG, ESRI, PROJ4, or WKT string that defines the destination crs

    Returns:
    -------
    dest_bounds: Tuple[float, float, float, float]
        (min_x, min_y, max_x, max_y) in destination crs
    """
    src_crs = get_rasterio_crs_object(src_crs_string)
    dest_crs = get_rasterio_crs_object(dest_crs_string)
    src_pts = get_corner_points_from_bounds(src_bounds)
    dest_x, dest_y = transform_coordinates(
        src_crs, dest_crs, src_pts[0], src_pts[1])
    dest_bounds = (min(dest_x), min(dest_y), max(dest_x), max(dest_y))
    padding = [-pad, -pad, pad, pad]
    dest_bounds = [ix + iy for ix, iy in zip(dest_bounds, padding)]
    return dest_bounds


def transform_coordinates(
    in_crs: Union[str, rs.crs.CRS],
    out_crs: Union[str, rs.crs.CRS],
    in_x: Union[float, List[float], np.ndarray],
    in_y: Union[float, List[float], np.ndarray],
    num_retries=5
):
    """Transform points from source crs to destination crs

    Parameters:
    ----------
    in_crs: string or rasterio.crs.CRS object
        Defines the coordinate ref system for the input
    out_crs: string or rasterio.crs.CRS object
        Defines the coordinate ref system for the output
    in_x: float or List or numpy array
        Input values in x direction (easting)
    out_x: float or List or numpy array
        Output values in y direction (northing)
    num_retries: int, optional
        Number of times to attempt to call rasterio.warp.transform if
        CPLE_AppDefinedError is encountered

    Returns:
    -------
    out_x: list of
    """

    # convert floats into list
    in_x = [in_x] if isinstance(in_x, float) else in_x
    in_y = [in_y] if isinstance(in_y, float) else in_y

    # convert multi dimensional numpy array to 1D array that transform needs
    if isinstance(in_x, np.ndarray):
        out_shape = in_x.shape
        in_x = in_x.flatten()
        in_y = in_y.flatten()
    assert len(in_x) == len(in_y)

    # Get CRS object for input crs
    if not isinstance(in_crs, rs.crs.CRS):
        in_crs = get_rasterio_crs_object(in_crs)
    if not isinstance(out_crs, rs.crs.CRS):
        out_crs = get_rasterio_crs_object(out_crs)
    assert out_crs.is_valid

    # Perform the transformation
    success = False
    for i in range(num_retries):
        try:
            out_x, out_y = transform(in_crs, out_crs, in_x, in_y)
        except CPLE_AppDefinedError as e:
            # Occasionally we see "Network error when accessing a remote resource"
            # Unclear why, but this is a rasterio issue.        
            pass
        else:
            success = True
            break
    assert success, 'Unable to transform coordinates\n'+e

    # reshape output to match the input numpy array
    if 'out_shape' in locals():
        out_x = np.reshape(out_x, out_shape)
        out_y = np.reshape(out_y, out_shape)
    return out_x, out_y


def get_raster_data(fpath: str, band: int = 1) -> np.ndarray:
    """ Get data saved in fpath (GeoTif file) in source crs """
    check_if_raster_file_exists(fpath)
    with rs.open(fpath) as src_img:
        src_data = src_img.read(band)
    return np.flipud(src_data)


def get_raster_bounds(fpath: str) -> Tuple[float, float, float, float]:
    """ Get bounds of the raster data in fpath (GeoTif file) """
    check_if_raster_file_exists(fpath)
    with rs.open(fpath) as src_img:
        src_bnds = [round(ix, 8) for ix in src_img.bounds]
    return src_bnds


def check_if_raster_file_exists(fpath: str):
    """ Check if GeoTif file exists """
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f'{fpath} not found!\nRun download()')


def get_rasterio_crs_object(crs_string: str) -> rs.crs.CRS:
    """ Returns a valid rasterio CRS object from given EPSG, ESRI, PROJ4,
    or WKT string """
    err_string = (f'Rasterio: {crs_string} is an invalid crs!\n'
                  'Choose a valid EPSG, ESRI, PROJ4, or WKT string')
    try:
        proj_crs = CRS.from_user_input(crs_string)
    except rs.errors.CRSError as _:
        raise ValueError(err_string) from None
    else:
        if not proj_crs.is_valid:
            raise ValueError(err_string)
    return proj_crs


def get_utm_string(west_lon: float) -> str:
    """ Returns PROJ4 string for UTM projection for a given zone, computed
    using the west (longitude) value

    Parameters:
    ----------
    west_lon : float
        Longitude for the western most point of the domain

    Returns:
    -------
    PROJ4 string for the UTM zone
    """

    datum = 'WGS84'
    ellps = 'WGS84'
    zone_number = int((west_lon + 180) / 6) + 1
    proj_string = (f'+proj=utm +zone={zone_number} +datum={datum} +units=m'
                   f'+no_defs +ellps={ellps} +towgs84=0,0,0')
    return proj_string


def get_corner_points_from_bounds(bounds: Tuple[float, float, float, float]):
    """ get coordinates of corner points from bounds list """
    xord = [bounds[0], bounds[0], bounds[2], bounds[2]]
    yord = [bounds[1], bounds[3], bounds[1], bounds[3]]
    return [xord, yord]


# Junk
# def is_bounds_within_bounds(
#     bounds_a: Tuple[float, float, float, float],
#     bounds_b: Tuple[float, float, float, float]
# ) -> bool:
#     """ Check if one bounds are within (subset) of other bounds"""
#     return_bool = False
#     if (
#         (bounds_a[0] >= bounds_b[0]) &
#         (bounds_a[1] >= bounds_b[1]) &
#         (bounds_a[2] <= bounds_b[2]) &
#         (bounds_a[3] <= bounds_b[3])
#     ):
#         return_bool = True
#     return return_bool

    # def compute_slope_and_aspect(
    #     tr_elev: np.ndarray,
    #     tr_res: float
    # ) -> Tuple[np.ndarray, np.ndarray]:
    #     """Computes terrain slope and aspect using terrain elevation with a
    #     given grid spacing"""

    #     print('Computing terrain slope and aspect...', end="", flush=True)
    #     grid_size = tr_elev.shape
    #     mask_reach = 1
    #     mask_center = np.array((mask_reach, mask_reach))
    #     mask_width = mask_reach * 2 + 1
    #     mask_size = np.array((mask_width, mask_width))
    #     mask_inverse_distances = np.empty(mask_size)
    #     mask_aspects = np.empty(mask_size)
    #     for r in range(mask_size[0]):
    #         for c in range(mask_size[1]):
    #             pos = np.array((r, c)) - mask_center
    #             if r == mask_reach and c == mask_reach:
    #                 mask_inverse_distances[r, c] = 0.0
    #                 mask_aspects[r, c] = 0.0
    #             else:
    #                 mask_inverse_distances[r, c] = 1000.0 / \
    #                     (tr_res * np.linalg.norm(pos))
    #                 # CCW angle relative to north
    #                 mask_aspects[r, c] = np.arctan2(pos[1], pos[0])
    #     mask_aspects_modified = np.flip(mask_aspects, axis=0) + np.pi
    #     # print(mask_inverse_distances,'\n',mask_aspects_modified*180/np.pi)
    #     tr_hgt_padded = np.pad(
    #         tr_elev, (mask_center, mask_center), mode='edge')
    #     terrain_slope = np.empty(grid_size)
    #     terrain_aspect = np.empty(grid_size)
    #     for r in range(grid_size[0]):
    #         for c in range(grid_size[1]):
    #             pos = np.array((r, c))
    #             center = pos + mask_center
    #             end = pos + mask_size
    #             source_region = np.flip(tr_hgt_padded[r:end[0], c:end[1]], axis=0)
    #             center_alt = tr_hgt_padded[center[0], center[1]]
    #             slopes = np.arctan((source_region - center_alt)
    #                                * mask_inverse_distances)
    #             max_index = np.unravel_index(np.argmax(slopes), slopes.shape)
    #             terrain_slope[r, c] = slopes[max_index]
    #             terrain_aspect[r, c] = mask_aspects_modified[max_index]
    #     print('done')
    #     return terrain_slope.astype(np.float32), terrain_aspect.astype(np.float32)


# def compute_proj_bounds_from_lonlat_bounds(
#         lonlat_bounds: Tuple[float, float, float, float],
#         proj_crs_string: str,
#         margin_km: float = 0.0
# ) -> List[float]:
#     """ Compute bounds of the rectangular region in projected coordinate
#     reference system (crs) that is contained within the bounds in geographical
#     (lon/lat) crs

#     Parameters:
#     ------
#     lonlat_bounds: Tuple[float, float, float, float]
#         Defines the bounds = (min_lon, min_lat, max_lon, max_lat) of the
#         rectangular region in lon/lat crs
#     proj_crs_string: str
#         EPSG, ESRI, PROJ4, or WKT string that defines the projected crs
#     margin_km: float, defaults to 0
#         padding at the corners of the rectangular region in projected crs

#     Returns:
#     -------
#     bounds: (min_lon, min_lat, max_lon, max_lat) in projected crs in meters
#     """
#     lonlat_crs = CRS.from_user_input('EPSG:4326')
#     proj_crs = CRS.from_user_input(proj_crs_string)
#     assert proj_crs.is_projected
#     lonlat_pts = get_corner_points_from_bounds(lonlat_bounds)
#     proj_xs, proj_ys = transform_crs(lonlat_crs, proj_crs,
#                                      lonlat_pts[0], lonlat_pts[1])
#     pad = [margin_km, margin_km, -margin_km, -margin_km]
#     proj_bounds = [min(proj_xs), min(proj_ys), max(proj_xs), max(proj_ys)]
#     return [ix + iy * 1000. for ix, iy in zip(proj_bounds, pad)]


# def compute_lonlat_bounds_from_proj_bounds(
#         proj_bounds: Tuple[float, float, float, float],
#         proj_crs_string: str,
#         margin_km: float = 0.0
# ) -> List[float]:
#     """ Set bounds in destination crs for the target region

#     Parameters:
#     -----------
#     dest_bounds: Tuple[float, float, float, float]
#         Defines the (min_x, min_y, max_x, max_y) of the target region in
#         destination crs where all values are in meters
#     margin_km: float, optional
#         Margin for ensuring target region in destination crs is subset of
#         bounds in lon/lat crs
#     """
#     lonlat_crs = CRS.from_user_input('EPSG:4326')
#     proj_crs = CRS.from_user_input(proj_crs_string)
#     assert proj_crs.is_projected
#     pad = [-margin_km, -margin_km, margin_km, margin_km]
#     pad_proj_bounds = [ix + iy * 1000. for ix, iy in zip(proj_bounds, pad)]
#     proj_pts = get_corner_points_from_bounds(pad_proj_bounds)
#     lons, lats = transform_crs(proj_crs, lonlat_crs, proj_pts[0], proj_pts[1])
#     lonlat_bounds = [min(lons), min(lats), max(lons), max(lats)]
#     return [round(ix, 8) for ix in lonlat_bounds]

    # def set_bounds_in_src_crs(
    #         self,
    #         src_bounds: Tuple[float, float, float, float]
    # ) -> None:
    #     """ Set bounds in source crs for the target region

    #     Parameters:
    #     -----------

    #     src_bounds: Tuple[float, float, float, float]
    #         Defines the (min_lon, min_lat, max_lon, max_lat) of the target
    #         region in lon/lat coordinate system
    #     """

    #     print('Terrain: Setting bounds in source crs..')
    #     src_pts = get_corner_points_from_bounds(src_bounds)
    #     dest_xs, dest_ys = transform_crs(self.src_crs, self.dest_crs,
    #                                      src_pts[0], src_pts[1])
    #     self.dest_bounds = [min(dest_xs), min(dest_ys),
    #                         max(dest_xs), max(dest_ys)]
    #     self.src_bounds = src_bounds

    # def set_bounds_in_dest_crs(
    #         self,
    #         dest_bounds: Tuple[float, float, float, float],
    #         margin_km: float = 0.1
    # ) -> None:
    #     """ Set bounds in destination crs for the target region

    #     Parameters:
    #     -----------
    #     dest_bounds: Tuple[float, float, float, float]
    #         Defines the (min_x, min_y, max_x, max_y) of the target region in
    #         destination crs where all values are in meters
    #     margin_km: float, optional
    #         Margin for ensuring target region in destination crs is subset of
    #         bounds in lon/lat crs
    #     """

    #     print('Terrain: Setting bounds in destination crs..')
    #     pad = [-margin_km, -margin_km, margin_km, margin_km]
    #     pad_dest_bounds = [ix + iy * 1000. for ix, iy in zip(dest_bounds, pad)]
    #     pad_dest_pts = get_corner_points_from_bounds(pad_dest_bounds)
    #     src_lons, src_lats = transform_crs(
    #         self.dest_crs, self.src_crs, pad_dest_pts[0], pad_dest_pts[1])
    #     self.src_bounds = [min(src_lons), min(src_lats),
    #                        max(src_lons), max(src_lats)]
    #     self.src_bounds = [round(ix, 8) for ix in self.src_bounds]
    #     self.dest_bounds = dest_bounds

    # def set_bounds_southwest_and_sizeinkm_dest(
    #     self,
    #     southwest_src: Tuple[float, float],
    #     sizeinkm_dest: Tuple[float, float]
    # ) -> None:
    #     """ Set bounds using southwest point in source crs and width and height
    #     in kms in destination crs

    #     Parameters:
    #     -----------
    #     southwest_src: Tuple[float, float]
    #         Defines the (lon, lat) of the southwest point for the target region
    #     sizeinkm_dest: Tuple[float, float]
    #         Defines the size (xsize, ysize) of region in kilometers
    #     """

    #     print('Terrain: Setting bounds using southwest_lonlat and size_km..')
    #     src_west, src_south = southwest_src
    #     dest_west, dest_south = transform_crs(
    #         self.src_crs, self.dest_crs, src_west, src_south)
    #     dest_west = dest_west[0]
    #     dest_south = dest_south[0]
    #     dest_east = dest_west + sizeinkm_dest[0] * 1000.
    #     dest_north = dest_south + sizeinkm_dest[1] * 1000.
    #     dest_bounds = (dest_west, dest_south, dest_east, dest_north)
    #     self.set_bounds_in_dest_crs(dest_bounds, 0.25)


# Junk
    # def download(self) -> None:
    #     """ Download data from the WMS server """

    #     # figure out the grid size of from bounds and resolution
    #     width = int(round((self.bnds[2] - self.bnds[0]) / self.res))
    #     height = int(round((self.bnds[3] - self.bnds[1]) / self.res))
    #     count_xtiles = int(width // self.max_gridsize) + 1
    #     count_ytiles = int(height // self.max_gridsize) + 1
    #     xbnds = np.linspace(self.bnds[0], self.bnds[2], count_xtiles + 1)
    #     ybnds = np.linspace(self.bnds[1], self.bnds[3], count_ytiles + 1)

    #     # break the region into smaller tiles
    #     k = 0
    #     fname_list = []
    #     bnds_list = []
    #     width_list = []
    #     height_list = []
    #     for i in range(count_xtiles):
    #         for j in range(count_ytiles):
    #             k_bnd = [xbnds[i], ybnds[j], xbnds[i + 1], ybnds[j + 1]]
    #             bnds_list.append(k_bnd)
    #             width_list.append(int(round((k_bnd[2] - k_bnd[0]) / self.res)))
    #             height_list.append(
    #                 int(round((k_bnd[3] - k_bnd[1]) / self.res)))
    #             fname_list.append(f'{self.out_dir}/wms_raw_{k}.tif')
    #             k += 1

    #     # tried requesting data using pathos but slower compared to serial
    #     # start = time.time()
    #     # with mp.Pool(2) as pool:
    #     #     imaps = pool.map(lambda i: self.wms.getmap(
    #     #         layers=[self.layer],
    #     #         srs=self.crsys,
    #     #         bbox=bnds_list[i],
    #     #         format=self.fmt,
    #     #         size=(width_list[i], height_list[i])
    #     #     ), range(len(bnds_list)))
    #     # for i, imap in enumerate(imaps):
    #     #     with open(fname_list[i], 'wb') as out_file:
    #     #         out_file.write(imap.read())
    #     # end = time.time()
    #     # print(f'parallel: took {round(end - start,2)} seconds')

    #     print(f'WMS: Downloading {count_xtiles}*{count_ytiles}={k} tiles..')
    #     for b, fp, w, h in zip(bnds_list, fname_list, width_list, height_list):
    #         imap = self.wms.getmap(
    #             layers=[self.layer],
    #             srs=self.crsys,
    #             bbox=b,
    #             format=self.fmt,
    #             size=(w, h)
    #         )
    #         with open(fp, 'wb') as out_file:
    #             out_file.write(imap.read())

    #     # merge tiled data and save it to a file
    #     print('WMS: Merging tiles..')
    #     src_files_to_mosaic = []
    #     for fname in fname_list:
    #         src = rs.open(fname)
    #         src_files_to_mosaic.append(src)
    #     mosaic, out_trans = merge(src_files_to_mosaic)
    #     out_meta = src.meta.copy()
    #     out_meta.update({"driver": "GTiff",
    #                      "height": mosaic.shape[1],
    #                      "width": mosaic.shape[2],
    #                      "transform": out_trans,
    #                      "crs": self.crsys
    #                      })
    #     print(f'WMS: Shape is ({mosaic.shape[1]},{mosaic.shape[2]})')
    #     with rs.open(self.fpath, "w", **out_meta) as dest:
    #         dest.write(mosaic)

    #     # close the opened raster files and do cleanup
    #     for src in src_files_to_mosaic:
    #         src.close()
    #         if self.cleanup:
    #             try:
    #                 os.remove(src.name)
    #             except OSError as e_obj:
    #                 if e_obj.errno != errno.ENOENT:
    #                     raise

    # def get_raw_raster(self):
    #     """ Get raw raster object for debugging """

    #     src = rs.open(self.fpath)
    #     return src

# def compute_bounds_using_southwest_and_size(
#     southwest_lonlat: Tuple[float, float],
#     sizeinkm: Tuple[float, float],
#     proj_crs_string: str,
#     padding: float = 0.001
# ):
#     """ Compute bounds in projected crs using southwest point in lon/lat crs
#     and width and height of the rect region in kms in a projected crs

#     Parameters:
#     ------
#     southwest_lonlat: Tuple[float, float]
#         Defines the (lon, lat) of the southwest point for the target region
#     sizeinkm: Tuple[float, float]
#         Defines the size (xsize, ysize) of region in kilometers
#     proj_crs_string: str
#         EPSG, ESRI, PROJ4, or WKT string that defines the projected crs

#     Returns:
#     -------
#     lonlat_bounds: Tuple[float, float, float, float]
#         (min_x, min_y, max_x, max_y) in lonlat crs
#     proj_bounds: Tuple[float, float, float, float]
#         (min_x, min_y, max_x, max_y) in projected crs
#     """

#     # get the CRS object for lonlat and projected crs
#     lonlat_crs = get_rasterio_crs_object('EPSG:4326')
#     proj_crs = get_rasterio_crs_object(proj_crs_string)
#     assert proj_crs.is_projected, f'{proj_crs_string} not a projected crs!'

#     # compute the bounds in projected crs
#     lon_west, lat_south = southwest_lonlat
#     proj_west, proj_south = transform_crs(lonlat_crs, proj_crs,
#                                           lon_west, lat_south)
#     proj_east = proj_west[0] + sizeinkm[0] * 1000.
#     proj_north = proj_south[0] + sizeinkm[1] * 1000.
#     proj_bounds = (proj_west[0], proj_south[0], proj_east, proj_north)
#     lonlat_bounds = transform_bounds(proj_bounds, proj_crs_string, 'EPSG:4236')
#     pad = [-padding, -padding, padding, padding]
#     lonlat_bounds = [round(ix + iy, 4) for ix, iy in zip(lonlat_bounds, pad)]
#     return lonlat_bounds, proj_bounds

# def get_raster_bounds_and_size(
#     bnds: Tuple[float, float, float, float],
#     proj_dx: float,
#     proj_dy: float
# ):
#     """  Get bounds compatible with width, height for a given resolution """
#     proj_west, proj_south, proj_east, proj_north = bnds
#     width = int(round(((proj_east - proj_west) / proj_dx)))
#     height = int(round(((proj_north - proj_south) / proj_dy)))
#     new_east = proj_west + width * proj_dx
#     new_north = proj_south + height * proj_dy
#     return (proj_west, proj_south, new_east, new_north), width, height
