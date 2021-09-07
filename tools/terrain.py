import os
import utm
import numpy as np
import elevation
import rasterio
from rasterio import transform, warp
from rasterio.crs import CRS
from typing import List, Tuple
from scipy.interpolate import RectBivariateSpline


def calculate_terrain_extent(
    southwest_lonlat: Tuple[float, float],
    terrain_width_km: Tuple[float, float]
):
    """ Calculates terrain bound in UTM km and lon/lat"""

    sw_latlon = list(reversed(southwest_lonlat))
    xref, yref, zone_n, zone_l = utm.from_latlon(*sw_latlon)
    xmin = xref / 1000.
    ymin = yref / 1000.
    xmax = xmin + terrain_width_km[0]
    ymax = ymin + terrain_width_km[1]
    tr_extent_km = [xmin, xmax, ymin, ymax]
    latmin, lonmin = utm.to_latlon(xmin * 1000., ymin * 1000., zone_n, zone_l)
    latmax, lonmax = utm.to_latlon(xmax * 1000., ymax * 1000., zone_n, zone_l)
    tr_extent_lonlat = [lonmin, lonmax, latmin, latmax]
    return tr_extent_km, tr_extent_lonlat


def extract_terrain_altitude_srtm(
    extent_lonlat: Tuple[float, float, float, float],
    extent_km: Tuple[float, float, float, float],
    terrain_res: float,
    data_dir: str
):
    """ Extracts terrain elevation from SRTM"""

    lonmin, lonmax, latmin, latmax = extent_lonlat
    xmin, xmax, ymin, ymax = extent_km
    srtm_bounds = (lonmin, latmin, lonmax, latmax)
    fpathname = data_dir + 'terrain_srtm.tif'
    try:
        srtm_prod = 'SRTM3' if terrain_res >= 90. else 'SRTM1'
        srtm_obj = SRTM(srtm_bounds, fpath=fpathname, product=srtm_prod)
        srtm_obj.download()
        srtm_x, srtm_y, srtm_z = srtm_obj.to_terrain()
        if np.any(srtm_z < 0.) or np.any(srtm_z == np.nan):
            print('Something went wrong with SRTM3, trying SRTM1...')
            raise Exception
    except:
        if srtm_prod == 'SRTM3':
            srtm_obj = SRTM(srtm_bounds, fpath=fpathname, product='SRTM1')
            srtm_obj.download()
            srtm_x, srtm_y, srtm_z = srtm_obj.to_terrain()
        else:
            print('Something wrong with SRTM altitude data!')
            exit('Try changing the terrain lon/lat bounds')
    xsize = int((xmax - xmin) * 1000. // terrain_res)
    ysize = int((ymax - ymin) * 1000. // terrain_res)
    xgrid = np.linspace(xmin * 1000., xmax * 1000., xsize)
    ygrid = np.linspace(ymin * 1000., ymax * 1000., ysize)
    interpfun = RectBivariateSpline(srtm_x[:, 0], srtm_y[0, :], srtm_z)
    tr_altitude = np.transpose(interpfun(xgrid, ygrid, grid=True))
    tr_altitude /= 1000.
    return tr_altitude.astype(np.float32)


def compute_slope_and_aspect(
    tr_elev: np.ndarray,
    tr_res: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes terrain slope and aspect using terrain elevation with a
    given grid spacing"""

    print('Computing terrain slope and aspect...', end="", flush=True)
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
                mask_inverse_distances[r, c] = 1000.0 / \
                    (tr_res * np.linalg.norm(pos))
                # CCW angle relative to north
                mask_aspects[r, c] = np.arctan2(pos[1], pos[0])
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
    print('done')
    return terrain_slope.astype(np.float32), terrain_aspect.astype(np.float32)


class Terrain(object):

    latlon_crs = CRS.from_dict(init='epsg:4326')

    def __init__(self, latlon_bounds, fpath='terrain.tif'):
        """Create container for manipulating GeoTIFF data in the
        specified region
        Usage
        =====
        latlon_bounds : list or tuple
            Latitude/longitude corresponding to west, south, east, and
            north bounds, used to define the source transformation.
        fpath : str, optional
            Where to save downloaded GeoTIFF (*.tif) data.
        """
        self.bounds = list(latlon_bounds)
        self._get_utm_crs()  # from bounds
        self.tiffdata = fpath
        self.have_terrain = False
        if not hasattr(self, 'have_metadata'):
            # set attribute if it hasn't been set already
            self.have_metadata = False

    def _get_utm_crs(self, datum='WGS84', ellps='WGS84'):
        """Get coordinate system from zone number associated with the
        longitude of the northwest corner
        Parameters
        ==========
        datum : str, optional
            Origin of destination coordinate system, used to describe
            PROJ.4 string; default is WGS84.
        ellps : str, optional
            Ellipsoid defining the shape of the earth in the destination
            coordinate system, used to describe PROJ.4 string; default
            is WGS84.
        """
        #west, south, east, north = self.bounds
        self.zone_number = int((self.bounds[0] + 180) / 6) + 1
        proj = '+proj=utm +zone={:d} '.format(self.zone_number) \
            + '+datum={:s} +units=m +no_defs '.format(datum) \
            + '+ellps={:s} +towgs84=0,0,0'.format(ellps)
        self.utm_crs = CRS.from_proj4(proj)

    def _get_bounds_from_metadata(self):
        """This is a stub"""
        assert self.have_metadata
        raise NotImplementedError()

    def to_terrain(self, dx, dy=None, resampling=warp.Resampling.bilinear):
        """Load geospatial raster data and reproject onto specified grid
        Usage
        =====
        dx,dy : float
            Grid spacings [m]. If dy is not specified, then uniform
            spacing is assumed.
        resampling : warp.Resampling value, optional
            See `list(warp.Resampling)`.
        """
        if dy is None:
            dy = dx

        # load raster
        if not os.path.isfile(self.tiffdata):
            raise FileNotFoundError('Need to download()')
        dem_raster = rasterio.open(self.tiffdata)

        # get source coordinate reference system, transform
        west, south, east, north = self.bounds
        src_height, src_width = dem_raster.shape
        src_crs = dem_raster.crs
        src_transform = transform.from_bounds(
            *self.bounds, src_width, src_height)
        src = dem_raster.read(1)

        # calculate destination coordinate reference system, transform
        dst_crs = self.utm_crs
        print('Projecting from', src_crs, 'to', dst_crs)
        # - get origin (the _upper_ left corner) from bounds
        orix, oriy = self.to_xy(north, west)
        origin = (orix, oriy)
        self.origin = origin
        dst_transform = transform.from_origin(*origin, dx, dy)
        # - get extents from lower right corner
        SE_x, SE_y = self.to_xy(south, east)
        Lx = SE_x - orix
        Ly = oriy - SE_y
        Nx = int(Lx / dx)
        Ny = int(Ly / dy)

        # reproject to uniform grid in the UTM CRS
        dem_array = np.empty((Ny, Nx))
        warp.reproject(src, dem_array,
                       src_transform=src_transform, src_crs=src_crs,
                       dst_transform=dst_transform, dst_crs=dst_crs,
                       resampling=resampling)
        utmx = orix + np.arange(0, Nx * dx, dx)
        utmy = oriy + np.arange((-Ny + 1) * dy, dy, dy)
        self.x, self.y = np.meshgrid(utmx, utmy, indexing='ij')
        self.z = np.flipud(dem_array).T

        self.zfun = RectBivariateSpline(utmx, utmy, self.z)
        self.have_terrain = True

        return self.x, self.y, self.z

    def to_latlon(self, x, y):
        """Transform uniform grid to lat/lon space"""
        if not hasattr(x, '__iter__'):
            assert ~hasattr(x, '__iter__')
            x = [x]
            y = [y]
        xlon, xlat = warp.transform(self.utm_crs,
                                    self.latlon_crs,
                                    x, y)
        try:
            shape = x.shape
        except AttributeError:
            xlat = xlat[0]
            xlon = xlon[0]
        else:
            xlat = np.reshape(xlat, shape)
            xlon = np.reshape(xlon, shape)
        return xlat, xlon

    def to_xy(self, lat, lon, xref=None, yref=None):
        """Transform lat/lon to UTM space"""
        if not hasattr(lat, '__iter__'):
            assert ~hasattr(lat, '__iter__')
            lat = [lat]
            lon = [lon]
        x, y = warp.transform(self.latlon_crs,
                              self.utm_crs,
                              lon, lat)
        try:
            shape = lon.shape
        except AttributeError:
            x = x[0]
            y = y[0]
        else:
            x = np.reshape(x, shape)
            y = np.reshape(y, shape)
        if xref is not None:
            x -= xref
        if yref is not None:
            y -= yref
        return x, y


class SRTM(Terrain):
    """Class for working with Shuttle Radar Topography Mission (SRTM) data"""
    data_products = {
        'SRTM1': 30.0,
        'SRTM3': 90.0,
    }

    def __init__(self, latlon_bounds, fpath='terrain.tif', product='SRTM3',
                 margin=0.05):
        """Create container for SRTM data in the specified region
        Usage
        =====
        latlon_bounds : list or tuple
            Latitude/longitude corresponding to west, south, east, and
            north bounds, used to define the source transformation.
        fpath : str, optional
            Where to save downloaded GeoTIFF (*.tif) data.
        product : str, optional
            Data product name, SRTM1 or SRTM3 (corresponding to 30- and
            90-m DEM).
        margin : float, optional
            Decimal degree margin added to the bounds (default is 3")
            when clipping the downloaded elevation data.
        """
        latlon_bounds = list(latlon_bounds)
        if margin is not None:
            latlon_bounds[0] -= margin
            latlon_bounds[1] -= margin
            latlon_bounds[2] += margin
            latlon_bounds[3] += margin
        super().__init__(latlon_bounds, fpath=fpath)
        assert (product in self.data_products.keys()), \
            'product should be one of ' + str(list(self.data_products.keys()))
        self.product = product
        self.margin = margin

    def download(self, cleanup=True):
        """Download the SRTM data in GeoTIFF format"""
        dpath = os.path.dirname(self.tiffdata)
        if not os.path.isdir(dpath):
            print('Creating path', dpath)
            os.makedirs(dpath)
        elevation.clip(self.bounds, product=self.product, output=self.tiffdata)
        if cleanup:
            elevation.clean()

    def to_terrain(self, dx=None, dy=None, resampling=warp.Resampling.bilinear):
        """Load geospatial raster data and reproject onto specified grid
        Usage
        =====
        dx,dy : float
            Grid spacings [m]. If dy is not specified, then uniform
            spacing is assumed.
        resampling : warp.Resampling value, optional
            See `list(warp.Resampling)`.
        """
        if dx is None:
            dx = self.data_products[self.product]
            print('Output grid at ds=', dx)
        if dy is None:
            dy = dx
        return super().to_terrain(dx, dy=dy, resampling=resampling)
