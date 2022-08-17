from math import sqrt, atan2, pi
from collections import OrderedDict
from logging import warning
from typing import Tuple, Dict
import warnings

from herbie.archive import Herbie
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from rasterio.crs import CRS, CRSError

from ssrs import raster
from ssrs.utils import construct_lonlat_mask


class HRRR:
    """
    This class provides a basic interface to HRRR GRIB2 files as
    downloaded by Herbie and accessed with xarray.
    """

    Albers_CRS = 'ESRI:102008'  # Albers Equal Area Conic

    proj_aliases = {
        'lambert_conformal_conic': 'lcc',
    }

    def __init__(self,
                 date: str = None,
                 valid_date: str = None,
                 fxx: int = 0,
                 projected_CRS=Albers_CRS):
        """
        Either `valid_date` or `date` should be specified. These are
        related through the forecast lead time (default=0 hrs), `fxx`:

            valid_date = date + fxx

        The available lead times depend on the selected model. See the
        Herbie class documentation for more information.

        This interface downloads HRRR "sfc" data products. More
        information can be found at
        https://home.chpc.utah.edu/~u0553130/Brian_Blaylock/HRRR_archive/hrrr_sfc_table_f00-f01.html,
        which was retrieved on 2022-08-12 and saved as "HRRR GRIB2 Tables.mht".

        Parameters
        ----------
        date: str
            The timestamp (in UTC) at which the model was initialized,
            in the format of YYYY-mm-dd HH:MM.

        valid_date: str
            The timestamp (in UTC) of a valid datetime within the
            forecast, in the format of YYYY-mm-dd HH:MM.

        fxx: int, optional
            The forecast lead time (in hours). Defaults to 0.

        projected_CRS: str, optional
            Projected coordinate reference system used for additional
            calculations. Defaults to Albers Equal Area Conic, valid
            for North America.
        """
        if date is None and valid_date is not None:
            self.hrrr = Herbie(valid_date=valid_date, model='hrrr',
                               product='sfc', fxx=fxx)
        elif date is not None and valid_date is None:
            self.hrrr = Herbie(date=date, model='hrrr', product='sfc', fxx=fxx)
        else:
            raise ValueError("Use `date` or `valid_date`")

        self.projected_CRS = projected_CRS

        # self.datasets is a cache for xarrays obtained from GRIB files
        # keys are the regular expressions that obtained the dataset,
        # values are the xarray datasets parsed from the GRIB files.

        self.datasets: Dict[str, xr.Dataset] = {}

    @staticmethod
    def nearest_pressures(h):
        """
        Find the nearest pressure below a given height. For example,
        Rawlins, WY is at 2083 m above sea level. That puts the closest
        allowed height below at 1314 m above sea level, so that is what
        this method returns.

        Parameters
        ----------
        h: float
            Height above sea level in meters

        Returns
        -------
        Dict[str, float]
            Key/value pairs that correspond to the heights and pressures
            found.
        """
        # Keys are heights above sea level in meters. Values are
        # pressures in mb. Using an OrderedDict to preserve the order of
        # the keys for subsequent lookups in numpy arrays.
        #
        # Heights that are keys in the self.height_to_mb dictionary

        height_to_mb = OrderedDict()
        height_to_mb[12675.] = 250
        height_to_mb[10812.] = 300
        height_to_mb[5920.] = 500
        height_to_mb[2940.] = 700
        height_to_mb[1314.] = 850
        height_to_mb[624.] = 925
        height_to_mb[0.] = 1000

        heights = np.array(list(height_to_mb.keys()))

        delta_h = heights - h
        closest_height_above = heights[delta_h >= 0][-1]
        closest_height_below = heights[delta_h < 0][0]
        closest_pressure_above = height_to_mb[closest_height_above]
        closest_pressure_below = height_to_mb[closest_height_below]

        return {
            'closest_height_above': closest_height_above,
            'closest_height_below': closest_height_below,
            'closest_pressure_above': closest_pressure_above,
            'closest_pressure_below': closest_pressure_below
        }


    @staticmethod
    def pressure_at_height(h, temp=0, p0=1000):
        """
        From:

        https://keisan.casio.com/exec/system/1224579725

        Approximate mapping pressures to height above sea level:

        P=250.13248052231262 mb, h=12674.674674674676 m (though this could be out of bounds of this equation)
        P=300.0804709297101 mb, h=10812.812812812812 m
        P=500.0986604144544 mb, h=5920.920920920921 m
        P=700.7509995357793 mb, h=2940.940940940941 m
        P=850.5269196845428 mb, h=1314.3143143143143 m
        P=925.3676002705938 mb, h=624.6246246246246 m
        P=1000.0 mb, h=0.0 m

        Parameters
        ----------
        h: float
            Height in meters for which the pressure is needed.

        temp: float
            The temperature in celsius.

        p0: float
            Pressure at baseline in millibars (mb)

        Returns
        -------
        float
            The pressure at that height.
        """
        return p0 * (1 - .0065*h/(temp+.0065*h+273.15)) ** 5.257

    @staticmethod
    def get_CRS_from_attrs(gribfile_projection, **kwargs):
        """Retrieves necessary information from the attributes of the
        'gribfile_projection' field of a GRIB2 xarray dataset to
        construct a CRS string.
        """
        try:
            attrs = gribfile_projection.attrs
        except AttributeError:
            attrs = gribfile_projection
        try:
            crs = CRS.from_wkt(attrs['crs_wkt'])
        except (KeyError, CRSError):
            # could not directly use the WKT string for some reason...
            proj_name = attrs['grid_mapping_name']
            mapping = dict(
                proj=HRRR.proj_aliases[proj_name],
                lon_0=attrs['longitude_of_central_meridian'],
                lat_0=attrs['latitude_of_projection_origin'],
                lat_1=attrs['standard_parallel'][0],
                lat_2=attrs['standard_parallel'][1],
                x_0=attrs['false_easting'],
                y_0=attrs['false_northing'],
            )
            for key,val in kwargs.items():
                if key in mapping.keys():
                    print('Overwriting',key,'with',val,'in CRS')
                mapping[key] = val
            crs = CRS.from_dict(mapping)
        return crs

    def get_xarray_for_regex(self, regex, remove_grib=False):
        """Calls Herbie to search the GRIB2 data for the requested
        field (described by regular expression) and construct an xarray
        Dataset. The resulting dataset is stored in dictionary of
        datasets and returned _by reference_.

        Parameters
        ----------
        regex: str
            The regular expression to match the desired messages in
            GRIB2 file.

        remove_grib: bool
            If False (the default), downloaded GRIB2 files are retained
            in a cache. If True, the GRIB2 files are deleted.

        Returns
        -------
        xarray.Dataset or List[xarray.Dataset]
            Returns one xarray dataset if the coordinates are the same
            for all variables requested. Returns a list of xarray
            Datasets if the requested variables have different
            coordinates.
        """
        if regex not in self.datasets:
            # Retrieve GRIB2 data with Herbie
            # Note: There is an issue with how Herbie handles regular
            # expressions with Pandas, and this context manager handles
            # those exceptions.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ds = self.hrrr.xarray(
                    regex,
                    remove_grib=remove_grib
                )

            # Convert longitude from degrees east with range (0,360) to
            # degrees east/west for +/- values with range (-180,180)
            lon = ds.coords['longitude'].values
            lon[np.where(lon > 180)] -= 360.
            ds.coords['longitude'] = (ds.coords['longitude'].dims, lon)

            self.datasets[regex] = ds

        return self.datasets[regex]

    def read_idx(self):
        """
        This is simply a pass-through to the underlying Herbie library.
        It returns the index dataframe.

        Returns
        -------
        pandas.DataFrame
            Returns the index for the GRIB2 file at the time being
            referenced.
        """
        return self.hrrr.read_idx()

    def wind_velocity_direction_at_altitude(
        self,
        center_lonlat: Tuple[float, float],
        ground_level_m=0.0,
        height_above_ground_m=80.0,
        extent_km_lat=6.0,
        extent_km_lon=6.0,
        fringe_deg_lat=0.03,
        fringe_deg_lon=0.03,
        projection=None,
        remove_grib: bool = False
    ):
        """
        This method retrieves UV values for wind and computes the speed
        and direction of the wind. The point of interest is described by
        the parameters to this method call, which dictates the field
        requested through the `get_xarray_for_regex` function.

        There are some quirks of how the altitude requested works.
        Either ground_level_m or height_above_ground_m should be
        specified.

        When the method first starts, it calculates the pressure of the
        isobar closest to (but not lower than) the ground level. That is
        consdered to be the wind speed at the highest height above
        ground. In the HRRR GRIB files, there are also UV data at 10m
        and 80m above ground level. This gives three altitudes of wind
        data that this method uses to find UV values:

        1. Highest: Closest pressure above ground
        2. Middle: 80 m above ground
        3. Lowest: 10 m above ground

        If 0 < height_above_ground_m < 45, the 10 m above ground wind
        values are used. If 45 <= height_above_ground_m < 100, the 80 m
        above ground wind values are used. If height_above_ground_m >
        100, then the closest pressure above ground is used. Also, if
        height_above_ground_m < 0, the closest pressure above ground is
        used. The final condition prevents errors when bad data with a
        negative height above ground is encountered.

        Parameters
        ----------
        center_lonlat: Tuple[float, float]
            Center of the area to query. Longitude is in degrees east.
        
        ground_level_m: float
            The height of the ground above sea level in meters at the
            point of interest.

        height_above_ground_m: float
            The height above the ground in meters at the point of
            interest.

        extent_km_lat: float
            The extent of the mask in the latitude direction in units of
            kilometers.

        extent_km_lon: float
            The extent of the mask in the longitude direction in units
            of kilometers.

        fringe_deg_lat: float
            The number of degrees in the latitude direction added to the
            edges of the extent in units of degrees.

        fringe_deg_lon: float
            The number of degrees in the longitude direction to add to
            edges of the extent in degrees.

        projection: str or None
            Projected coordinate reference system. If None, defaults to
            the initialized value in this HRRR class. If "native", then
            use the HRRR simulation projection as described in the GRIB
            file. Otherwise, this should be a valid CRS string.

        remove_grib: bool
            If True, the GRIB is deleted from the cache after it is
            accessed. If False, the cached copy is preserved.

        Returns
        -------
        Dict[str, float]
            Key, value pairs with the results of the wind speed and
            direction, lats/lons over which the averages of U and V were
            taken, the number of points the values were averaged over,
            and the grib field/message used to find the u and v values.
        """

        # Determine the altitude to query the HRRR file using rules
        # explained in the docstring.

        varname = '(U|V)GRD'
        if height_above_ground_m > 0.0 and height_above_ground_m < 45.0:
            grib_field = f'{varname}:10 m above ground:anl'
            u_data_var = 'u10'
            v_data_var = 'v10'
        elif height_above_ground_m >= 45.0 and height_above_ground_m < 100.0:
            grib_field = f'{varname}:80 m above ground:anl'
            u_data_var = 'u'
            v_data_var = 'v'
        else:
            nearest_pressures = self.nearest_pressures(ground_level_m)
            closest_pressure_above = nearest_pressures['closest_pressure_above']
            grib_field = f'{varname}:{closest_pressure_above} mb'
            u_data_var = 'u'
            v_data_var = 'v'

        # Read the cached or download a new GRIB file. Catch warnings
        # that Herbie generates when finding the GRIB data.

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            uv_grd = self.get_xarray_for_regex(grib_field,
                                               remove_grib=remove_grib)

        # Determine output CRS
        if projection is None:
            out_crs = self.projected_CRS
        elif projection == 'native':
            out_crs = self.get_CRS_from_attrs(uv_grd['gribfile_projection'])

        # Find x, y of the center location given
        center_lon, center_lat = center_lonlat
        center_x, center_y = raster.transform_coordinates(
            in_crs='EPSG:4326',
            out_crs=out_crs,
            in_x=center_lon,
            in_y=center_lat
        )

        # Create the selection mask.
        mask = self.centered_mask_at_coordinates(
            uv_grd,
            center_lonlat=center_lonlat,
            extent_km_lat=extent_km_lat,
            extent_km_lon=extent_km_lon,
            fringe_deg_lat=fringe_deg_lat,
            fringe_deg_lon=fringe_deg_lon
        )

        if projection == 'native':

            # We can create a regular grid with this transformation.
            # Don't use ds.where(mask), which will subset a region
            # that is not square in Cartesian space and introduce NaNs
            # everywhere else.

            # Create a subset based on limiting coordinates in the mask
            slicedict = {
                dim: slice(np.min(indices), np.max(indices))
                for dim,indices in zip(mask.dims, np.where(mask))
            }
            data_subset = uv_grd.isel(**slicedict)
            us = data_subset['u'].values
            vs = data_subset['v'].values

            # Calculate the number of points that were found
            dims = list(data_subset.sizes.values())
            n = np.prod(dims)

            # Transform the lats/lons to x, y
            lats = data_subset.coords['latitude'].values.ravel()
            lons = data_subset.coords['longitude'].values.ravel()
            xs, ys = raster.transform_coordinates(
                in_crs='EPSG:4326',
                out_crs=out_crs,
                in_x=lons,
                in_y=lats
            )

            # Get the Cartesian coordinates
            xs = xs.reshape(dims)
            ys = ys.reshape(dims)
            y1d = ys[:,0]
            x1d = xs[0,:]

            # Sanity check: Are we in Cartesian (not curvilinear) space?
            dx = np.diff(x1d)
            dy = np.diff(y1d)
            assert np.allclose(dx, dx[0])
            assert np.allclose(dy, dy[0])
            assert np.allclose(x1d, xs[-1,:])
            assert np.allclose(y1d, ys[:,-1])

            # Set the dimension coordinate
            xs = x1d
            ys = y1d
            data_subset = data_subset.assign_coords(y=ys, x=xs)

            # Use bilinear interpolation to find U and V values
            u_interp = data_subset['u'].interp(x=center_x, y=center_y)
            v_interp = data_subset['v'].interp(x=center_x, y=center_y)
            u_interp = float(np.squeeze(u_interp.values))
            v_interp = float(np.squeeze(v_interp.values))

        else:

            # Extract that lats and lons that were found in the mask
            lats_data_array = uv_grd.coords['latitude'].where(mask)
            lons_data_array = uv_grd.coords['longitude'].where(mask)

            # Transform the lats/lons to x, y
            lats = np.array(lats_data_array).ravel()
            lons = np.array(lons_data_array).ravel()
            lats = lats[~np.isnan(lats)]
            lons = lons[~np.isnan(lons)]
            xs, ys = raster.transform_coordinates(
                in_crs='EPSG:4326',
                out_crs=out_crs,
                in_x=lons,
                in_y=lats
            )

            # Calculate the number of points that were found
            n = float(mask.sum())

            # Mask the u and v values
            us = uv_grd[u_data_var].where(mask, drop=True).values.flatten()
            vs = uv_grd[v_data_var].where(mask, drop=True).values.flatten()
            us = us[~np.isnan(us)]
            vs = vs[~np.isnan(vs)]

            # Use linear B-spline interpolation to find U and V values
            pts = np.stack([xs,ys], axis=-1)
            xi = (center_x, center_y)
            u_interp = float(np.squeeze(griddata(pts, us, xi, method='linear')))
            v_interp = float(np.squeeze(griddata(pts, vs, xi, method='linear')))

        # Calculate wind speed and direction, given easterly and
        # northerly velocity components, u and v, respectively
        speed = sqrt(u_interp**2 + v_interp**2)
        direction_deg = 180. \
                      + np.degrees(np.arctan2(u_interp, v_interp)) 

        return {
            'speed': speed,
            'direction_deg': direction_deg,
            'u_interp': u_interp,
            'v_interp': v_interp,
            'center_lat': center_lat,
            'center_lon': center_lon,
            'center_x': center_x,
            'center_y': center_y,
            'lats': lats,
            'lons': lons,
            'xs': xs,
            'ys': ys,
            'us': us,
            'vs': vs,
            'grib_field': grib_field,
            'n': n
        }

    def convective_velocity_xarray(self):
        """
        Retrieves the HRRR variables that allow convective velocity to
        be calcuated.

        Performs additional conversions and also outputs the final wstar
        value.

        Returns
        -------
        xarray.Dataset
            A unified dataset with the all the variables in one
            hypercube.
        """
        data = self.get_xarray_for_regex(
            ':(HPBL|POT|SHTFL|LHTFL|GFLUX):',
            remove_grib=False
        )
        data = xr.combine_by_coords(data)

        g = 9.81    # m/s^2
        rho = 1.225 # kg/m^3
        cp = 1005   # J/(kg*K)

        # Energy budget
        sensible = data['shtfl']/(rho*cp)
        latent   = data['lhtfl']/(rho*cp)
        gflux    = data['gflux']/(rho*cp)
        qs_Kms = sensible + latent - gflux
        
        # Get wstar of convective conditions
        qs_Kms = qs_Kms.clip(min=0)

        data['wstar'] = ( g * data.hpbl * qs_Kms / data.pt )**(1/3)

        return data

    @staticmethod
    def km_to_deg(extent_km, radius_of_earth_km=6371.):
        """Convert extent in meters to degrees

        From Deziel, Chris. "How to Convert Distances From Degrees to Meters"
        sciencing.com, https://sciencing.com/convert-distances-degrees-meters-7858322.html.
        7 April 2022.
        """
        return extent_km * 360 / (2 * pi * radius_of_earth_km)

    @staticmethod
    def centered_mask_at_coordinates(
        data,
        center_lonlat,
        extent_km_lat=3.0,
        extent_km_lon=3.0, 
        fringe_deg_lat=0.15, 
        fringe_deg_lon=0.9
    ):
        """
        Parameters
        ----------
        data: xarray.Dataset
            The dataset being masked.

        center_lonlat: Tuple[float, float]
            The center of the latitude and longitude to retrieve.

        extent_km_lat: float
            The extent of the mask in the latitude direction in units of
            kilometers.

        extent_km_lon: float
            The extent of the mask in the longitude direction in units
            of kilometers.

        fringe_deg_lat: float
            The number of degrees in the latitude direction added to the 
            edges of the extent in units of degrees.

        fringe_deg_lon: float
            The number of degrees in the longitude direction to add to
            edges of the extent in degrees.

        Returns
        -------
        xarray.core.dataarray.DataArray
            The mask to be used with the coordinates of the xarray
            dataset.
        """
        center_lon, center_lat = center_lonlat
        extent_deg_lat = HRRR.km_to_deg(extent_km_lat)
        extent_deg_lon = HRRR.km_to_deg(extent_km_lon)

        min_lat = center_lat - (extent_deg_lat / 2.0) - (fringe_deg_lat / 2.0)
        max_lat = center_lat + (extent_deg_lat / 2.0) + (fringe_deg_lat / 2.0)
        min_lon = center_lon - (extent_deg_lon / 2.0) - (fringe_deg_lon / 2.0)
        max_lon = center_lon + (extent_deg_lon / 2.0) + (fringe_deg_lon / 2.0)

        return construct_lonlat_mask(
            data.coords['longitude'],
            data.coords['latitude'],
            min_lon, max_lon,
            min_lat, max_lat
        )
    
    @staticmethod
    def mask_at_coordinates(
        data,
        southwest_lonlat,
        extent_km_lat=3.0,
        extent_km_lon=3.0, 
        fringe_deg_lat=0.15, 
        fringe_deg_lon=0.9
    ):
        """
        Parameters
        ----------
        data: xarray.Dataset
            The dataset being masked.

        southwest_lonlat: Tuple[float, float]
            The southwest corner of the latitude and longitude to
            retrieve.

        extent_km_lat: float
            The extent of the mask in the latitude direction in units of
            kilometers.

        extent_km_lon: float
            The extent of the mask in the longitude direction in units
            of kilometers.

        fringe_deg_lat: float
            The number of degrees in the latitude direction added to the 
            edges of the extent in units of degrees.

        fringe_deg_lon: float
            The number of degrees in the longitude direction to add to
            edges of the extent in degrees.

        Returns
        -------
        xarray.core.dataarray.DataArray
            The mask to be used with the coordinates of the xarray
            dataset.
        """
        southwest_lon, southwest_lat = southwest_lonlat
        extent_deg_lat = HRRR.km_to_deg(extent_km_lat)
        extent_deg_lon = HRRR.km_to_deg(extent_km_lon)

        min_lat = southwest_lat - fringe_deg_lat
        min_lon = southwest_lon - fringe_deg_lon
        max_lat = southwest_lat + extent_deg_lat + fringe_deg_lat
        max_lon = southwest_lon + extent_deg_lon + fringe_deg_lon

        return construct_lonlat_mask(
            data.coords['longitude'],
            data.coords['latitude'],
            min_lon, max_lon,
            min_lat, max_lat
        )

    def get_convective_velocity(self,
        southwest_lonlat=(-106.21, 42.78),
        extent=None,
        res=50
    ):
        """
        Returns the convective velocity.

        Parameters
        ----------
        southwest_lonlat: Tuple[float, float]
            The southwest corner of the latitude and longitude to
            retrieve.
        extent: Tuple[float, float, float, float]
            Domain extents xmin, ymin, xmax, ymax. If none is provided,
            the function returns an xarray on lat/lon on an irregular
            grid. If extent and res are provided, a grid is created and
            values interpolatd on that grid is returned, alongside the
            meshgrid values.
        res: float
            Resolution of the grid the HRRR data will be interpolated
            onto.

        Returns
        -------
        If extent is given:
            wstar: xarray.Dataset
                A dataset containing the calculated wstar value with
                coordinates lat/lon 

        Else:
            wstar: np.array
                An array of wstar interpolated onto a regular grid xx,yy
            xx, yy: np.array
                Grid in meshgrid format
        """

        # Get the variables for calculating convective velocity
        data = self.get_xarray_for_regex(
            ':(HPBL|POT|SHTFL|LHTFL|GFLUX):',
            remove_grib=False
        )
        data = xr.combine_by_coords(data)

        g = 9.81    # m/s^2
        rho = 1.225 # kg/m^3
        cp = 1005   # J/(kg*K)

        # Energy budget
        sensible = data['shtfl']/(rho*cp)
        latent   = data['lhtfl']/(rho*cp)
        gflux    = data['gflux']/(rho*cp)
        qs_Kms = sensible + latent - gflux
        
        # Get wstar of convective conditions
        qs_Kms = qs_Kms.clip(min=0)

        # Calculate wstar
        data['wstar'] = ( g * data.hpbl * qs_Kms / data.pt )**(1/3)


        if southwest_lonlat is None:
            southwest_lonlat = (-106.21, 42.78)   # TOTW

        mask = self.mask_at_coordinates(
            data,
            southwest_lonlat=southwest_lonlat
        )

        wstar = data['wstar'].where(mask, drop=True)

        if extent is not None:
            return self.convert_to_regular_grid(wstar,
                                                southwest_lonlat,
                                                extent,
                                                res)

        return wstar
    
    
    
    def get_albedo(self, southwest_lonlat=None, extent=None, res=50):
        """
        Returns the albedo.
        alpha = shortwave rad upward / shortwave rad downward

        Parameters
        ----------
        southwest_lonlat: Tuple[float, float]
            The southwest corner of the latitude and longitude to
            retrieve.  This parameter defaults to None. If this default
            is used, this value is set to (-106.21, 42.78) within the
            method.
        extent: Tuple[float, float, float, float]
            Domain extents xmin, ymin, xmax, ymax. If none is provided,
            the function returns an xarray on lat/lon on an irregular
            grid. If extent and res are provided, a grid is created and
            values interpolatd on that grid is returned, alongside the
            meshgrid values.
        res: float
            Resolution of the grid that the HRRR data will be
            interpolated onto.

        Returns
        -------
        albedo: np.array
            An array of albedo interpolated onto a regular grid xx, yy
        xx, yy: np.array
            Grid in meshgrid format
        """

        if southwest_lonlat is None:
            southwest_lonlat = (-106.21, 42.78)   # TOTW

        # Get the variables for calculating albedo, shortwave radiation
        Su, xx, yy = self.get_single_var_on_grid(
            ':USWRF:surface',  # shortwave upward
            southwest_lonlat,
            extent,
            res
        )
        Sd, xx, yy = self.get_single_var_on_grid(
            ':DSWRF:surface',  # shortwave downward
            southwest_lonlat,
            extent,
            res
        )

        if np.mean(Su) == 0:
            alpha_surface_albedo = np.ones_like(Su) # night
            # TODO: this is a placeholder. We should just compute the
            # albedo at noon of the same day. Setting it to 1 works
            # because upward radiation will only be zero at night, which
            # it is not a time of interest right now.
        else:
            alpha_surface_albedo = Su/Sd

        return alpha_surface_albedo, xx, yy


    @staticmethod
    def interp_onto_regular_grid(
        data,
        southwest_lonlat,
        bounds,
        res=50,
        out_crs=Albers_CRS
    ):
        from scipy.interpolate import griddata

        xSW, ySW = raster.transform_coordinates(
            in_crs='EPSG:4326',
            out_crs=out_crs,
            in_x=southwest_lonlat[0],
            in_y=southwest_lonlat[1]
        )
        
        # reference (0,0)
        xref = xSW[0]
        yref = ySW[0]

        # Get the transformed lat/long using the whole flattened array.
        # Remember to change long degrees E to W
        xform_long, xform_lat = raster.transform_coordinates(
            in_crs='EPSG:4326',
            out_crs=out_crs,
            in_x=data.longitude.values.flatten(),
            in_y=data.latitude.values.flatten()
        )

        # Now reshape them into the same form. These are in meshgrid format
        xform_long_sq = np.reshape(xform_long, np.shape(data.longitude.values))
        xform_lat_sq  = np.reshape(xform_lat,  np.shape(data.latitude.values))
        # Adjust reference point
        xform_long_sq = xform_long_sq - xref
        xform_lat_sq = xform_lat_sq - yref

        # create grid
        x = np.arange(bounds[0], bounds[2], res)
        y = np.arange(bounds[1], bounds[3], res)
        xx, yy = np.meshgrid(x, y, indexing='ij')

        # interpolate
        points = np.column_stack(
            (xform_long_sq.flatten(), xform_lat_sq.flatten())
        )
        values = np.array(data).flatten()
        data_interp = griddata(points, values, (xx, yy), method='linear')

        return data_interp, xx, yy


    def get_single_var_on_grid(self, regex, southwest_lonlat, bounds, res_m):
        """
        Designed to get a single variable, as defined by a regex
        expression, onto a regular grid.
        """
        return self.get_var_on_grid(regex, southwest_lonlat, bounds, res_m, [0])


    def get_var_on_grid(self,
        regex,
        southwest_lonlat,
        bounds,
        res_m,
        selected=None
    ):
        """
        Designed to get variable(s), as defined by a regex expression,
        onto a regular grid.

        Parameters
        ----------
        regex: str
            The regular expression to match the desired messages in
            GRIB2 file.

        southwest_lonlat: Tuple[float, float]
            The southwest corner of the latitude and longitude to
            retrieve.

        bounds: Tuple[float, float, float, float]
            With res_m, defines the regular grid to interpolate to the
            data onto: [xmin, ymin, xmax, ymax]; these values are
            relative to the projected x,y for the southwest corner of
            the region.

        res_m: float
            Spacing (in meters) of the regular grid onto which values
            are interpolated.

        selected: list or None
            Interpolate only selected variables, primarly for backwards
            compatibility.

        """

        # Get the data
        data = self.get_xarray_for_regex(regex, remove_grib=False)

        # Create a new dataset, copy selected coords and attrs
        data_interp = xr.Dataset()
        new_coords = {}
        for coord,val in data.coords.items():
            dims = data.coords[coord].dims
            if len(dims) == 0:
                new_coords[coord] = val
        data_interp = data_interp.assign_coords(new_coords)
        data_interp = data_interp.assign_attrs(data.attrs)

        # Create data mask
        mask = self.mask_at_coordinates(
            data,
            southwest_lonlat=southwest_lonlat,
            extent_km_lat=bounds[2] - min(0, bounds[0]),
            extent_km_lon=bounds[3] - min(0, bounds[1])
        )

        # Get list of data variables to interpolate
        # Note: Currently assumes 2D slice data with y,x dimensions
        datavars = [
            dvar for dvar in data.data_vars
            if data[dvar].dims == ('y','x')
        ]

        # Truncate the list of interpolated data variables if requested
        if selected is not None:
            tmp = []
            for ivar in selected:
                try:
                    tmp.append(datavars[ivar])
                except IndexError:
                    print('Requested var',ivar,'which does not exist')
            datavars = tmp

        # Interpolate all requested vars
        for varname in datavars:

            # Mask it based on latlon limits
            data_masked = data[varname].where(mask, drop=True)

            # Convert to an orthogonal grid
            fi, xx, yy = self.interp_onto_regular_grid(
                data_masked,
                southwest_lonlat,
                bounds,
                res_m,
                self.projected_CRS
            )

            data_interp[varname] = (('x','y'), fi)

        # Set dimension coordinate
        data_interp = data_interp.assign_coords(x=xx[:,0], y=yy[0,:])

        return data_interp, xx, yy

