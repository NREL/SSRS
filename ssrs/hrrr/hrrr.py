from math import sqrt, atan2, pi
from collections import OrderedDict
from logging import warning
from typing import Tuple
import warnings

from herbie.archive import Herbie
import numpy as np
import xarray as xr

from ssrs import raster


class HRRR:
    """
    This class provides a basic interface to HRRR GRIB2 files as downloaded
    by Herbie and accessed with xarray.
    """

    def __init__(self, date: str = None, valid_date: str= None, fxx: int = 0):
        """
        Parameters
        ----------
        timestamp: str
            The timestamp of interest, in the format of YYYY-MM-DD HH:mm in UTC

        fxx: int
            The forecast hour of the model. Defaults to zero.
        """
        if date is None and valid_date is not None:
            print('Using valid_date')
            self.hrrr = Herbie(valid_date=valid_date, model='hrrr', product='sfc', fxx=fxx)
        elif date is not None and valid_date is None:
            print('Using date')
            self.hrrr = Herbie(date=date, model='hrrr', product='sfc', fxx=fxx)
        else:
            raise ValueError("Use `date` or `valid_date`")

    @staticmethod
    def nearest_pressures(h):
        """
        Find the nearest pressure below a given height. For example,
        Rawlins, WY is at 2083 m above sea level. That puts the
        closest allowed height below at 1314 m above sea level, so
        that is what this method returns.

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
        # Keys are heights above sea level in meters. Values are pressures
        # in mb. Using an OrderedDict to preserve the order of the keys for
        # subsequent lookups in numpy arrays.
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

    def get_xarray_for_regex(self, regex, remove_grib=False):
        """
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
            for all variables requested. Returns a list of xarray.Datasets
            if the requested variables have different coordinates.
        """

        # There is an issue with how Herbie handles regular expressions
        # with Pandas, and this context manager handles those exceptions.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = self.hrrr.xarray(regex, remove_grib=remove_grib)

        return result

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
        southwest_lonlat: Tuple[float, float],
        h: float,
        remove_grib: bool = False
    ):
        """
        Parameters
        ----------
        southwest_lonlat: Tuple[float, float]
            The southwest corner of the area to query.
        
        h: float
            Height above sea level 

        remove_grib: bool
            If True, the GRIB is deleted from the cache after it is
            accessed. If False, the cached copy is preserved.

        Returns
        -------
        Dict[str, float]
            Key, value pairs
        """
        nearest_pressures = self.nearest_pressures(h)
        closest_pressure_above = nearest_pressures['closest_pressure_above']
        uv_grd_field = f'(U|V)GRD:{closest_pressure_above} mb'

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            uv_grd = self.hrrr.xarray(uv_grd_field, remove_grib=remove_grib)

        mask = self.mask_at_coordinates(uv_grd, southwest_lonlat)

        # Extract that lats and lons that were found
        lats_data_array = uv_grd.coords['latitude'].where(mask)
        lons_data_array = uv_grd.coords['longitude'].where(mask)
        lats = np.array(lats_data_array).flatten()
        lons = np.array(lons_data_array).flatten()
        lats = lats[~np.isnan(lats)]
        lons = lons[~np.isnan(lons)]

        # Calculate the number of points that were found
        n = float(mask.sum())

        # Mask the u and v values
        u_points = uv_grd['u'].where(mask)
        v_points = uv_grd['v'].where(mask)
        
        # Average u, v values
        u = float(u_points.sum()) / n
        v = float(v_points.sum()) / n

        # Calculate wind speed and direction
        deg_per_radian = 57.296
        speed = sqrt(u**2 + v**2)
        direction_deg = atan2(u, v) * deg_per_radian  # Convert radians to degrees

        return {
            'speed': speed,
            'direction_deg': direction_deg,
            'closest_pressure_above': closest_pressure_above,
            'lats': lats,
            'lons': lons,
            'n': n
        }

    def convective_velocity_xarray(self):
        """
        Retrieves the HRRR variables that allow convective velocity to be 
        calcuated.

        Performs additional conversions and also outputs the final wstar value.

        Returns
        -------
        xarray.Dataset
            A unified dataset with the all the variables in one hypercube.
        """
        data = self.get_xarray_for_regex(':(HPBL|POT|SHTFL|LHTFL|GFLUX):', remove_grib=False)
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
    def mask_at_coordinates(data, southwest_lonlat, extent_km_lat=3.0, extent_km_lon=3.0, fringe_deg_lat=0.15, fringe_deg_lon=0.9):
        """
        Parameters
        ----------
        data: xarray.Dataset
            The dataset being masked.

        southwest_lonlat: Tuple[float, float]
            The southwest corner of the latitude and longitude to retrieve.

        Returns
        -------
        xarray.core.dataarray.DataArray
            The mask to be used with the coordinates of the xarray dataset.
        """

        # Convert extent in meters to degrees
        # From Deziel, Chris. "How to Convert Distances From Degrees to Meters" sciencing.com, https://sciencing.com/convert-distances-degrees-meters-7858322.html. 7 April 2022.
        radius_of_earth_km = 6371
        extent_deg_lat = extent_km_lat * 360 / (2 * pi * radius_of_earth_km)
        extent_deg_lon = extent_km_lon * 360 / (2 * pi * radius_of_earth_km)

        # Extract lon and lat
        southwest_lon, southwest_lat = southwest_lonlat

        # longitude in degrees East (unusual; for GRIB)
        min_lat = southwest_lat - fringe_deg_lat
        min_lon = 180 - southwest_lon - fringe_deg_lon
        max_lat = min_lat + extent_deg_lat + fringe_deg_lat
        max_lon = min_lon + extent_deg_lon + fringe_deg_lon

        # Construct the mask
        latc = data.coords['latitude']
        lonc = data.coords['longitude']
        latc_mask = (latc >= min_lat) & (latc <= max_lat)
        lonc_mask = (lonc >= min_lon) & (lonc <= max_lon)
        mask = latc_mask & lonc_mask

        return mask

    def get_convective_velocity(self, southwest_lonlat=None, extent=None, res=50):
        """
        Returns the convective velocity.

        Parameters
        ----------
        southwest_lonlat: Tuple[float, float]
            The southwest corner of the latitude and longitude to retrieve.
            This parameter defaults to None. If this default is used, this
            value is set to (-106.21, 42.78) within the method.
        extent: Tuple[float, float, float, float]
            Domain extents xmin, ymin, xmax, ymax. If none is provided, the function
            returns an xarray on lat/lon on an irregular grid. If extent and res
            are provided, a grid is created and values interpolatd on that grid 
            is returned, alongside the meshgrid values.
        res: float
            Resolution of the grid the HRRR data will be interpolatd onto.

        Returns
        -------
        If extent is given:
        wstar: xarray.Dataset
            A dataset containing the calculated wstar value with coordinates
            lat/lon 
        Else:
        wstar: np.array
            An array of wstar interpolated onto a regular grid xx, yy
        xx, yy: np.array
            Grid in meshgrid format
        """

        # Get the variables for calculating convective velocity
        data = self.get_xarray_for_regex(':(HPBL|POT|SHTFL|LHTFL|GFLUX):', remove_grib=False)
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

        mask = self.mask_at_coordinates(data, southwest_lonlat=southwest_lonlat)

        wstar = data['wstar'].where(mask, drop=True)

        if extent is not None:
            return  self.convertToRegularGrid(wstar, southwest_lonlat, extent, res)

        return wstar
    
    
    
    def get_albedo(self, southwest_lonlat=None, extent=None, res=50):
        """
        Returns the albedo.
        alpha = shortwave rad upward / shortwave rad downward

        Parameters
        ----------
        southwest_lonlat: Tuple[float, float]
            The southwest corner of the latitude and longitude to retrieve.
            This parameter defaults to None. If this default is used, this
            value is set to (-106.21, 42.78) within the method.
        extent: Tuple[float, float, float, float]
            Domain extents xmin, ymin, xmax, ymax. If none is provided, the function
            returns an xarray on lat/lon on an irregular grid. If extent and res
            are provided, a grid is created and values interpolatd on that grid 
            is returned, alongside the meshgrid values.
        res: float
            Resolution of the grid the HRRR data will be interpolatd onto.

        Returns
        -------
        albedo: np.array
            An array of albedo interpolated onto a regular grid xx, yy
        xx, yy: np.array
            Grid in meshgrid format
        """

        if southwest_lonlat is None:
            southwest_lonlat = (-106.21, 42.78)   # TOTW

        # Get the variables for calculating albedo, shortwave upward/downward radiation
        Su, xx, yy = self.getSingleVariableOnGrid(':USWRF:surface', southwest_lonlat, extent, res)   # short wave upward
        Sd, xx, yy = self.getSingleVariableOnGrid(':DSWRF:surface', southwest_lonlat, extent, res)   # short wave downward

        if np.mean(Su) == 0:
            alpha_surface_albedo = np.ones_like(Su) # night
            # TODO: this is a placeholder. We should just compute the albedo at noon of the same day.
            # Setting it to 1 works because upward radiation will only be zero at night, which it is
            # not a time of interest right now.
        else:
            alpha_surface_albedo = Su/Sd

        return alpha_surface_albedo, xx, yy

    
    @staticmethod
    def convertToRegularGrid(data, southwest_lonlat, extent, res=50):

        from scipy.interpolate import griddata

        xSW, ySW = raster.transform_coordinates('EPSG:4326','ESRI:102008', southwest_lonlat[0], southwest_lonlat[1])
        
        # reference (0,0)
        xref = xSW[0]
        yref = ySW[0]

        # Get the transformed lat/long using the whole flattened array. Remember to change long degrees E to W
        xform_long, xform_lat = raster.transform_coordinates('EPSG:4326','ESRI:102008', 180-data.longitude.values.flatten(),
                                                                                            data.latitude.values.flatten())
        # Now reshape them into the same form. These are in meshgrid format
        xform_long_sq = np.reshape(xform_long, np.shape(data .longitude.values))
        xform_lat_sq  = np.reshape(xform_lat,  np.shape(data.latitude.values))
        # Adjust reference point
        xform_long_sq = xform_long_sq - xref
        xform_lat_sq = xform_lat_sq - yref

        # create grid
        x = np.arange(extent[0], extent[2], res)
        y = np.arange(extent[1], extent[3], res)
        xx, yy = np.meshgrid(x, y, indexing='ij')

        # interpolate
        points = np.column_stack( (xform_long_sq.flatten(), xform_lat_sq.flatten()) )
        values = np.array(data).flatten()
        data_interp = griddata(points, values, (xx, yy), method='linear')

        return data_interp, xx, yy


    def getSingleVariableOnGrid(self, regex, southwest_lonlat, extent, res):
        '''
        Designed to get a single variable, as defined by a regex expression, onto a regular
        grid.

        '''

        # Get the data
        data = self.get_xarray_for_regex(regex, remove_grib=False)

        # Get the name of the varialbe related to the `regex`
        varname = list(data.data_vars)[0]

        # Mask it based on latlon limits
        mask = self.mask_at_coordinates(data, southwest_lonlat=southwest_lonlat)
        data_masked = data[varname].where(mask, drop=True)

        # Convert to an orthogonal grid
        data_interp, xx, yy =  self.convertToRegularGrid(data_masked, southwest_lonlat, extent, res)

        return data_interp, xx, yy







