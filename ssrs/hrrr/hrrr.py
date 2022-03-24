from collections import OrderedDict
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

    def wind_uv_at_height(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        ground_height: float,
        height_above_ground: float,
        remove_grib: bool = False
    ):
        """
        Parameters
        ----------
        min_lat: float
            Minimum latitude

        min_lon: float
            Minimum longitude

        max_lat: float
            Maximum latitude

        max_lon: float
            Maximum longitude

        ground_height: float
            The ground level height at the point of calculation.

        height_above_ground: float
            The height above ground that 
        """
        height = ground_height + height_above_ground
        pressures = self.nearest_pressures(height)
        isobar_1_mb = pressures['closest_pressure_below']
        isobar_2_mb = pressures['closest_pressure_above']
        
        result = self.wind_uv(
            min_lat,
            min_lon,
            max_lat,
            max_lon,
            isobar_1_mb,
            isobar_2_mb,
            remove_grib
        )

        return result

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

    def wind_uv(
        self,
        southwest_lonlat: Tuple[float, float],
        isobar_1_mb: int,
        isobar_2_mb: int,
        remove_grib: bool = False
    ):
        """
        Parameters
        ----------
        southwest_lonlat: Tuple[float, float]
            The southwest corner of the area to query.

        isobar_1_mb: int
            This must match one of the available HRRR GRIB fields. These are
            250 mb, 300 mb, 500 mb, 700 mb, 850 mb, 925 mb, 1000 mb.

        isobar_2_mb: int
            This must match one of the available HRRR GRIB fields. These are
            250 mb, 300 mb, 500 mb, 700 mb, 850 mb, 925 mb, 1000 mb.

        remove_grib: bool
            If True, the GRIB is deleted from the cache after it is
            accessed. If False, the cached copy is preserved.

        Returns
        -------
        dict
            Dictionary with what was found.
        """
        # Field names for the isobars
        isobar_field_1 = f'(U|V)GRD:{isobar_1_mb} mb'
        isobar_field_2 = f'(U|V)GRD:{isobar_2_mb} mb'

        # Download the HRRR data for both isobars. There is a warning in how
        # Herbie uses regular expressions that is suppressed here.

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            isobar_1 = self.hrrr.xarray(isobar_field_1, remove_grib=remove_grib)
            isobar_2 = self.hrrr.xarray(isobar_field_2, remove_grib=remove_grib)

        # Both isobars have the same coordinate mask, so just compute
        # the mask on isobar1.

        mask = self.mask_at_coordinates(isobar_1, southwest_lonlat)

        # Extract that lats and lons that were found
        lats_data_array = isobar_1.coords['latitude'].where(mask)
        lons_data_array = isobar_1.coords['longitude'].where(mask)
        lats = np.array(lats_data_array).flatten()
        lons = np.array(lons_data_array).flatten()
        lats = lats[~np.isnan(lats)]
        lons = lons[~np.isnan(lons)]

        # Calculate the number of points that were found
        n = mask.sum()

        # Mask the u, v values for both isobars
        u_points_isobar_1 = isobar_1['u'].where(mask)
        v_points_isobar_1 = isobar_1['v'].where(mask)
        u_points_isobar_2 = isobar_2['u'].where(mask)
        v_points_isobar_2 = isobar_2['v'].where(mask)

        # Average the u, v values
        u_1 = u_points_isobar_1.sum() / n
        v_1 = v_points_isobar_1.sum() / n
        u_2 = u_points_isobar_2.sum() / n
        v_2 = v_points_isobar_2.sum() / n

        # Make tuples of the data
        uv_1 = float(u_1.values), float(v_1.values)
        uv_2 = float(u_2.values), float(v_2.values)

        return {
            'uv_1': uv_1,
            'uv_2': uv_2,
            'isobar_1_mb': isobar_1_mb,
            'isobar_2_mb': isobar_2_mb,
            'lats_deg_north': lats,
            'lons_deg_east': lons,
            'n': float(n.values)
        }

    def convective_velocity_xarray(self):
        """
        Retrieves the HRRR variables that allow convective velocity to be 
        calcuated. These variables are HPBL, POT, SHTFL, GFLUX.

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

        # Heat flux is given in W/m2. To convert it to K-m/s, divide it by rho*cp
        # We are only interested in wstar of convective times, hence the clip
        #data['gflux_Kms'] = (data['gflux']/(rho*cp)).clip(min=0)
        # Calculate wstar
        #data['wstar'] = ( g * data.hpbl * data.gflux_Kms / data.pt )**(1/3)

        data['wstar'] = ( g * data.hpbl * qs_Kms / data.pt )**(1/3)

        return data

    @staticmethod
    def mask_at_coordinates(data, southwest_lonlat):
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

        # create mask to get values around the region of interest. Arbitrarily setting 0.8 degrees
        xSW, ySW = raster.transform_coordinates('EPSG:4326','ESRI:102008', southwest_lonlat[0], southwest_lonlat[1])
        
        # reference (0,0)
        xref = xSW[0]
        yref = ySW[0]

        # longitude in degrees East (unusual; for GRIB)
        min_lat = southwest_lonlat[1] - 0.15
        min_lon = 180 - southwest_lonlat[0] - 0.9
        max_lat = min_lat + 0.7
        max_lon = min_lon + 1.1

        # longitude in degrees West (typical)
        min_lon_degW = southwest_lonlat[0] - 0.1
        max_lon_degW = min_lon_degW + 0.8

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

        # Heat flux is given in W/m2. To convert it to K-m/s, divide it by rho*cp
        # We are only interested in wstar of convective times, hence the clip
        #data['gflux_Kms'] = (data['gflux']/(rho*cp)).clip(min=0)
        # Calculate wstar
        #data['wstar'] = ( g * data.hpbl * data.gflux_Kms / data.pt )**(1/3)

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
        If extent is given:
        wstar: xarray.Dataset
            A dataset containing the calculated albedo value with coordinates
            lat/lon 
        Else:
        wstar: np.array
            An array of albedo interpolated onto a regular grid xx, yy
        xx, yy: np.array
            Grid in meshgrid format
        """

        # Get the variables for calculating
        data = self.get_xarray_for_regex(':(USWRF|DSWRF):surface', remove_grib=False)

        # Shortwave radiation
        Su  = data['uswrf']
        Sd  = data['dswrf']
        
        if np.mean(Su) == 0:
            data['alpha_surface_albedo'] = np.ones_like(Su) # night
        else:
            data['alpha_surface_albedo'] = Su/Sd

        if southwest_lonlat is None:
            southwest_lonlat = (-106.21, 42.78)   # TOTW

        mask = self.mask_at_coordinates(data, southwest_lonlat=southwest_lonlat)

        alpha_surface_albedo = data.where(mask, drop=True)

        if extent is not None:
            return  self.convertToRegularGrid(alpha_surface_albedo, southwest_lonlat, extent, res)

        return alpha_surface_albedo

    
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







