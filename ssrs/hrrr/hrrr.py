from collections import OrderedDict
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

    def __init__(self, timestamp: str, fxx: int = 0):
        """
        Parameters
        ----------
        timestamp: str
            The timestamp of interest, in the format of YYYY-MM-DD HH:mm in UTC

        fxx: int
            The forecast hour of the model. Defaults to zero.
        """
        self.hrrr = Herbie(timestamp, model='hrrr', product='sfc', fxx=fxx)

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
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        isobar_1_mb: int,
        isobar_2_mb: int,
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

        # Get a mask for the closest points to the lat lon boundary provided
        latc = isobar_1.coords['latitude']
        lonc = isobar_1.coords['longitude']
        latc_mask = (latc >= min_lat) & (latc <= max_lat)
        lonc_mask = (lonc >= min_lon) & (lonc <= max_lon)
        mask = latc_mask & lonc_mask
        
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

        Returns
        -------
        xarray.Dataset
            A unified dataset with the all the variables in one hypercube.
        """
        data = self.get_xarray_for_regex(':(HPBL|POT|SHTFL|GFLUX):', remove_grib=False)
        data = xr.combine_by_coords(data)
        return data

    def convective_velcoity_variables(self, southwest_lonlat=None):
        """
        Parameters
        ----------
        southwest_lonlat: Tuple[float, float]
            The southwest corner of the latitude and longitude to retrieve.
            This parameter defaults to None. If this default is used, this
            value is set to (-106.21, 42.78) within the method.

        Returns
        -------
        Dict[str, xarray.Dataset]
            A dictionary with two keys: wstar_gflux_masked and wstar_shtfl_masked.
            Each value is an xarray.Dataset that corresponds to the masked values
            as referenced by the southwest lat/lon coordinates
        """
        # Get the variables for calculating convective velocity
        data = self.convective_velocity_xarray()

        if southwest_lonlat is None:
            southwest_lonlat = (-106.21, 42.78)   # TOTW

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

        wstar_gflux_masked = data['wstar_gflux'].where(mask, drop=True)
        wstar_shtfl_masked = data['wstar_shtfl'].where(mask, drop=True)

        return {
            'wstar_gflux_masked': wstar_gflux_masked,
            'wstar_shtfl_masked': wstar_shtfl_masked
        }
