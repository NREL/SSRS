""" Module for importing WindToolkit data from AWS or from NREL's HPC system"""

import os
import errno
from datetime import datetime
from typing import List, Tuple, Union
import pathos.multiprocessing as mp
import numpy as np
import pandas as pd
from .wtksource import WtkSource
from ..hrrr import HRRR


class WTK(WtkSource):
    """ Class for importing WTK data

    Parameters:
    ----------
    source_name: str
        Name of the source for WTK files, valid source names can
        be found in WTK.valid_sources
    lonlat_bounds: Tuple[float, float, float, float]
        Bounds in lon/lat crs for finding out WTK source points
    varnames: List of str
        Variables names to be imported from WTK
    out_dir: str
        Output directory where to save the dataframes in .csv
    padding: float, defaults to 0.02
        padding around lonlat bounds to ensure edges of bounds are within
        WTK coverage are
    """

    datetime_format: str = 'y%Ym%md%dh%H'

    def __init__(
        self,
        source_name: str,
        lonlat_bounds: Tuple[float, float, float, float],
        varnames: Union[List[str], str],
        out_dir: str,
        padding: float = 0.02 # deg
    ):

        # initiate
        super().__init__(source_name)
        self.out_dir = out_dir
        makedir_if_not_exists(self.out_dir)

        # pad the bounds
        pad = [-padding, -padding, padding, padding]
        self.lonlat_bounds = [ix + iy for ix, iy in zip(lonlat_bounds, pad)]

        # validate variable names
        varnames = [varnames] if isinstance(varnames, str) else varnames
        if self.valid_layers is not None:
            self.varnames = set(varnames).intersection(self.valid_layers)
            if len(self.varnames) > 0:
                print(('WTK: Downloading following layers:\n'
                       f'{chr(10).join(self.varnames)}'))
            else:
                raise ValueError(('WTK: No valid layer found among:\n'
                                  f'{chr(10).join(varnames)}\n'))

    def validate_requested_time(
        self,
        req_time: datetime
    ) -> None:
        """ Validate if the timestamp is available from the WTK source """
        if not isinstance(req_time, datetime):
            raise ValueError('Provide a valid datetime.datetime object')
        if req_time.year not in self.years:
            raise ValueError((f'{req_time.year} not found in '
                              f'{self.years}'))

    def download_locations(self) -> None:
        """ Download wtk locations within the set bounds """
        fpath = os.path.join(self.out_dir, 'wtk_locations.csv')
        with self.hsds.File(self.file_names[0], mode='r') as f_obj:
            # print(f'WTK: Available variables: {self.layers}')
            ts_lat_all = f_obj['coordinates'][:, 0]
            lat_index_all = np.where(np.logical_and(
                ts_lat_all > self.lonlat_bounds[1],
                ts_lat_all < self.lonlat_bounds[3])
            )[0]
            ts_lon = f_obj['coordinates'][min(
                lat_index_all):max(lat_index_all), 1]
        ts_lat = ts_lat_all[min(lat_index_all):max(lat_index_all)]
        lat_bool = np.logical_and(ts_lat > self.lonlat_bounds[1],
                                  ts_lat < self.lonlat_bounds[3])
        lon_bool = np.logical_and(ts_lon > self.lonlat_bounds[0],
                                  ts_lon < self.lonlat_bounds[2])
        wtk_ind = np.where(np.logical_and(lon_bool, lat_bool))[0]
        dfbase = pd.DataFrame({
            'Indices': min(lat_index_all) + wtk_ind,
            'Longitude': ts_lon[wtk_ind],
            'Latitude': ts_lat[wtk_ind]
        })
        dfbase.to_csv(fpath)

    def get_locations(self) -> pd.DataFrame:
        """ Returns dataframe containing lat/lon coordinate for wtk points """
        fpath = os.path.join(self.out_dir, 'wtk_locations.csv')
        if not os.path.isfile(fpath):
            self.download_locations()
        dfbase = pd.read_csv(fpath, index_col=0)
        if not (
            (dfbase['Longitude'].min() <= self.lonlat_bounds[0]) &
            (dfbase['Latitude'].min() <= self.lonlat_bounds[1]) &
            (dfbase['Longitude'].max() >= self.lonlat_bounds[2]) &
            (dfbase['Latitude'].max() >= self.lonlat_bounds[3])
        ):
            print(dfbase['Longitude'].min(), '<=?', self.lonlat_bounds[0])
            print(dfbase['Latitude'].min(),  '<=?', self.lonlat_bounds[1])
            print(dfbase['Longitude'].max(), '>=?', self.lonlat_bounds[2])
            print(dfbase['Latitude'].max(),  '>=?', self.lonlat_bounds[3])
            raise ValueError
        return dfbase

    def download_data_for_this_time(
        self,
        req_time: datetime
    ) -> pd.DataFrame:
        """Extracts WTK data for a given time and returns the dataframe"""

        if self.module_name == 'herbie':
            hrrr = HRRR(req_time)
            # TODO: the HRRR module should take the exact domain extent and CRS
            # as input, but they're currently not available to the WTK class
            approx_x_extent_km = (self.lonlat_bounds[2] - self.lonlat_bounds[0]) * 111.
            approx_y_extent_km = (self.lonlat_bounds[3] - self.lonlat_bounds[1]) * 111.
            print('Approx extent for HRRR retrieval [km]:',
                  approx_x_extent_km, approx_y_extent_km)
            vel80 = hrrr.wind_velocity_direction_at_altitude(
                lonlat=self.lonlat_bounds[:2], # SW corner
                centered_lonlat=False,
                extent_km_lon=approx_x_extent_km,
                extent_km_lat=approx_y_extent_km,
                height_above_ground_m=80.0)

            fpath = os.path.join(self.out_dir, 'wtk_locations.csv')
            newdf = pd.DataFrame({
                'Latitude': vel80['lats'],
                'Longitude': vel80['lons'],
            })
            newdf.to_csv(fpath)

            fpath = os.path.join(self.out_dir, self.get_filename(req_time))
            u = vel80['us']
            v = vel80['vs']
            newdf['ugrd'] = u
            newdf['vgrd'] = v
            newdf['windspeed_80m'] = np.sqrt(u**2 + v**2)
            newdf['winddirection_80m'] = 180. + np.degrees(np.arctan2(u, v))
            newdf.to_csv(fpath)

            return newdf

        # validate request
        self.validate_requested_time(req_time)
        time_str = req_time.strftime('%I %p, %d %b %Y')
        print(f'WTK: Downloading data for {time_str}', flush=True)

        # calculate time index of req time
        time_diff = req_time - datetime(req_time.year, 1, 1, 0)
        time_index = time_diff.days * 24 + time_diff.seconds // 3600

        # download the data
        newdf = self.get_locations()
        inds = newdf['Indices'].values
        source_fname = self.file_names[self.years.index(req_time.year)]
        with self.hsds.File(source_fname, mode='r') as fobj:
            for varname in self.varnames:
                try:
                    inorm = fobj[varname].attrs['scale_factor']
                    # wtk_units.append(fobj[varname].attrs['units'])
                    if self.module_name == 'h5pyd':
                        wtk_data_raw = fobj[varname][time_index, min(
                            inds):max(inds) + 1] / inorm
                        newdf[varname] = wtk_data_raw[inds - min(inds)]
                    else:
                        newdf[varname] = fobj[varname][time_index,
                                                       inds] / inorm
                except Exception as e_obj:
                    raise ValueError(
                        f'{varname} not found in {*list(fobj),}') from e_obj
        #  newdf.columns = pd.MultiIndex.from_tuples(
        #     zip(newdf.columns, wtk_units), names=('Variable', 'Unit'))
        fpath = os.path.join(self.out_dir, self.get_filename(req_time))
        newdf.to_csv(fpath)
        return newdf

    def get_dataframe_for_this_time(self, req_time: datetime) -> pd.DataFrame:
        """ Returns saved dataframe"""
        fpath = os.path.join(self.out_dir, self.get_filename(req_time))
        if self.module_name != 'herbie':
            dfbase = self.get_locations()
        try:
            newdf = pd.read_csv(fpath, index_col=0)
        except FileNotFoundError as _:
            print('WTK: Need to download first!')
            newdf = self.download_data_for_this_time(req_time)
        else:
            if (self.module_name != 'herbie') and \
                (not newdf['Indices'].equals(dfbase['Indices'])):
                raise FileNotFoundError
        return newdf

    def download_data(
        self,
        req_times: Union[List[datetime], datetime],
        max_cores: int = 1
    ) -> None:
        """ Downloads wtk data for all required datetimes"""
        req_times = [req_times] if isinstance(
            req_times, datetime) else req_times
        num_cores = min(len(req_times), max_cores)
        if num_cores > 1:
            with mp.Pool(num_cores) as pool:
                _ = pool.map(self.download_data_for_this_time, req_times)
        else:
            for i, _ in enumerate(req_times):
                self.download_data_for_this_time(req_times[i])

    def get_coordinates(self):
        """ returns lon and lat coordinates of wtk source points """
        dfbase = self.get_locations()
        return dfbase['Longitude'].values, dfbase['Latitude'].values

    def get_filename(self, at_time: datetime):
        """ Standard file naming for saving WTK data """
        return f'{at_time.strftime(self.datetime_format)}_wtk.csv'


def makedir_if_not_exists(dirname: str) -> None:
    """ Create the directory if it does not exists"""
    try:
        os.makedirs(dirname)
    except OSError as e_name:
        if e_name.errno != errno.EEXIST:
            raise
