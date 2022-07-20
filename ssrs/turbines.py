""" Module for dealing with wind turbine data from US Wind Turbine Database
(USWTB) dataset """

import os
from typing import Tuple, Optional
from urllib.error import HTTPError
import pandas as pd
from .raster import transform_coordinates


class TurbinesUSWTB:
    """ Class for dealing with turbine data from the US Wind Turbine Database
        dataset supplied by USGS.

    Parameters:
    ----------
    bounds: Tuple[float, float, float, float]
        Bounds for which the turbine data needs to be extracted
    crs_string: str, defaults to 'EPSG:4326' the geo ref system
        Defines the coordinate ref system of the user-supplied bounds
        could be an EPSG, ESRI, PROJ4, or WKT string
    min_hubheight: float, defaults to 50.0 meters
        Minimum value of turbine hub height, for filtering the data
    wfarm_names: List of strings, defaults to None
        Specific names of wind power plants that needs to be in the data
        to get names of projects use large bounds, and print df.p_name.unique()
    print_details: bool, defaults to True
        if filtered data needs to be printed
    """

    url = ('https://eersc.usgs.gov/api/uswtdb/v1/turbines?&t_cap=gt.0&'
           'select=t_state,p_name,p_year,t_cap,t_hh,t_rd,xlong,ylat')
    lonlat_crs = 'EPSG:4326'  # lon/lat crs

    def __init__(
        self,
        bounds: Tuple[float, float, float, float],
        crs_string: str = 'EPSG:4326',
        min_hubheight: float = 50.,  # in meters
        out_fpath: str = 'turbines.csv',
        print_verbose: bool = False
    ):
        try:
            self.dframe = pd.read_csv(out_fpath)
            print('TurbinesUSWTDB: Loaded saved turbine data from USWTDB')
        except (ValueError, FileNotFoundError, pd.errors.ParserError):
            self._download_db(bounds, crs_string, min_hubheight, out_fpath)
        if print_verbose:
            self.print_details()

    def _download_db(
        self,
        bounds: Tuple[float, float, float, float],
        crs_string: str = 'EPSG:4326',
        min_hubheight: float = 50., # in meters
        out_fpath: str = 'turbines.csv',
    ):
        # load the USWTB turbine dataset in pandas dataframe
        print('TurbinesUSWTDB: Importing turbine data from USWTDB..')
        try:
            dfraw = pd.read_json(self.url)
            print('TurbinesUSWTDB: Successfully imported turbine data from USWTDB..')
        except HTTPError as httperr:
            print(f'Connection issues with USWTB database (error code {httperr.code}; ' \
                   'see codes at https://eerscmap.usgs.gov/uswtdb/api-doc/).')
            self.dframe = None
        except Exception as _:
            print('Unknown connection issues with USWTDB database. Please submit a bug report.')
            self.dframe = None
        else:
            # compute the turbine locations in projected crs, if needed
            if crs_string.lower() != 'epsg:4326':
                self.__xcol = 'x'
                self.__ycol = 'y'
                xlocs, ylocs = transform_coordinates(
                    self.lonlat_crs,
                    crs_string,
                    dfraw['xlong'].values,
                    dfraw['ylat'].values
                )
                dfraw[self.__xcol] = xlocs
                dfraw[self.__ycol] = ylocs
            else:
                self.__xcol = 'xlong'
                self.__ycol = 'ylat'

            # find the turbines within the requested bounds
            xbool = dfraw[self.__xcol].between(bounds[0], bounds[2], 'both')
            ybool = dfraw[self.__ycol].between(bounds[1], bounds[3], 'both')
            hhbool = dfraw['t_hh'].between(min_hubheight, 10000., 'left')
            self.dframe = dfraw.loc[xbool & ybool & hhbool, :]
            if out_fpath is not None:
                self.dframe.to_csv(out_fpath)

    def get_locations(self):
        """ Returns the locations of turbines """
        xylocs = self.dframe.loc[:, [self.__xcol, self.__ycol]].values
        return xylocs[:, 0], xylocs[:, 1]

    def get_locations_for_this_project(self, pname: str):
        """ Returns the locations of turbines """
        xlocs = self.dframe.loc[self.dframe['p_name']
                                == pname, self.__xcol].values
        ylocs = self.dframe.loc[self.dframe['p_name']
                                == pname, self.__ycol].values
        return xlocs, ylocs

    def get_project_names(self):
        """ Returns the wind power plant project names  """
        return self.dframe['p_name'].unique()

    def print_details(self):
        """ Prints basic details of the turbines within specified region """
        if self.dframe.shape[0] > 0:
            print(f'Number of projects: {self.dframe.p_name.nunique()}')
            print(f'Number of turbines: {self.dframe.shape[0]}')
            print((f'Hub height (min,median,max): {self.dframe.t_hh.min()}, '
                   f'{self.dframe.t_hh.median()}, {self.dframe.t_hh.max()}'))
            print((f'Rotor Dia (min,median,max): {self.dframe.t_rd.min()}, '
                   f'{self.dframe.t_rd.median()}, {self.dframe.t_rd.max()}'))
            print(f'    {"Project":<26}{"State":<6}{"Year":<6}' +
                  f'{"Count":<6}{"Hub_Hght":<10}{"Rotor_Dia":<10}')
            pnames = self.dframe.sort_values(
                by='t_rd', ascending=False).loc[:, 'p_name'].unique()
            for i, wfname in enumerate(pnames):
                ibool = self.dframe['p_name'] == wfname
                wf_state = self.dframe.loc[ibool, 't_state'].iloc[0]
                wf_year = int(self.dframe.loc[ibool, 'p_year'].iloc[0])
                wf_count = self.dframe[ibool].shape[0]
                wf_hh = self.dframe.loc[ibool, 't_hh'].median()
                wf_rd = self.dframe.loc[ibool, 't_rd'].median()
                wf_id = str(i + 1) + '.'
                print(f'{wf_id:<4}{wfname[:24]:<26}{wf_state:<6}{wf_year:<6}' +
                      f'{wf_count:<6}{wf_hh:<10}{wf_rd:<10}')
        else:
            print('TurbinesUSWTB: No wind turbines found within the bounds!')
