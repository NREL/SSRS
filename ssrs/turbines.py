""" Module for dealing with wind turbine data from US Wind Turbine Database
(USWTB) dataset """

import os
from typing import Tuple, Optional
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
        out_dir: Optional[str] = None
    ):

        # load the USWTB turbine dataset in pandas dataframe
        print('TurbinesUSWTB: Importing turbine data from USWTB..')
        dfraw = pd.read_json(self.url)

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
        print(f'TurbinesUSWTB: Minimum hub height set to {min_hubheight} m')
        xbool = dfraw[self.__xcol].between(bounds[0], bounds[2], 'both')
        ybool = dfraw[self.__ycol].between(bounds[1], bounds[3], 'both')
        hhbool = dfraw['t_hh'].between(min_hubheight, 10000., 'left')
        # if wfarm_names is not None:
        #     namebool = dfraw['p_name'].isin(wfarm_names)
        # else:
        #     namebool = pd.Series([True] * xbool.shape[0], dtype=bool)
        self.df = dfraw.loc[xbool & ybool & hhbool, :]
        if out_dir is not None:
            fpath = os.path.join(out_dir, self.get_filename())
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            self.df.to_csv(fpath)

    @classmethod
    def get_filename(cls):
        """ Returns file name for saving the dataframe """
        return 'turbines.csv'

    def get_locations(self):
        """ Returns the locations of turbines """
        xylocs = self.df.loc[:, [self.__xcol, self.__ycol]].values
        return xylocs[:, 0], xylocs[:, 1]

    def get_locations_for_this_project(self, pname: str):
        """ Returns the locations of turbines """
        xlocs = self.df.loc[self.df['p_name'] == pname, self.__xcol].values
        ylocs = self.df.loc[self.df['p_name'] == pname, self.__ycol].values
        return xlocs, ylocs

    def get_project_names(self):
        """ Returns the wind power plant project names  """
        return self.df['p_name'].unique()

    def print_details(self):
        """ Prints basic details of the turbines within specified region """
        if self.df.shape[0] > 0:
            print(f'Number of projects: {self.df.p_name.nunique()}')
            print(f'Number of turbines: {self.df.shape[0]}')
            print((f'Hub height (min,median,max): {self.df.t_hh.min()}, '
                   f'{self.df.t_hh.median()}, {self.df.t_hh.max()}'))
            print((f'Rotor Dia (min,median,max): {self.df.t_rd.min()}, '
                   f'{self.df.t_rd.median()}, {self.df.t_rd.max()}'))
            print(f'    {"Project":<26}{"State":<6}{"Year":<6}' +
                  f'{"Count":<6}{"Hub_Hght":<10}{"Rotor_Dia":<10}')
            pnames = self.df.sort_values(
                by='t_rd', ascending=False).loc[:, 'p_name'].unique()
            for i, wfname in enumerate(pnames):
                ibool = self.df['p_name'] == wfname
                wf_state = self.df.loc[ibool, 't_state'].iloc[0]
                wf_year = int(self.df.loc[ibool, 'p_year'].iloc[0])
                wf_count = self.df[ibool].shape[0]
                wf_hh = self.df.loc[ibool, 't_hh'].median()
                wf_rd = self.df.loc[ibool, 't_rd'].median()
                wf_id = str(i + 1) + '.'
                print(f'{wf_id:<4}{wfname[:24]:<26}{wf_state:<6}{wf_year:<6}' +
                      f'{wf_count:<6}{wf_hh:<10}{wf_rd:<10}')
        else:
            print('TurbinesUSWTB: No wind turbines found within the bounds!')
