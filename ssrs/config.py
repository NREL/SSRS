""" Module for setting up dataclass defining the configuration settings for
SSRS"""

import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    """ Configuration parameters for SSRS simulation """

    # general parameters for the SSRS simulation
    run_name: str = 'default'  # name of this run, determines directory names
    out_dir: str = os.path.join(os.path.abspath(os.path.curdir), 'output')
    max_cores: int = 8  # maximum of cores to use
    sim_mode: str = 'uniform'  # snapshot, seasonal, uniform

    # parameters defining the terrain
    southwest_lonlat: Tuple[float, float] = (-106.21, 42.78)
    projected_crs: str = 'ESRI:102008'  # ESRI, EPSG, PROJ4 or WKT string
    region_width_km: Tuple[float, float] = (30., 20.)
    resolution: int = 100.  # desired terrain resolution (meters)

    # parameters for snapshot mode
    uniform_winddirn: float = 270.  # northerly = 0., easterly = 90, westerly=270
    uniform_windspeed: float = 10.  # uniform wind speed in m/s

    # parameters for seasonal mode
    seasonal_start: Tuple[int, int] = (3, 1)  # start of season (month, day)
    seasonal_end: Tuple[int, int] = (12, 1)  # end of season (month, day)
    seasonal_timeofday: str = 'daytime'  # morning, afternoon, evening, daytime
    seasonal_count: int = 8  # number of seasonal updraft computations

    # parameters for snapshot mode
    snapshot_datetime: Tuple[int, int, int, int] = (2010, 6, 17, 13)

    # downloading data from WTK
    wtk_source: str = 'AWS'  # 'EAGLE', 'AWS', 'EAGLE_LED'
    wtk_orographic_height: int = 100  # WTK wind conditions at this height
    wtk_thermal_height: int = 100  # WTK pressure, temperature, at this height
    wtk_interp_type: str = 'linear'  # 'nearest' 'linear' 'cubic'

    # parameters for simulating tracks
    track_direction: str = 'north'  # south,north,east,west,nwest,neast,swest,seast
    track_count: str = 100  # number of simulated eagle tracks
    track_start_region: Tuple[float, float, float, float] = (5, 45, 0, 0)
    track_start_type: str = 'random'  # uniform, random
    track_stochastic_nu: float = 1.  # scaling of move probs, 0 = random walk
    track_dirn_restrict: int = 2  # consideration of previous moves, options: 0,1,2

    # plotting related
    fig_height: float = 6.
    fig_dpi: int = 200  # increase this to get finer plots
    turbine_minimum_hubheight: float = 50.  # for select turbine locations
    turbine_mrkr_styles = ('1k', '2k', '3k', '4k',
                           '+k', 'xk', '*k', '.k', 'ok')
    turbine_mrkr_size: float = 3.
    turbine_box_around_wfarm: bool = False
    presence_smoothing_radius: bool = 10  # smoothing radius in meters

    # def __repr__(self):
    #     mystr = [f'{k}={v}' for k, v in self.__dict__.items()]
    #     return f'{chr(10).join(mystr)}'
