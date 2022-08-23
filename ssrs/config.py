""" Module for setting up dataclass defining the configuration settings for
SSRS"""

import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    """Configuration parameters for SSRS simulation """

    # general parameters for the SSRS simulation
    run_name: str = 'default'  # name of this run, determines directory names
    out_dir: str = os.path.join(os.path.abspath(os.path.curdir), 'output')
    max_cores: int = 8  # maximum of cores to use
    sim_seed: int = -1  # random number seed
    sim_mode: str = 'uniform'  # snapshot, seasonal, uniform
    sim_movement: str = 'fluid-analogy' # fluid-analogy, heuristics
    sim_seed: int = -1 # random number seed

    print_verbose: bool = False  # if want to print verbose

    # H-SSRS parameters (used if `sim_movement` == 'heuristics')
    movement_ruleset: str = 'default' # TODO: add list of valid options
    random_walk_freq: int = 300 # if > 0, inverse of how often random walks will randomly occur -- approx every 1/random_walk_freq steps
    random_walk_step_size: float = 50.0 # when a random walk does occur, the distance traveled in each random movement
    random_walk_step_range: tuple = (None,None) # when a random walk does occur, the number of random steps will occur in this range
    look_ahead_dist: float = 2000.0
    thermal_intensity_scale: float = 2.0 # #1 gives weak random thermal field, 3 gives v strong random thermal field
    
    # parameters defining the terrain
    southwest_lonlat: Tuple[float, float] = (-106.21, 42.78)
    projected_crs: str = 'ESRI:102008'  # ESRI, EPSG, PROJ4 or WKT string
    region_width_km: Tuple[float, float] = (60., 50.)
    resolution: int = 100.  # desired terrain resolution (meters)

    # parameters for uniform mode
    uniform_winddirn: float = 270.  # northerly = 0., easterly = 90, westerly=270
    uniform_windspeed: float = 10.  # uniform wind speed in m/s

    # parameters for snapshot mode
    snapshot_datetime: Tuple[int, int, int, int] = (2010, 6, 17, 13)

    # parameters for seasonal mode
    seasonal_start: Tuple[int, int] = (3, 20)  # start of season (month, day)
    seasonal_end: Tuple[int, int] = (5, 15)  # end of season (month, day)
    seasonal_timeofday: str = 'daytime'  # morning, afternoon, evening, daytime
    seasonal_count: int = 8  # number of seasonal updraft computations

    # downloading data from WTK
    wtk_source: str = 'AWS'  # 'EAGLE', 'AWS', 'EAGLE_LED'
    wtk_orographic_height: int = 100  # WTK wind conditions at this height
    wtk_thermal_height: int = 100  # WTK pressure, temperature, at this height
    wtk_interp_type: str = 'linear'  # 'nearest' 'linear' 'cubic'

    # parameters defining the updraft calculation
    thermals_realization_count: bool = 0  # number of realizations of thermals
    updraft_threshold: float = 0.75  # only use updrafts higher than this
    movement_model: str = 'fluid-flow'  # fluid-flow, drw, heuristics

    # parameters for simulating tracks
    track_direction: float = 0  # movement direction measured clockwise from north
    track_count: str = 1000  # number of simulated eagle tracks
    track_start_region: Tuple[float, float, float, float] = (5, 55, 1, 2)
    track_start_type: str = 'structured'  # structured, random
    track_stochastic_nu: float = 1.  # scaling of move probs, 0 = random walk
    track_dirn_restrict: int = 1  # restrict within 45 deg of previous # moves

    # plotting related
    fig_height: float = 6.
    fig_dpi: int = 300  # increase this to get finer plots
    
    # turbine related
    turbine_minimum_hubheight: float = 50.  # for select turbine locations
    turbine_mrkr_styles = ('1k', '2k', '3k', '4k',
                           '+k', 'xk', '*k', '.k', 'ok')
    turbine_mrkr_size: float = 3.

    def __str__(self):
        out_str = self.__doc__ + '\n'
        for i, (k, _) in enumerate(self.__dict__.items()):
            if i == 0:
                out_str += '\n:::: General settings\n'
            elif i == 6:
                out_str += '\n:::: Terrain settings\n'
            elif i == 10:
                out_str += '\n:::: Uniform mode\n'
            elif i == 12:
                out_str += '\n:::: Snapshot mode\n'
            elif i == 13:
                out_str += '\n:::: Seasonal mode\n'
            elif i == 17:
                out_str += '\n:::: WindToolKit settings\n'
            elif i == 21:
                out_str += '\n:::: Updraft computation\n'
            elif i == 23:
                out_str += '\n:::: Simulating tracks\n'
            elif i == 30:
                out_str += '\n:::: Plotting and wind turbines\n'
            out_str += f'{k} = {self.__dict__[k]}\n'
        return out_str
