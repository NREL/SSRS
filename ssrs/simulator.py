""" Module for setting up SSRS """

import os
import json
import time
import pickle
import random
from typing import List, Tuple, Optional
from datetime import datetime
from dataclasses import asdict
import requests
#import multiprocessing as mp
import pathos.multiprocessing as mp
from itertools import repeat
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from scipy.interpolate import griddata
from scipy import ndimage
from matplotlib.colors import LogNorm, ListedColormap
from .terrain import Terrain
from .wtk import WTK
from .hrrr import HRRR
from .turbines import TurbinesUSWTB
from .config import Config
from .layers import (calcOrographicUpdraft, calcAspectDegrees,
                     calcSlopeDegrees, compute_random_thermals,
                     get_above_threshold_speed,
                     get_above_threshold_hard_cutoff,
                     blurQuantity,
                     calcSx, highRes2lowRes, compute_thermals_3d)
from .raster import (lonlat_crs,
                     get_raster_in_projected_crs,
                     transform_bounds, transform_coordinates)

from .movmodel import (generate_heuristic_eagle_track,
                       compute_smooth_presence_counts_HSSRS)
from .heuristics import rulesets
#from .randomthermals import est_random_thermals
from .movmodel import (MovModel,
                       compute_smooth_presence_counts,
                       generate_simulated_tracks)

from .utils import (makedir_if_not_exists, get_elapsed_time,
                    get_extent_from_bounds, empty_this_directory,
                    create_gis_axis, get_sunrise_sunset_time,
                    calc_MSE)


class Simulator(Config):
    """ Class for SSRS simulation """

    time_format = 'y%Ym%md%dh%H'

    def __init__(self, in_config: Config = None, **kwargs) -> None:
        # Initiate the config parameters
        if in_config is None:
            super().__init__(**kwargs)
        else:
            super().__init__(**asdict(in_config))

        print(f'\n---- SSRS in {self.sim_mode} mode')
        print(f'---- movements based on {self.sim_movement} model')
        print(f'---- using ruleset {self.movement_ruleset}')
        print(f'Run name: {self.run_name}')

        # Perform some checks
        self.checkInputs()

        # Re-init random number generator for results reproducibility
        if self.sim_seed >= 0:
            print('Specified random number seed:', self.sim_seed)
            np.random.seed(self.sim_seed)

        # Create directories for saving data and figures
        print(f'Output dir: {os.path.join(self.out_dir, self.run_name)}')
        self.data_dir = os.path.join(self.out_dir, self.run_name, 'data/')
        self.fig_dir = os.path.join(self.out_dir, self.run_name, 'figs/')
        self.mode_data_dir = os.path.join(self.data_dir, self.sim_mode)
        self.mode_fig_dir = os.path.join(self.fig_dir, self.sim_mode)
        for dirname in (self.mode_data_dir, self.mode_fig_dir):
            makedir_if_not_exists(dirname)

        # Save the config file
        fpath = os.path.join(self.out_dir, self.run_name,
                             f'{self.run_name}.json')
        with open(fpath, 'w', encoding='utf-8') as cfile:
            json.dump(self.__dict__, cfile, ensure_ascii=False, indent=2)

        # Determine grid size of the region
        print(f'Analysis resolution = {self.resolution} m')
        xsize = int(round((self.region_width_km[0] * 1000. / self.resolution)))
        ysize = int(round((self.region_width_km[1] * 1000. / self.resolution)))
        self.gridsize = (ysize, xsize)
        print(f'Analysis grid size = {self.gridsize} elements')

        # Determine high-res terrain grid size of the region
        print(f'Terrain resolution = {self.resolution_terrain} m')
        xsize_terrain = int(
            round((self.region_width_km[0] * 1000. / self.resolution_terrain)))
        ysize_terrain = int(
            round((self.region_width_km[1] * 1000. / self.resolution_terrain)))
        self.gridsize_terrain = (ysize_terrain, xsize_terrain)
        print(f'Terrain grid size = {self.gridsize_terrain} elements')

        # Determine bounds in both lon/lat and in projected crs
        if self.center_lonlat == (999,999):
            proj_west, proj_south = transform_coordinates(
                lonlat_crs, self.projected_crs,
                *self.southwest_lonlat)
            proj_west = proj_west[0]
            proj_south = proj_south[0]
        else:
            print('Using specified center lonlat =',*self.center_lonlat)
            assert (self.center_lonlat[0] >= -180) & (self.center_lonlat[0] <= 180)
            assert (self.center_lonlat[1] >= -90) & (self.center_lonlat[1] <= 90)
            ori_west, ori_south = transform_coordinates(
                lonlat_crs, self.projected_crs,
                *self.center_lonlat)
            proj_west = ori_west[0] - xsize//2 * self.resolution
            proj_south = ori_south[0] - ysize//2 * self.resolution
        proj_east = proj_west + (xsize - 1) * self.resolution
        proj_north = proj_south + (ysize - 1) * self.resolution
        self.bounds = (proj_west, proj_south, proj_east, proj_north)
        self.extent = get_extent_from_bounds(self.bounds)
        self.lonlat_bounds = transform_bounds(
            self.bounds, self.projected_crs, lonlat_crs)
        # print(self.lonlat_bounds) # (lon_min, lat_min, lon_max, lat_max)

        # Get meshgrids for pcolormesh plotting
        xgrid_terrain, ygrid_terrain = self.get_terrain_grid()
        self.xx_terrain, self.yy_terrain = np.meshgrid(
            xgrid_terrain, ygrid_terrain, indexing='ij')
        xgrid, ygrid = self.get_terrain_grid(self.resolution, self.gridsize)
        self.xx, self.yy = np.meshgrid(xgrid, ygrid, indexing='ij')

        # Download terrain layers from USGS's 3DEP dataset
        self.region = Terrain(self.lonlat_bounds, self.data_dir)
        if self.terrain_data_source == 'auto':
            try:
                if self.slopeAspectMode == 'download':
                    self.terrain_layers = {
                        'Elevation': 'DEM',
                        'Slope': 'Slope Degrees',
                        'Aspect': 'Aspect Degrees'
                    }
                elif self.slopeAspectMode == 'compute':
                    self.terrain_layers = {
                        'Elevation': 'DEM',
                    }
                else:
                    raise ValueError('Mode can only be compute or download')

                self.region.download(self.terrain_layers.values())
            except requests.exceptions.ReadTimeout:
                print(
                    'Timeout issues with 3DEP WMS service. It might be worth stop and try again. Trying SRTM1..')
                self.terrain_layers = {'Elevation': 'SRTM1'}
                self.region.download(self.terrain_layers.values())
            except Exception as _:
                print('Unknown issue with 3DEP WMS service. Trying SRTM1..')
                self.terrain_layers = {'Elevation': 'SRTM1'}
                self.region.download(self.terrain_layers.values())

        else:
            # specified data source
            if self.terrain_data_source == '3DEP':
                if self.slopeAspectMode == 'download':
                    self.terrain_layers = {
                        'Elevation': 'DEM',
                        'Slope': 'Slope Degrees',
                        'Aspect': 'Aspect Degrees'
                    }
                elif self.slopeAspectMode == 'compute':
                    self.terrain_layers = {
                        'Elevation': 'DEM',
                    }
                else:
                    raise ValueError('Mode can only be compute or download')
            else:
                self.terrain_layers = {'Elevation': self.terrain_data_source}

            self.region.download(self.terrain_layers.values())

        # Setup turbine data
        turbine_fpath = os.path.join(self.mode_data_dir, 'turbines.csv')
        self.turbines = TurbinesUSWTB(self.bounds, self.projected_crs,
                                      self.turbine_minimum_hubheight,
                                      turbine_fpath, self.print_verbose)

        # figure out wtk and its layers to extract
        if self.orographic_model.lower() == 'original':
            wtk_height = self.wtk_orographic_height
        else:
            wtk_height = self.href
        self.wtk_layers = {
            'wspeed': f'windspeed_{str(int(wtk_height))}m',
            'wdirn': f'winddirection_{str(int(wtk_height))}m',
            'pressure': f'pressure_{str(int(self.wtk_thermal_height))}m',
            'temperature': f'temperature_{str(int(self.wtk_thermal_height))}m',
            'blheight': 'boundary_layer_height',
            'surfheatflux': 'surface_heat_flux',
        }

        # Mode specific settings; get case ids
        if self.sim_mode.lower() == 'uniform':
            print(f'Uniform mode: Wind speed = {self.uniform_windspeed_h} m/s')
            print(
                f'Uniform mode: Wind dirn = {self.uniform_winddirn_h} deg (cw)')
            self.case_ids = [self._get_uniform_id()]

        else:
            # Get list of times depending on mode
            if self.sim_mode.lower() == 'seasonal':
                self.dtimes = self.get_seasonal_datetimes()
            elif self.sim_mode.lower() == 'snapshot':
                self.dtimes = [datetime(*self.snapshot_datetime)]

            # Initilize the wind data object depending on source
            if self.wind_data_source == 'wtk':
                self.wtk = WTK(self.wtk_source, self.lonlat_bounds,
                               self.wtk_layers.values(), self.mode_data_dir)
            elif self.wind_data_source == 'hrrr':
                pass
                #self.hrrr = HRRR(valid_date=self.dtimes)

            # Download wind data
            if self.wind_data_source == 'wtk':
                self.wtk.download_data(self.dtimes, self.max_cores)
            elif self.wind_data_source == 'hrrr':
                #self.hrrr.download_data(self.dtimes, self.max_cores)
                self.hrrr = [HRRR(valid_date=d) for d in self.dtimes]

            # Assemble list of case IDs
            self.case_ids = [dt.strftime(self.time_format)
                             for dt in self.dtimes]

        # Calculate the orographic updraft based on mode
        if self.movement_model == 'drw':
            self.compute_dummy_updraft_field()
        elif self.sim_mode.lower() == 'uniform':
            self.compute_orographic_updrafts_uniform()
        else:
            if self.wind_data_source == 'wtk':
                self.compute_orographic_updrafts_using_wtk()
            elif self.wind_data_source == 'hrrr':
                self.compute_orographic_updrafts_using_hrrr()

        # Calculate the thermal updraft for each case IDs (diff timestamps)
        for case_id in self.case_ids:
            self.compute_thermal_updrafts(case_id)

        # Consider a distribution of updraft thresholds
        if self.updraft_threshold_stdev > 0:
            # calculate potential fields for updraft fields with a range of
            # discrete threshold values
            threshmin = self.updraft_threshold_realization_min
            threshmax = self.updraft_threshold_realization_max
            threshstep = self.updraft_threshold_realization_step
            self.threshold_realizations = np.arange(
                threshmin, threshmax + threshstep, threshstep)
            self.threshold_realizations = self.updraft_threshold + \
                self.updraft_threshold_stdev * self.threshold_realizations
            print('Discrete thresholds to consider:',
                  self.threshold_realizations)
        else:
            # legacy model
            self.threshold_realizations = None

        # Plotting settings
        fig_aspect = self.region_width_km[0] / self.region_width_km[1]
        self.fig_size = (self.fig_height * fig_aspect, self.fig_height)
        self.km_bar = min([1, 5, 10], key=lambda x: abs(
            x - self.region_width_km[0] // 4))

        print(f'Case id is {self.case_ids}')
        print(f'Reference height is {self.href}')
        print(f'Analysis height is {self.h}')
        print('SSRS Simulator initiation done.\n')

    def checkInputs(self):
        """ Check some inputs and print warning messages """

        if self.thermal_model is None:
            self.thermal_model = 'none'

        if self.thermal_model not in ['none', 'naive', 'improvedAllen']:
            raise ValueError(
                f"Invalid option for thermal model. Options are 'none', 'naive', 'improvedAllen'")

        if self.orographic_model not in ['original', 'improved']:
            raise ValueError(
                f"Invalid option for orographic model. Options are 'original', 'improved'")

        if self.wind_data_source not in ['hrrr', 'wtk']:
            raise ValueError(
                f"Invalid option for wind data source. Options are 'hrrr', 'wtk'")

        if self.thermal_model == 'improvedAllen':
            if self.sim_mode == 'uniform':
                raise ValueError(f"uniform mode is not compatible with improvedAllen thermal model. Pick",
                                 f"either snapshot of seasonal mode and provide the desired timestamp")
            if self.wind_data_source == 'wtk':
                raise ValueError(
                    f"Improved Allen model for thermal updrafts require wind_data_source='hrrr'")

        if self.thermal_model == 'naive':
            if self.sim_mode != 'uniform':
                print(
                    f"WARNING: Naive thermal model is only recommended for 'uniform` mode.")
                print(
                    f"         When date is provided, it is recommended that thermal_model='improvedAllen'")

        if self.thermal_model == 'none':
            try:
                nreal = self.thermals_realization_count
            except AttributeError:
                # No thermal model selected, setting number of realizations to zero
                self.thermals_realization_count = 0

            if self.thermals_realization_count != 0:
                print(f"WARNING: {self.thermals_realization_count} thermal realization requested, but",
                      f"no model specified. No realizations being computed. Valid options for `thermal_model`",
                      f"are 'naive', and 'improvedAllen'")

        if self.uniform_windspeed_h != self.uniform_windspeed:
            print(
                f"WARNING: Make sure `uniform_windspeed_h` is the same as `uniform_windspeed`")
        if self.uniform_winddirn_h != self.uniform_winddirn:
            print(
                f"WARNING: Make sure `uniform_winddirn_h` is the same as `uniform_winddirn`")

        if self.uniform_windspeed_href != self.uniform_windspeed:
            print(f"WARNING: Check if `uniform_windspeed_href` is not supposed to be the same as `uniform_windspeed`")
        if self.uniform_winddirn_href != self.uniform_winddirn:
            print(f"WARNING: Check if `uniform_winddirn_href` is not supposed to be the same as `uniform_winddirn`")

        if self.thermal_intensity_scale != 2 and self.thermal_model != 'naive':
            print(f"WARNING: thermal intensity appears to be set. This option is only valid with naive thermal model.")

        if self.resolution < 30 and self.orographic_model == 'improved':
            print(
                f"WARNING: Analysis resolution might be too high for Sx calculation. Consider reducing it.")

    def _get_starting_indices(self) -> List[int]:
        """ get starting indices of eagle tracks """
        # validate inputs
        extent = self.region_width_km  # extent of full simulation domain [km]
        if self.track_start_region_width > 0:
            # specify starting region based on width, origin, and rotation
            # parameters; track_start_region will be overwritten for plotting
            # purposes
            if not self.track_start_region_origin_xy:
                # convert from lon/lat
                track_start_x, track_start_y = transform_coordinates(
                    lonlat_crs, self.projected_crs, *self.track_start_region_origin)
                self.track_start_region_origin = (
                    (track_start_x[0] - self.bounds[0]) / 1000.,
                    (track_start_y[0] - self.bounds[1]) / 1000.,
                )
                self.track_start_region_origin_xy = True
            sbounds = (self.track_start_region_origin[0] - self.track_start_region_width / 2,
                       self.track_start_region_origin[0] +
                       self.track_start_region_width / 2,
                       self.track_start_region_origin[1] -
                       self.track_start_region_depth / 2,
                       self.track_start_region_origin[1] + self.track_start_region_depth / 2)
            self.track_start_region = sbounds
            print('Note: track_start_region_width specified; '
                  f'track_start_region overwritten with {str(self.track_start_region)}')
        else:
            sbounds = self.track_start_region  # [xmin, xmax, ymin, ymax]
            if ((sbounds[1] < sbounds[0]) or
                (sbounds[3] < sbounds[2]) or
                (sbounds[0] < 0.) or
                (sbounds[2] < 0.) or
                (sbounds[1] > extent[0]) or
                    (sbounds[3] > extent[1])):
                raise ValueError('track_start_region is invalid!')

        res_km = self.resolution / 1000.
        if self.track_start_region_rotation == 0:
            # original index selection code
            xind_max = np.ceil(extent[0] / res_km)
            yind_max = np.ceil(extent[1] / res_km)
            xind_low = min(
                max(np.floor(sbounds[0] / res_km) - 1, 1), xind_max - 2)
            xind_upp = max(min(np.ceil(sbounds[1] / res_km), xind_max - 1), 2)
            yind_low = min(
                max(np.floor(sbounds[2] / res_km) - 1, 1), yind_max - 2)
            yind_upp = max(min(np.ceil(sbounds[3] / res_km), yind_max - 1), 2)
            print(xind_max, yind_max, xind_low, xind_upp, yind_low, yind_upp)
            xmesh, ymesh = np.mgrid[xind_low:xind_upp, yind_low:yind_upp]
            base_inds = np.vstack((np.ravel(ymesh), np.ravel(xmesh)))
        else:
            print(
                f'Rotating start region by {self.track_start_region_rotation:g} deg')
            ang = np.radians(self.track_start_region_rotation)
            x1 = np.arange(0, extent[0] + res_km, res_km) - \
                self.track_start_region_origin[0]
            y1 = np.arange(0, extent[1] + res_km, res_km) - \
                self.track_start_region_origin[1]
            xx, yy = np.meshgrid(x1, y1)
            xx_r = xx * np.cos(ang) - yy * np.sin(ang)
            yy_r = xx * np.sin(ang) + yy * np.cos(ang)
            base_inds = np.where((xx_r > -self.track_start_region_width / 2) &
                                 (xx_r < self.track_start_region_width / 2) &
                                 (yy_r > -self.track_start_region_depth / 2) &
                                 (yy_r < self.track_start_region_depth / 2))
            base_inds = np.vstack(base_inds)
        base_count = base_inds.shape[1]

        if self.track_start_type == 'structured':
            idx = np.round(np.linspace(0, base_count - 1,
                           self.track_count % base_count))
            if self.track_count > base_count:
                start_inds = np.tile(
                    base_inds, (1, self.track_count // base_count))
                start_inds = np.hstack(
                    (start_inds, start_inds[:, idx.astype(int)]))
            else:
                start_inds = base_inds[:, idx.astype(int)]
        elif self.track_start_type == 'random':
            idx = np.random.randint(0, base_count, self.track_count)
            start_inds = base_inds[:, idx]
        else:
            raise ValueError((f'Model:Invalid sim_start_type of {self.track_start_type}\n'
                              'Options: structured, random'))
        start_inds = start_inds.astype(int)
        return start_inds[0, :], start_inds[1, :]


########## terrain related functions ############


    def get_terrain_elevation(self):
        """ Returns data for terrain layer inprojected crs """
        elev = self.get_terrain_layer('Elevation')
        fname = self._get_terrain_quantity_fname(
            self.case_ids[0], 'elev', self.mode_data_dir)
        if not os.path.isfile(f'{fname}.npy'):
            np.save(f'{fname}.npy', elev.astype(np.float32))
        return elev

    def get_terrain_slope(self):
        """ Returns data for terrain layer inprojected crs """
        if self.slopeAspectMode == 'download':
            try:
                slope = self.get_terrain_layer('Slope')
            except OSError:
                print(
                    '---- This should NOT be printed. This is downloaded, it should always find the layer')
            return slope

        elif self.slopeAspectMode == 'compute':
            try:
                slopefname_str = 'slope'
                if self.orographic_model.lower() != 'original':
                    slopefname_str += f'_blur{int(self.h)}m'
                slope = self.load_terrain_quantity(
                    self.case_ids[0], slopefname_str)
                print(f'Found slope map. Loading it..')
            except OSError:
                slope = self.compute_slope_degrees_case()
            return slope
        else:
            raise ValueError('Mode can only be compute or download')

    def get_terrain_aspect(self):
        """ Returns data for terrain layer inprojected crs """
        if self.slopeAspectMode == 'download':
            try:
                aspect = self.get_terrain_layer('Aspect')
            except OSError:
                print(
                    '--- This should NOT be printed. This is downloaded, it should always find the layer')
            return aspect

        elif self.slopeAspectMode == 'compute':
            try:
                aspectfname_str = 'aspect'
                if self.orographic_model.lower() != 'original':
                    aspectfname_str += f'_blur{int(self.h)}m'
                aspect = self.load_terrain_quantity(
                    self.case_ids[0], aspectfname_str)
                print(f'Found aspect map. Loading it..')
            except OSError:
                aspect = self.compute_aspect_degrees_case()
            return aspect
        else:
            raise ValueError('Mode can only compute or download')

    def get_terrain_sx(self):
        """ Returns low-res data for terrain Sx layer inprojected crs """
        try:

            # wind direction with proper convention for Sx calculation
            wdirn_sx_ = (self.Ang + 180) % 360
            # Approximate to closest 5 deg
            wdirn_sx = 5 * round(wdirn_sx_ / 5)

            sxfname_str = f'sx_d{int(wdirn_sx)}_lowres'
            sx = self.load_terrain_quantity(self.case_ids[0], sxfname_str)

            print(
                f'Found sx map for {int(wdirn_sx)} deg (angle flipped). Loading it..')
        except OSError:
            print(f'Could not find {sxfname_str}.')
            sx = self.compute_sx_case()
        return sx

    def get_terrain_layer(self, lname: str):
        """ Returns data for terrain layer inprojected crs """
        ldata = get_raster_in_projected_crs(
            self.region.get_raster_fpath(self.terrain_layers[lname]),
            self.bounds, self.gridsize_terrain, self.resolution_terrain, self.projected_crs)
        return ldata

    def get_terrain_grid(self, resolution=None, gridsize=None):
        """
        Returns xgrid and ygrid for the terrain depending on resolution
        """

        if resolution is None:
            resolution = self.resolution_terrain
        if gridsize is None:
            gridsize = self.gridsize_terrain
        xgrid = np.linspace(self.bounds[0],
                            self.bounds[0] + (gridsize[1] - 1) *
                            resolution, gridsize[1])
        ygrid = np.linspace(self.bounds[1],
                            self.bounds[1] + (gridsize[0] - 1) *
                            resolution, gridsize[0])
        return xgrid, ygrid


########## Computing Updrafts ##########

    def compute_sx_case(self, lowres=True) -> None:
        """
        Computes and saves Sx quantity in low-res

        The computation of the sheltering angle needs to be in low
        resolution because of speed.
        """
        # Get sx based on low-resolution terrain data
        xgrid, ygrid = self.get_terrain_grid(self.resolution, self.gridsize)
        elev_lowres = highRes2lowRes(
            self.get_terrain_elevation(), self.resolution_terrain, self.resolution)

        # wind direction with proper convention for Sx calculation
        wdirn_sx_ = (self.Ang + 180) % 360
        # Approximate to closest 5 deg
        wdirn_sx = 5 * round(wdirn_sx_ / 5)

        print(
            f'Getting Sx for {wdirn_sx:.2f} degrees (angle flipped), based on {self.resolution} m resolution maps')
        sxfname_str = f'sx_d{int(wdirn_sx)}_lowres'

        sx = calcSx(xgrid, ygrid, elev_lowres.T, wdirn_sx)
        sx = sx.T  # adjust convention

        fname = self._get_terrain_quantity_fname(
            self.case_ids[0], sxfname_str, self.mode_data_dir)
        np.save(f'{fname}.npy', sx.astype(np.float32))
        return sx

    def compute_slope_degrees_case(self) -> None:
        """ Computes and saves slope """
        elev = self.get_terrain_elevation()
        if self.orographic_model.lower() == 'original':
            print('Calculating slope based elevation map..')
            slopefname_str = 'slope'
        else:
            print('Calculating blurred version of slope based on blurred elevation map..')
            elev = blurQuantity(elev, self.resolution_terrain, self.h)
            slopefname_str = f'slope_blur{int(self.h)}m'

        # actually calculate it
        slope = calcSlopeDegrees(elev, self.resolution_terrain)
        fname = self._get_terrain_quantity_fname(
            self.case_ids[0], slopefname_str, self.mode_data_dir)
        np.save(f'{fname}.npy', slope.astype(np.float32))
        return slope

    def compute_aspect_degrees_case(self) -> None:
        """ Computes and saves aspect """
        elev = self.get_terrain_elevation()
        if self.orographic_model.lower() == 'original':
            print('Calculating aspect based elevation map..')
            aspectfname_str = 'aspect'
        else:
            print(
                'Calculating blurred version of aspect based on blurred elevation map..')
            elev = blurQuantity(elev, self.resolution_terrain, self.h)
            aspectfname_str = f'aspect_blur{int(self.h)}m'

        aspect = calcAspectDegrees(elev, self.resolution_terrain)
        fname = self._get_terrain_quantity_fname(
            self.case_ids[0], aspectfname_str, self.mode_data_dir)
        np.save(f'{fname}.npy', aspect.astype(np.float32))
        return aspect

    def compute_orographic_updrafts_uniform(self) -> None:
        """
        Computing orographic updrafts for uniform mode.
        Computed orographic updrafts are in low resolution (analysis resolution)
        """

        print(
            f'Computing orographic updrafts (uniform mode) using {self.orographic_model} model..')
        slope = self.get_terrain_slope()
        aspect = self.get_terrain_aspect()
        elev = self.get_terrain_elevation()

        if self.orographic_model.lower() == 'original':
            wspeed = self.uniform_windspeed_h * np.ones(self.gridsize_terrain)
            wdirn = self.uniform_winddirn_h * np.ones(self.gridsize_terrain)
            h = self.h
            sx = None
        else:
            wspeed = self.uniform_windspeed_href * \
                np.ones(self.gridsize_terrain)
            wdirn = self.uniform_winddirn_href * np.ones(self.gridsize_terrain)
            h = self.href
            self.Ang = np.mean(wdirn)
            sx = self.get_terrain_sx()  # low-res

        updraft = calcOrographicUpdraft(elev, wspeed, wdirn, slope, aspect,
                                        self.resolution_terrain, self.resolution, sx, h)

        fname = self._get_orographicupdraft_fname(
            self.case_ids[0], self.mode_data_dir)
        np.save(f'{fname}.npy', updraft.astype(np.float32))

    def compute_orographic_updrafts_using_hrrr(self) -> None:
        """ 
        Computing orographic updrafts using HRRR data for all datetimes.
        Computed orographic updrafts are in low resolution (analysis resolution)
        """

        slope = self.get_terrain_slope()
        aspect = self.get_terrain_aspect()
        elev = self.get_terrain_elevation()

        start_time = time.time()
        for t, (dtime, case_id) in enumerate(zip(self.dtimes, self.case_ids)):
            #wtk_df = self.wtk.get_dataframe_for_this_time(dtime)
            # wspeed, wdirn = self._get_interpolated_wind_conditions(
            #    wtk_df[self.wtk_layers['wspeed']],
            #    wtk_df[self.wtk_layers['wdirn']]
            # )
            print(
                f'Computing orographic updrafts (using HRRR data at {dtime} UTC) using {self.orographic_model} model..')
            # The function below only given a scalar value. This is used for the orographic updraft model,
            # so it is fine. Needs to get the actual grid if doing larger extents.
            hrrr_obj = self.hrrr[t]
            conditions = hrrr_obj.wind_velocity_direction_at_altitude(
                [self.lonlat_bounds[0] + (self.lonlat_bounds[2] - self.lonlat_bounds[0]) / 2,
                 self.lonlat_bounds[1] + (self.lonlat_bounds[3] - self.lonlat_bounds[1]) / 2],
                height_above_ground_m=self.h,
                extent_km_lon=self.region_width_km[0],
                extent_km_lat=self.region_width_km[1]
            )
            wspeed_hrrr = conditions['speed']
            wdirn_hrrr = conditions['direction_deg']
            wspeed = wspeed_hrrr * np.ones(self.gridsize_terrain)
            wdirn = wdirn_hrrr * np.ones(self.gridsize_terrain)

            if self.orographic_model.lower() == 'original':
                sx = None
                h = self.h
            else:
                h = self.href
                self.Ang = wdirn_hrrr
                sx = self.get_terrain_sx()

            updraft = calcOrographicUpdraft(elev, wspeed, wdirn, slope, aspect,
                                            self.resolution_terrain, self.resolution, sx, h)

            fname = self._get_orographicupdraft_fname(
                case_id, self.mode_data_dir)
            np.save(f'{fname}.npy', updraft.astype(np.float32))
        print(
            f'took {get_elapsed_time(start_time)} to get orographic updrafts using HRRR', flush=True)

    def compute_orographic_updrafts_using_wtk(self) -> None:
        """ 
        Computing orographic updrafts using wtk data for all datetimes.
        Computed orographic updrafts are in low resolution (analysis resolution)
        """

        print(
            f'Computing orographic updrafts (using WTK) using {self.orographic_model} model..')

        slope = self.get_terrain_slope()
        aspect = self.get_terrain_aspect()
        elev = self.get_terrain_elevation()

        if self.wtk_source == 'HRRR':
            wspdlayer = 'windspeed_80m'
            wdirlayer = 'winddirection_80m'
        else:
            wspdlayer = self.wtk_layers['wspeed']
            wdirlayer = self.wtk_layers['wdirn']

        start_time = time.time()
        for dtime, case_id in zip(self.dtimes, self.case_ids):
            wtk_df = self.wtk.get_dataframe_for_this_time(dtime)
            wspeed, wdirn = self._get_interpolated_wind_conditions(
                wtk_df[wspdlayer],
                wtk_df[wdirlayer]
            )
            if self.orographic_model.lower() == 'original':
                sx = None
                h = self.h
            else:
                h = self.href
                self.Ang = np.mean(wdirn)
                sx = self.get_terrain_sx()

            updraft = calcOrographicUpdraft(elev, wspeed, wdirn, slope, aspect,
                                            self.resolution_terrain, self.resolution, sx, h)

            fname = self._get_orographicupdraft_fname(
                case_id, self.mode_data_dir)
            np.save(f'{fname}.npy', updraft.astype(np.float32))
        print(
            f'took {get_elapsed_time(start_time)} to get orographic updrafts using WTK', flush=True)

    def compute_dummy_updraft_field(self) -> None:
        """This should be deprecated in the future!

        For DRW, updraft field need not be calculated. However, a number
        of routines (e.g., plot_*()) will load the updraft field from
        file just to get the domain size. This should be fixed in a
        future code refactor.
        """
        updraft = np.zeros(self.gridsize)
        for dtime, case_id in zip(self.dtimes, self.case_ids):
            fname = self._get_orographicupdraft_fname(
                case_id, self.mode_data_dir)
            np.save(f'{fname}.npy', updraft.astype(np.float32))

    def compute_thermal_updrafts(self, case_id: str):
        """ Computes updrafts for the particular case """
        if self.thermals_realization_count > 0:
            print(f'Computing {self.thermals_realization_count} realizations of',
                  f'thermal updrafts using {self.thermal_model} model', flush=True)
            aspectHighRes = self.get_terrain_aspect()

            # Thermal updrafts are computed using aspect (as of right now). Coarsening
            # the aspect field so that we can later sum both orog and thermal updraft
            aspect = highRes2lowRes(
                aspectHighRes, self.resolution_terrain, self.resolution)

            for real_id in range(self.thermals_realization_count):
                if self.thermal_model == 'naive':
                    thermals = compute_random_thermals(
                        aspect, self.thermal_intensity_scale)
                    fname = self._get_thermal_updraft_fname(
                        case_id, real_id, self.mode_data_dir)
                    np.save(f'{fname}.npy', thermals.astype(np.float32))

                elif self.thermal_model == 'improvedAllen':
                    # Computing using low-res quantities
                    # We have a list of datetime (possibly of size 1), let's iterate through them
                    for time in self.dtimes:
                        print(f'Calculating thermals for {time}')
                        # xmin, ymin, xmax, ymax
                        extent_for_thermals_3d = [
                            0, 0, self.region_width_km[0] * 1000, self.region_width_km[1] * 1000]
                        thermals = compute_thermals_3d(aspect, self.southwest_lonlat, extent_for_thermals_3d, self.resolution,
                                                       time, self.h, wfipInformed=False)

                        thermals = thermals.T
                        fname = self._get_thermal_updraft_fname(
                            case_id, real_id, self.mode_data_dir)
                        np.save(f'{fname}.npy', thermals.astype(np.float32))

        else:
            print('No thermals requested')

    def load_updrafts(self, case_id: str, apply_threshold=True):
        """ Computes updrafts for the particular case """
        fname = self._get_orographicupdraft_fname(case_id, self.mode_data_dir)
        orographicupdraft = np.load(f'{fname}.npy')
        print(
            f'Found orographic updraft {os.path.basename(fname)}. Loading it...')

        # note: updrafts should be a list of updraft realizations
        # by default (w/o any thermals), we have only a single updraft field
        updrafts = [orographicupdraft]

        # Removing this for heuristics. This function needs to return only the orographic
        # Add thermal updrafts to the `updrafts` field if available
        if self.sim_movement != 'heuristics':
            if self.thermals_realization_count > 0:
                assert self.threshold_realizations is None, \
                    'using a random threshold with thermals has not been tested'
                for real_id in range(self.thermals_realization_count):
                    fname = self._get_thermal_updraft_fname(
                        case_id, real_id, self.mode_data_dir)
                    updrafts.append(updrafts + np.load(f'{fname}.npy'))
            if apply_threshold:
                if apply_threshold == True:
                    # use user-specified threshold in input Config
                    threshold = self.updraft_threshold
                else:
                    # apply an arbitrary threshold, when potential fields corresponding
                    # to a range of threshold values is needed, to allow more efficient
                    # estimation of individual potential fields when each simulated
                    # individual has a random threshold value (with hard cutoff)
                    assert isinstance(apply_threshold, float)
                    threshold = apply_threshold
                if self.smooth_threshold_cutoff:
                    # legacy model with threshold function
                    assert self.threshold_realizations is None, \
                        'random threshold with threshold function not supported'
                    updrafts = [get_above_threshold_speed(w0, threshold)
                                for w0 in updrafts]
                else:
                    updrafts = [get_above_threshold_hard_cutoff(w0, threshold)
                                for w0 in updrafts]

        return updrafts  # a list of updraft fields

    def load_terrain_quantity(self, case_id: str, quant: str):
        """ Load specific quantity for the particular case """
        fname = self._get_terrain_quantity_fname(
            case_id, quant, self.mode_data_dir)
        metric = np.load(f'{fname}.npy')
        return metric

    def _get_terrain_quantity_fname(self, case_id: str, quant: str, dirname: str = './'):
        """ Returns file path for saving terrain quantity data """
        return os.path.join(dirname, f'{case_id}_{quant}_{self.orographic_model}Model')

    def _get_orographicupdraft_fname(self, case_id: str, dirname: str = './'):
        """ Returns file path for saving orographic updrafts data """
        if self.orographic_model.lower() == 'original':
            return os.path.join(dirname, f'{case_id}_orographicupdraft_{self.orographic_model}Model')
        elif self.orographic_model.lower() == 'improved':
            return os.path.join(dirname, f'{case_id}_orographicupdraft_{self.orographic_model}Model_{int(self.h)}m')

    def _get_thermal_updraft_fname(self, case_id: str, real_id: int, dirname: str = './'):
        """ Returns file path for saving thermal updrafts data for each realization real_id"""
        if self.thermal_model.lower() == 'naive':
            return os.path.join(dirname, f'{case_id}_r{real_id}_thermalupdraft_{self.thermal_model}Model')
        elif self.thermal_model.lower() == 'improvedAllen':
            return os.path.join(dirname, f'{case_id}_r{real_id}_thermalupdraft_{self.thermal_model}Model_{int(self.h)}m')

    def plot_simulation_output(self, plot_turbs=True, show=False) -> None:
        """ Plots oro updraft and tracks """
        self.plot_orographic_updrafts(plot_turbs, show)
        self.plot_sm_orographic_updrafts(plot_turbs, show)
        self.plot_thermal_updrafts(plot_turbs, show)
        self.plot_updrafts(plot_turbs, show)
        #self.plot_directional_potentials(plot_turbs, show)
        self.plot_simulated_tracks_dem(plot_turbs, show)
        self.plot_simulated_tracks_w(plot_turbs, show)
        if self.sim_movement == 'fluid-analogy':
            self.plot_presence_map(plot_turbs, show)
        elif self.sim_movement == 'heuristics':  # heuristics
            self.plot_presence_map_HSSRS(plot_turbs, show)


######## Compute and plot directional potential ##########


    def get_directional_potential(self, updraft, case_id, real_id):
        """ Computes the migration potential by solving sparse linear system"""
        mov_model = MovModel(self.track_direction, self.gridsize)
        bndry_nodes, bndry_energy = mov_model.get_boundary_nodes()
        row_inds, col_inds, facs = mov_model.assemble_sparse_linear_system()
        fname = self._get_potential_fname(case_id, real_id, self.mode_data_dir)
        id_str = self._get_id_string(case_id, real_id)
        try:
            potential = np.load(f'{fname}.npy')
            if potential.shape != self.gridsize:
                raise FileNotFoundError
            if (self.sim_seed < 0) & (real_id != 0):
                raise FileNotFoundError
            print(f'{id_str}: Found saved potential')
        except FileNotFoundError as _:
            start_time = time.time()
            print(f'{id_str}: Computing potential..', end="", flush=True)
            potential = mov_model.solve_sparse_linear_system(
                updraft,
                bndry_nodes,
                bndry_energy,
                row_inds,
                col_inds,
                facs
            )
            print(f'took {get_elapsed_time(start_time)}', flush=True)
            np.save(f'{fname}.npy', potential.astype(np.float32))
        if np.isnan(potential).any():
            print('NANs found in potential!')
        return potential

    def get_directional_potential_realizations(self, updraft, case_id, real_id):
        """Different potential fields are possible from different
        updraft threshold values

        TODO: parallelize this
        """
        assert real_id == 0, 'should have only one orographic updraft field'
        potential_realizations = []
        for real_id, threshold in enumerate(self.threshold_realizations):
            print('Calculating potential for updraft threshold =', threshold)
            updraft_with_threshold = get_above_threshold_hard_cutoff(updraft,
                                                                     threshold)
            potential = self.get_directional_potential(
                updraft_with_threshold, case_id, real_id)
            potential_realizations.append(potential)
        return potential_realizations

    def _get_id_string(self, case_id: str, real_id: Optional[int] = None):
        """ Commong id string for saving/reading data and screen output"""
        dirn_str = f'd{int(self.track_direction % 360)}'
        threshold_str = f't{int(self.updraft_threshold*100)}'
        mov_str = f'{self.movement_model}'
        out_str = f'{case_id}_{dirn_str}_{threshold_str}_{mov_str}'
        if real_id is not None:
            out_str += f'_r{int(real_id)}'
        if self.orographic_model == 'improved':
            out_str += f'_h{self.h}'
        return out_str

    def _get_potential_fname(self, case_id: str, real_id: int, dirname: str):
        """ Returns file path for saving directional potential data"""
        fname = f'{self._get_id_string(case_id, real_id)}_potential'
        return os.path.join(dirname, fname)

    def plot_directional_potentials(self, plot_turbs=True, show=False) -> None:
        """ Plot directional potential """
        if self.movement_model == 'fluid-flow':
            print('Plotting directional potential..')
            for case_id in self.case_ids:
                updrafts = self.load_updrafts(case_id, apply_threshold=True)
                if self.threshold_realizations is None:
                    iterator = enumerate(updrafts)
                else:
                    assert len(updrafts) == 1
                    iterator = enumerate(self.threshold_realizations)
                for real_id, tmp in iterator:
                    fname = self._get_potential_fname(case_id, real_id,
                                                      self.mode_data_dir)
                    potential = np.load(f'{fname}.npy')
                    fig, axs = plt.subplots(figsize=self.fig_size)
                    lvls = np.linspace(0., np.amax(potential), 11)
                    curm = axs.contourf(potential, lvls, cmap='cividis',
                                        origin='lower',
                                        extent=self.extent)
                    cbar, _ = create_gis_axis(fig, axs, curm, self.km_bar)
                    cbar.set_label('Directional potential')
                    if plot_turbs:
                        self.plot_turbine_locations(axs)
                    if self.threshold_realizations is not None:
                        axs.set_title(f'threshold = {tmp:g} m/s')
                    axs.set_xlim([self.extent[0], self.extent[1]])
                    axs.set_ylim([self.extent[2], self.extent[3]])
                    fname = self._get_potential_fname(case_id, real_id,
                                                      self.mode_fig_dir)
                    self.save_fig(fig, f'{fname}.png', show)

########### Simulate and plot tracks ###########

    def simulate_tracks(self):
        """ Simulate tracks """
        print(f'Movement model = {self.movement_model}')
        print(f'Updraft threshold = {self.updraft_threshold} m/s')
        print(f'Movement direction = {self.track_direction} deg (cw)')
        # print(f'Memory parameter = {self.track_dirn_restrict}')
        # print('Getting starting locations for simulating eagle tracks..')
        starting_rows, starting_cols = self._get_starting_indices()
        if self.threshold_realizations is not None:
            # "best practice" RNG -- this is self contained and _should not_
            # affect other calls to np.random.* when using the reseeded
            # legacy BitGenerator (np.random.seed), i.e., when Config.sim_seed
            # is specified
            from numpy.random import RandomState, MT19937
            rng = RandomState(MT19937(self.updraft_threshold_seed))
            cutoffs = rng.normal(loc=self.updraft_threshold,
                                 scale=self.updraft_threshold_stdev,
                                 size=len(starting_rows))
            cutoffs[np.where(cutoffs < self.threshold_realizations[0])
                    ] = self.threshold_realizations[0]
            cutoffs[np.where(cutoffs > self.threshold_realizations[-1])
                    ] = self.threshold_realizations[-1]
            initial_conditions = [[x, y, w]
                                  for x, y, w in zip(starting_rows, starting_cols, cutoffs)]
        else:
            # smooth function or hard cutoff, with constant threshold value
            initial_conditions = [[x, y]
                                  for x, y in zip(starting_rows, starting_cols)]

        num_cores = min(self.track_count, self.max_cores)
        for case_id in self.case_ids:
            if self.threshold_realizations is None:
                updrafts = self.load_updrafts(case_id, apply_threshold=True)
            else:
                # range of threshold values to be evaluated later
                updrafts = self.load_updrafts(case_id, apply_threshold=False)
            # loop over updraft realizations (default is to have only 1)
            for real_id, updraft in enumerate(updrafts):
                id_str = self._get_id_string(case_id, real_id)

                if self.movement_model == 'fluid-flow':
                    if self.threshold_realizations is None:
                        potential = self.get_directional_potential(
                            updraft, case_id, real_id)
                    else:
                        # create list of potential fields corresponding
                        #   to different cutoff thresholds; call to
                        #   generate_simulated_tracks() below will
                        #   interpolate between potential fields based
                        #   on initial conditions that include x0,y0 as
                        #   well as a randomly chosen cutoff value
                        potential = self.get_directional_potential_realizations(
                            updraft, case_id, real_id)

                    start_time = time.time()
                    if self.track_converge_tol == 0:
                        print(f'{id_str}: Simulating {self.track_count} tracks..',
                              end="", flush=True)
                        tracks = self._parallel_run(generate_simulated_tracks,
                                                    initial_conditions,
                                                    self.track_direction,
                                                    self.track_dirn_restrict,
                                                    self.track_stochastic_nu,
                                                    updraft,
                                                    potential,  # single field -OR- list of fields
                                                    threshold_realizations=self.threshold_realizations
                                                    )

                    else:
                        assert self.track_converge_tol > 0
                        if self.threshold_realizations is not None:
                            raise NotImplementedError(
                                'Track convergence mode not set up for random updraft thresholds')
                        if self.track_start_type != 'random':
                            print(
                                'WARNING: In track convergence mode, track_start_type should be random')
                        print(f'{id_str}: Simulating up to {self.track_count} tracks'
                              f' (tol={self.track_converge_tol:g})...\n',
                              end="", flush=True)
                        tracks = []
                        last_presence_map = None
                        for istart in range(0,
                                            self.track_count,
                                            self.track_converge_check_interval):
                            start_loc_range = slice(
                                istart, istart + self.track_converge_check_interval)
                            # append to list of all tracks simulated thus far
                            tracks += self._parallel_run(generate_simulated_tracks,
                                                         initial_conditions[start_loc_range],
                                                         self.track_direction,
                                                         self.track_dirn_restrict,
                                                         self.track_stochastic_nu,
                                                         updraft,
                                                         potential,
                                                         )
                            Ntracks = len(tracks)
                            # calc new presence map
                            prprob = self.calc_presence_map(
                                tracks, radius=1000.)
                            if self.track_converge_check_plot:
                                fig, ax = self._plot_presence(
                                    prprob, 0.1, False)
                                fname = self._get_presence_fname(case_id, real_id,
                                                                 self.mode_fig_dir)
                                ax.set_title(f'{Ntracks:d} tracks')
                                self.save_fig(
                                    fig, f'{fname}_ntracks{Ntracks:05d}.png')
                            # compare change in presence maps
                            if istart > 0:
                                assert last_presence_map is not None
                                mse = calc_MSE(prprob, last_presence_map)
                                print(
                                    f'  MSE after simulating {Ntracks} tracks: {mse:g}')
                                if mse < self.track_converge_tol:
                                    break
                            last_presence_map = prprob

                elif self.movement_model == 'drw':
                    start_time = time.time()
                    print(f'{id_str}: Simulating {self.track_count} tracks..',
                          end="", flush=True)
                    tracks = self._parallel_run(generate_simulated_tracks,
                                                initial_conditions,
                                                self.track_direction,
                                                self.track_dirn_restrict,
                                                self.track_stochastic_nu,
                                                domain_shape=updraft.shape
                                                )

                print(f'took {get_elapsed_time(start_time)}', flush=True)
                fname = self._get_tracks_fname(
                    case_id, real_id, self.mode_data_dir)
                np.save(f'{fname}_initcond.npy', initial_conditions)
                with open(f'{fname}.pkl', "wb") as fobj:
                    pickle.dump(tracks, fobj)

    def simulate_tracks_HSSRS(self, PAM_stdev=0.0, **hssrs_kwargs):
        """Simulate tracks

        Parameters
        ----------
        PAM_stdev: float
            stdev of normally distributed PAM offsets [deg]
        **hssrs_kwargs:
            Optional keyword argument(s) passed to
            `generate_heuristic_eagle_track`, applicable if
            sim_movement=='heuristics'
        """
        print(f'Movement model = {self.movement_model}')
        print(f'Updraft threshold = {self.updraft_threshold} m/s')
        print(f'Movement direction = {self.track_direction} deg (cw)')

        if self.sim_movement == 'fluid-analogy':
            self.compute_directional_potential()
        elif self.sim_movement == 'heuristics':  # heuristics
            if self.movement_ruleset not in rulesets.keys():
                raise ValueError(
                    f'{self.movement_ruleset} is not defined.  Valid rulesets: {rulesets.keys()}')
            else:
                print('Ruleset:')
                for i, action in enumerate(rulesets[self.movement_ruleset]):
                    print(f'{i+1}.', action)

        starting_rows, starting_cols = self._get_starting_indices()
        starting_locs = [[x, y] for x, y in zip(starting_rows, starting_cols)]

        if PAM_stdev == 0:
            #dir_offsets = np.zeros(len(starting_locs))
            dir_offsets = 0
        else:
            # dir_offsets = np.random.normal(scale=PAM_stdev,
            #                               size=len(starting_locs))
            dir_offsets = np.random.normal(scale=PAM_stdev)

        self.PAM = self.track_direction  # + dir_offsets

        #starting_locs_PAM = list(zip(starting_rows, starting_cols, self.PAM))
        starting_locs_PAM = list(zip(starting_rows, starting_cols))

        num_cores = min(self.track_count, self.max_cores)

        for case_id in self.case_ids:
            #            tmp_str = f'{case_id}_{int(self.track_direction)}'
            #            print(f'{tmp_str}: Simulating {self.track_count} tracks..',
            #                  end="", flush=True)
            # Load orographic updraft only
            orographicupdraft = self.load_updrafts(
                case_id, apply_threshold=True)
            assert len(orographicupdraft) == 1, \
                'Should only have a single orographic updraft field'
            
            elevation = highRes2lowRes(
                self.get_terrain_elevation(), self.resolution_terrain, self.resolution)

            for real_id, updraft in enumerate(updrafts):
                id_str = self._get_id_string(case_id, real_id)

                start_time = time.time()

                if self.sim_movement == 'fluid-analogy':
                    potential = self.get_directional_potential(
                        orographicupdraft, case_id, real_id)
                    print(f'{id_str}: Simulating {self.track_count} tracks..',
                          end="", flush=True)

                    tracks = self._parallel_run(generate_simulated_tracks,
                                                starting_locs,
                                                self.track_direction,
                                                self.track_dirn_restrict,
                                                self.track_stochastic_nu,
                                                orographicupdraft,
                                                potential
                                                )

                elif self.movement_model == 'drw':
                    print(f'{id_str}: Simulating {self.track_count} tracks..',
                          end="", flush=True)

                    tracks = self._parallel_run(generate_simulated_tracks,
                                                starting_locs,
                                                self.track_direction,
                                                self.track_dirn_restrict,
                                                self.track_stochastic_nu,
                                                domain_shape=orographicupdraft.shape
                                                )

                elif self.sim_movement == 'heuristics':

                    fname_thermal = self._get_thermal_updraft_fname(
                        case_id, real_id, self.mode_data_dir)
                    try:
                        thermalupdraft = np.load(f'{fname_thermal}.npy')
                    except FileNotFoundError:
                        # no thermals in this case. Let's set a zero array for that
                        thermalupdraft = np.zeros_like(orographicupdraft)

                    print(f'{id_str}: Simulating {self.track_count} tracks..',
                          end="", flush=True)

                    tracks = self._parallel_run(generate_heuristic_eagle_track,
                                                starting_locs_PAM,  # start_loc
                                                self.PAM,
                                                self.movement_ruleset,
                                                orographicupdraft,
                                                thermalupdraft,
                                                elevation,  # db added
                                                self.resolution,
                                                self.uniform_windspeed_h,  # TODO needs to be generalized to wind from WTK
                                                self.uniform_winddirn_h,  # TODO needs to be generalized to wind from WTK
                                                self.updraft_threshold,
                                                self.look_ahead_dist,
                                                **hssrs_kwargs
                                                )

                print(f'took {get_elapsed_time(start_time)}', flush=True)

                fname = self._get_tracks_fname(
                    case_id, real_id, self.mode_data_dir)
                with open(f'{fname}.pkl', "wb") as fobj:
                    pickle.dump(tracks, fobj)

    def _parallel_run(self, func, initial_conditions, *args, **kwargs):
        num_cores = min(self.track_count, self.max_cores)
# use with import pathos.multiprocessing as mp
        #print('Using map with num_cores=',num_cores)
        with mp.Pool(num_cores) as pool:
            output = pool.map(lambda ic: func(ic, *args, **kwargs),
                              initial_conditions)
# use with import multiprocessing as mp
#        #print('Using starmap with num_cores=',num_cores)
#        arglist = [repeat(arg) for arg in args]
#        with mp.Pool(num_cores, initializer=self.init_worker) as pool:
#            output = pool.starmap(func, zip(start_locs,*arglist))
        return output

    def _get_tracks_fname(self, case_id: str, real_id: int, dirname: str):
        """ Returns file path for saving simulated tracks """
        fname = f'{self._get_id_string(case_id, real_id)}_tracks'
        return os.path.join(dirname, fname)

    def plot_simulated_tracks(self, plot_turbs=True, show=False) -> None:
        """ Plots simulated tracks """
        print('Plotting simulated tracks..')
        lwidth = 0.15 if self.track_count > 251 else 0.4
        # Use analysis-resolution information
        elevation = highRes2lowRes(
            self.get_terrain_elevation(), self.resolution_terrain, self.resolution)
        xgrid, ygrid = self.get_terrain_grid(self.resolution, self.gridsize)
        for case_id in self.case_ids:
            updrafts = self.load_updrafts(case_id, apply_threshold=True)
            for real_id, _ in enumerate(updrafts):
                fig, axs = plt.subplots(figsize=self.fig_size)
                _ = axs.imshow(elevation, alpha=0.75, cmap='Greys',
                               origin='lower', extent=self.extent)
                fname = self._get_tracks_fname(
                    case_id, real_id, self.mode_data_dir)
                with open(f'{fname}.pkl', 'rb') as fobj:
                    tracks = pickle.load(fobj)
                    for itrack in tracks:
                        axs.plot(xgrid[itrack[0, 1]], ygrid[itrack[0, 0]], 'b.',
                                 markersize=1.0)
                        axs.plot(xgrid[itrack[:, 1]], ygrid[itrack[:, 0]],
                                 '-r', linewidth=lwidth, alpha=0.5)
                _, _ = create_gis_axis(fig, axs, None, self.km_bar)
                if plot_turbs:
                    self.plot_turbine_locations(axs)
                if self.track_start_region_rotation == 0:
                    left = self.extent[0] + self.track_start_region[0] * 1000.
                    bottom = self.extent[2] + \
                        self.track_start_region[2] * 1000.
                    width = self.track_start_region[1] - \
                        self.track_start_region[0]
                    hght = self.track_start_region[3] - \
                        self.track_start_region[2]
                    rect = mpatches.Rectangle((left, bottom), width * 1000.,
                                              hght * 1000., alpha=0.2,
                                              edgecolor='none', facecolor='b')
                    axs.add_patch(rect)
                else:
                    # TODO: transform rectangle (e.g., https://stackoverflow.com/questions/43000288/unable-to-rotate-a-matplotlib-patch-object-about-a-specific-point-using-rotate-d)
                    print(
                        f'Starting region is rotated by {self.track_start_region_rotation:g} deg')
                axs.set_xlim([self.extent[0], self.extent[1]])
                axs.set_ylim([self.extent[2], self.extent[3]])
                fname = self._get_tracks_fname(
                    case_id, real_id, self.mode_fig_dir)
                self.save_fig(fig, f'{fname}.png', show)

    def plot_simulated_tracks_altamont(self, plot_turbs=True, show=False,
                                       fig=None, axs=None, in_alpha=0.25) -> None:
        """ Plots simulated tracks """
        print('Plotting simulated tracks..')
        lwidth = 0.15 if self.track_count > 251 else 0.4
        #elevation = self.get_terrain_elevation()
        elevation = highRes2lowRes(
            self.get_terrain_elevation(), self.resolution_terrain, self.resolution)
        xgrid, ygrid = self.get_terrain_grid(self.resolution, self.gridsize)
        for case_id in self.case_ids:
            updrafts = self.load_updrafts(case_id, apply_threshold=True)
            for real_id, _ in enumerate(updrafts):
                if axs is None:
                    fig, axs = plt.subplots(figsize=self.fig_size)
                _ = axs.imshow(elevation, alpha=0.75, cmap='Greys',
                               origin='lower', extent=self.extent)
                fname = self._get_tracks_fname(
                    case_id, real_id, self.mode_data_dir)
                with open(f'{fname}.pkl', 'rb') as fobj:
                    tracks = pickle.load(fobj)
                    for itrack in tracks:
                        axs.plot(xgrid[itrack[0, 1]], ygrid[itrack[0, 0]], 'b.',
                                 markersize=1.0)
                        axs.plot(xgrid[itrack[:, 1]], ygrid[itrack[:, 0]],
                                 '-r', linewidth=lwidth, alpha=in_alpha)
                _, _ = create_gis_axis(fig, axs, None, self.km_bar)
                if plot_turbs:
                    self.plot_turbine_locations(axs)
                if self.track_start_region_rotation == 0:
                    left = self.extent[0] + self.track_start_region[0] * 1000.
                    bottom = self.extent[2] + \
                        self.track_start_region[2] * 1000.
                    width = self.track_start_region[1] - \
                        self.track_start_region[0]
                    hght = self.track_start_region[3] - \
                        self.track_start_region[2]
                    rect = mpatches.Rectangle((left, bottom), width * 1000.,
                                              hght * 1000., alpha=0.2,
                                              edgecolor='none', facecolor='b')
                    axs.add_patch(rect)
                else:
                    # TODO: transform rectangle (e.g., https://stackoverflow.com/questions/43000288/unable-to-rotate-a-matplotlib-patch-object-about-a-specific-point-using-rotate-d)
                    print(
                        f'Starting region is rotated by {self.track_start_region_rotation:g} deg')
                axs.set_xlim([self.extent[0], self.extent[1]])
                axs.set_ylim([self.extent[2], self.extent[3]])
                fname = self._get_tracks_fname(
                    case_id, real_id, self.mode_fig_dir)
        return fig, axs

    def plot_simulated_tracks_altamont_zoom(self, plot_turbs=True, show=False,
                                            fig=None, axs=None, in_alpha=0.25) -> None:
        """ Plots simulated tracks
          This function zooms the terrain data and plots the same tracks on top of the zoomed-in terrain
        """
        print('Plotting simulated tracks..')
        lwidth = 0.15 if self.track_count > 251 else 0.4
        elevation = self.get_terrain_elevation()
        #elevation = highRes2lowRes(self.get_terrain_elevation(), self.resolution_terrain, self.resolution)

        # manually crop the elevation map
        elevation = elevation[0:int(self.gridsize[0]), 0:int(self.gridsize[1])]
        xgrid, ygrid = self.get_terrain_grid(self.resolution, self.gridsize)
        for case_id in self.case_ids:
            updrafts = self.load_updrafts(case_id, apply_threshold=True)
            for real_id, _ in enumerate(updrafts):
                if axs is None:
                    fig, axs = plt.subplots(figsize=self.fig_size)
                _ = axs.imshow(elevation, alpha=0.75, cmap='Greys',
                               origin='lower', extent=self.extent)
                fname = self._get_tracks_fname(
                    case_id, real_id, self.mode_data_dir)
                with open(f'{fname}.pkl', 'rb') as fobj:
                    tracks = pickle.load(fobj)
                    for itrack in tracks:
                        axs.plot(xgrid[itrack[0, 1]], ygrid[itrack[0, 0]], 'b.',
                                 markersize=1.0)
                        axs.plot(xgrid[itrack[:, 1]], ygrid[itrack[:, 0]],
                                 '-r', linewidth=lwidth, alpha=in_alpha)
                _, _ = create_gis_axis(fig, axs, None, self.km_bar)
                if plot_turbs:
                    self.plot_turbine_locations(axs)
                if self.track_start_region_rotation == 0:
                    left = self.extent[0] + self.track_start_region[0] * 1000.
                    bottom = self.extent[2] + \
                        self.track_start_region[2] * 1000.
                    width = self.track_start_region[1] - \
                        self.track_start_region[0]
                    hght = self.track_start_region[3] - \
                        self.track_start_region[2]
                    rect = mpatches.Rectangle((left, bottom), width * 1000.,
                                              hght * 1000., alpha=0.2,
                                              edgecolor='none', facecolor='b')
                    axs.add_patch(rect)
                else:
                    # TODO: transform rectangle (e.g., https://stackoverflow.com/questions/43000288/unable-to-rotate-a-matplotlib-patch-object-about-a-specific-point-using-rotate-d)
                    print(
                        f'Starting region is rotated by {self.track_start_region_rotation:g} deg')
                axs.set_xlim([self.extent[0], self.extent[1]])
                axs.set_ylim([self.extent[2], self.extent[3]])
                fname = self._get_tracks_fname(
                    case_id, real_id, self.mode_fig_dir)
        return fig, axs

    def plot_simulated_tracks_HSSRS(self, plot_turbs=True, show=False) -> None:
        """ Plots simulated tracks """
        print('Plotting simulated tracks..')
        lwidth = 0.3 if self.track_count > 250 else 0.75
        # Use analysis-resolution information
        elevation = highRes2lowRes(
            self.get_terrain_elevation(), self.resolution_terrain, self.resolution)
        xgrid, ygrid = self.get_terrain_grid(self.resolution, self.gridsize)
        for case_id in self.case_ids:
            for real_id in range(self.thermals_realization_count):
                fig, axs = plt.subplots(figsize=self.fig_size)
                _ = axs.imshow(elevation, alpha=0.75, cmap='Greys',
                               origin='lower', extent=self.extent)
                fname = self._get_tracks_fname(
                    case_id, real_id, self.mode_data_dir)
                with open(f'{fname}.pkl', 'rb') as fobj:
                    tracks = pickle.load(fobj)
                    for itrack in tracks:
                        axs.plot(xgrid[itrack[0, 1]], ygrid[itrack[0, 0]], 'b.',
                                 markersize=2.0)
                        axs.plot(xgrid[itrack[:, 1]], ygrid[itrack[:, 0]],
                                 '-r', linewidth=lwidth, alpha=0.5)
                _, _ = create_gis_axis(fig, axs, None, self.km_bar)
                if plot_turbs:
                    self.plot_turbine_locations(axs)
                if self.track_start_region_rotation == 0:
                    left = self.extent[0] + self.track_start_region[0] * 1000.
                    bottom = self.extent[2] + \
                        self.track_start_region[2] * 1000.
                    width = self.track_start_region[1] - \
                        self.track_start_region[0]
                    hght = self.track_start_region[3] - \
                        self.track_start_region[2]
                    rect = mpatches.Rectangle((left, bottom), width * 1000.,
                                              hght * 1000., alpha=0.2,
                                              edgecolor='none', facecolor='b')
                    axs.add_patch(rect)
                else:
                    # TODO: transform rectangle (e.g., https://stackoverflow.com/questions/43000288/unable-to-rotate-a-matplotlib-patch-object-about-a-specific-point-using-rotate-d)
                    print(
                        f'Starting region is rotated by {self.track_start_region_rotation:g} deg')
                axs.set_xlim([self.extent[0], self.extent[1]])
                axs.set_ylim([self.extent[2], self.extent[3]])

                xtext = self.extent[0] + 0.5 * \
                    (self.extent[1] - self.extent[0])
                ytext = self.extent[2] + 0.04 * \
                    (self.extent[3] - self.extent[2])
                if self.movement_ruleset != 'step_ahead_look_ahead':
                    self.look_ahead_dist = 0.0
                if self.orographic_model == 'original':
                    orogmod = 'DO04'
                if self.orographic_model == 'improved':
                    orogmod = 'EVVE'
                if self.movement_ruleset == 'local_moves':
                    axs.text(xtext, ytext, f'move model = {self.sim_movement}\n'
                                           f'ruleset = {self.movement_ruleset}\n'
                                           f'wo model = {orogmod}\n'
                                           f'look ahead dist (km) = {self.look_ahead_dist/1000:.1f}\n'
                                           f'thermal intensity scale = {self.thermal_intensity_scale:.1f}\n'
                                           f'wind = {self.sim_mode} {self.uniform_winddirn_h:.0f} deg {self.uniform_windspeed_h:.1f} m/s\n'
                                           f'random walk freq = {1/self.random_walk_freq:.4f}\n'
                                           f'n tracks = {self.track_count:5d}',
                                           # f'n thermal realizations = {self.thermals_realization_count:3d}\n'\
                             fontsize='xx-small', color='black')
                else:
                    axs.text(xtext, ytext, f'move dir (deg) = {self.track_direction:6.1f} \n'
                                           f'move model = {self.sim_movement} \n'
                                           f'ruleset = {self.movement_ruleset} \n'
                                           f'wo model = {orogmod}\n'
                                           f'look ahead dist (km)= {self.look_ahead_dist/1000:2.1f} \n'
                                           f'thermal intensity scale = {self.thermal_intensity_scale:4.1f} \n'
                                           f'wind = {self.sim_mode} {self.uniform_winddirn_h:4.0f} deg {self.uniform_windspeed_h:4.1f} m/s \n'
                                           f'random walk freq = {1/self.random_walk_freq:8.6f} \n'
                                           f'n tracks = {self.track_count:5d}',
                                           # f'n thermal realizations = {self.thermals_realization_count:3d}\n'\
                             fontsize='xx-small', color='black')

                fname = self._get_tracks_fname(
                    case_id, real_id, self.mode_fig_dir)
                self.save_fig(fig, f'{fname}.png', show)

########### Plot updrafts and WTK layers ###########

    def plot_updrafts_zoom(self, apply_threshold=True, plot_turbs=True,
                           show=False, plot='imshow', figsize=None, maxval=None, in_alpha=0.25) -> None:
        """ Plot updrafts with or without applying the threshold
        function to zoom on the lower left region
        """
        print('Plotting updraft fields..')
        # Updraft field is on the coarse analysis grid
        lwidth = 0.15 if self.track_count > 251 else 0.4
        xgrid, ygrid = self.get_terrain_grid(self.resolution, self.gridsize)
        if figsize is None:
            figsize = self.fig_size
        for case_id in self.case_ids:
            updrafts = self.load_updrafts(case_id, apply_threshold)
            for real_id, updraft in enumerate(updrafts):

                # get the corner and re-size so everything else is the same
                updraft = updraft[0:int(
                    self.gridsize[0] / 5), 0:int(self.gridsize[1] / 5)]
                updraft = downsample_field(updraft, 50, 10)

                fig, axs = plt.subplots(figsize=self.fig_size)
                if maxval is None:
                    maxval = min(max(1, int(round(np.nanmean(updraft)))), 5)

                if plot == 'pcolormesh':
                    #xx = self.xx[0:int(self.gridsize[0]/5),0:int(self.gridsize[1]/5)]
                    #yy = self.yy[0:int(self.gridsize[0]/5),0:int(self.gridsize[1]/5)]
                    xx = self.xx
                    yy = self.yy
                    curm = axs.pcolormesh(xx, yy, updraft.T,
                                          cmap='viridis', rasterized=True, vmin=0, vmax=maxval)
                else:
                    curm = axs.imshow(updraft, cmap='viridis', extent=self.extent,
                                      origin='lower', vmin=0, vmax=maxval)
                fname = self._get_tracks_fname(
                    case_id, real_id, self.mode_data_dir)
                with open(f'{fname}.pkl', 'rb') as fobj:
                    tracks = pickle.load(fobj)
                    for itrack in tracks:
                        # axs.plot(xgrid[itrack[0, 1]], ygrid[itrack[0, 0]], 'b.',
                        #         markersize=1.0)
                        axs.plot(xgrid[itrack[:, 1]], ygrid[itrack[:, 0]],
                                 '-r', linewidth=lwidth, alpha=in_alpha)
                cbar, _ = create_gis_axis(fig, axs, curm, self.km_bar)
                if real_id == 0:
                    lbl = 'Orographic updraft (m/s)'
                else:
                    lbl = 'Orographic + Thermal (m/s)'
                if apply_threshold:
                    lbl += ', usable'
                cbar.set_label(lbl)
                if plot_turbs:
                    self.plot_turbine_locations(axs)
                fname = f'{self._get_id_string(case_id, real_id)}_updraft.png'
                fpath = os.path.join(self.mode_fig_dir, fname)
                #self.save_fig(fig, fpath, show)

    def plot_updrafts(self, apply_threshold=True, plot_turbs=True,
                      show=False, plot='imshow', figsize=None, vmax=None) -> None:
        """ Plot updrafts with or without applying the threshold """
        print('Plotting updraft fields..')
        # Updraft field is on the coarse analysis grid
        if figsize is None:
            figsize = self.fig_size
        for case_id in self.case_ids:
            updrafts = self.load_updrafts(case_id, apply_threshold)

            # Identify if one updraft field or list of updrafts for enumerate
            for real_id, updraft in enumerate(updrafts):
                fig, axs = plt.subplots(figsize=self.fig_size)
                if vmax is None:
                    maxval = min(max(1, int(round(np.mean(updraft)))), 5)
                else:
                    maxval = vmax
                if plot == 'pcolormesh':
                    curm = axs.pcolormesh(self.xx, self.yy, updraft.T,
                                          cmap='viridis', rasterized=True, vmin=0, vmax=maxval)
                else:
                    curm = axs.imshow(updraft, cmap='viridis', extent=self.extent,
                                      origin='lower', vmin=0, vmax=maxval)
                cbar, _ = create_gis_axis(fig, axs, curm, self.km_bar)
                if real_id == 0:
                    lbl = 'Orographic updraft (m/s)'
                else:
                    lbl = 'Orographic + Thermal (m/s)'
                if apply_threshold:
                    lbl += ', usable'
                cbar.set_label(lbl)
                if plot_turbs:
                    self.plot_turbine_locations(axs)
                fname = f'{self._get_id_string(case_id, real_id)}_updraft.png'
                fpath = os.path.join(self.mode_fig_dir, fname)
                self.save_fig(fig, fpath, show)

    def plot_thermal_updrafts(self, plot_turbs=True, show=False, vmax=None) -> None:
        """ Plot estimated thermal updrafts """
        for case_id in self.case_ids:
            for real_id in range(self.thermals_realization_count):

                fname_thermal = self._get_thermal_updraft_fname(
                    case_id, real_id, self.mode_data_dir)
                thermalupdraft = np.load(f'{fname_thermal}.npy')

                if vmax is None:
                    maxval = min(
                        max(1, int(round(np.mean(thermalupdraft)))), 5)

                fig, axs = plt.subplots(figsize=self.fig_size)
                curm = axs.imshow(thermalupdraft, cmap='viridis',
                                  extent=self.extent, origin='lower',
                                  vmin=0, vmax=vmax)

                cbar, _ = create_gis_axis(fig, axs, curm, self.km_bar)
                cbar.set_label('Thermal updraft velocity (m/s)')

                if plot_turbs:
                    self.plot_turbine_locations(axs)

                fname = f'{self._get_id_string(case_id, real_id)}_thermal_updrafts.png'
                fpath = os.path.join(self.mode_fig_dir, fname)
                self.save_fig(fig, fpath, show)

    def plot_sm_orographic_updrafts(self, plot_turbs=True, show=False) -> None:
        """ Plot orographic updrafts """
        for case_id in self.case_ids:
            orograph = np.load(self._get_orograph_fpath(case_id))
            # db added # rt: resolution independent. needs to be fixed
            wo_smoothed = ndimage.gaussian_filter(
                orograph, sigma=3, mode='constant')
            fig, axs = plt.subplots(figsize=self.fig_size)
            maxval = min(max(2, int(round(np.nanmean(orograph)))), 5)
            curm = axs.imshow(wo_smoothed, cmap='viridis',
                              extent=self.extent, origin='lower',
                              vmin=0, vmax=maxval)
            cbar, _ = create_gis_axis(fig, axs, curm, self.km_bar)
            cbar.set_label('Smoothed Wo (m/s)')
            if plot_turbs:
                self.plot_turbine_locations(axs)
            fname = os.path.join(
                self.mode_fig_dir, f'{case_id}_smoothed.orograph.png')
            self.save_fig(fig, fname, show)

    def plot_wtk_layers(self, plot_turbs=True, show=False) -> None:
        """ Plot all the layers in Wind Toolkit data """
        try:
            for dtime, case_id in zip(self.dtimes, self.case_ids):
                wtk_df = self.wtk.get_dataframe_for_this_time(dtime)
                for wtk_lyr in self.wtk.varnames:
                    vardata = wtk_df.loc[:, wtk_lyr].values.flatten()
                    interp_data = self._interpolate_wtk_vardata(vardata)
                    fig, axs = plt.subplots(figsize=self.fig_size)
                    if 'direction' in wtk_lyr:
                        plotopts = dict(cmap='hsv', vmin=0, vmax=360)
                    else:
                        plotopts = dict(cmap='viridis')
                    curm = axs.imshow(interp_data,
                                      origin='lower', extent=self.extent,
                                      alpha=0.75, **plotopts)
                    cbar, _ = create_gis_axis(fig, axs, curm, self.km_bar)
                    cbar.set_label(wtk_lyr)
                    if 'direction' in wtk_lyr:
                        cbar.set_ticks(np.arange(0, 361, 45))
                        cbar.set_ticklabels(
                            ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'])
                    axs.set_xlim([self.extent[0], self.extent[1]])
                    axs.set_ylim([self.extent[2], self.extent[3]])
                    if plot_turbs:
                        self.plot_turbine_locations(axs)
                    fname = f'{case_id}_{wtk_lyr}.png'
                    self.save_fig(fig, os.path.join(self.mode_fig_dir, fname),
                                  show)
        except AttributeError as _:
            print('No WTK data to plot in uniform mode!')


########### Compute and plot presence maps ###########


    def _plot_presence(self, in_prob, in_val, plot_turbs, wfarm_level=False, normalized=True):
        """Plots a presence density """
        fig, axs = plt.subplots(figsize=self.fig_size)
        in_prob[in_prob <= in_val] = 0.
        if normalized:
            vmin = in_val
            vmax = 1.0
        else:
            maxval = np.max(in_prob)
            vmin = in_val * maxval
            vmax = maxval
        _ = axs.imshow(in_prob, extent=self.extent, origin='lower',
                       cmap='Reds', alpha=0.75,
                       norm=LogNorm(vmin=vmin, vmax=vmax))
        if wfarm_level:
            _, _ = create_gis_axis(fig, axs, None, 1.)
        else:
            _, _ = create_gis_axis(fig, axs, None, self.km_bar)
        if plot_turbs:
            self.plot_turbine_locations(axs)
        axs.set_xlim([self.extent[0], self.extent[1]])
        axs.set_ylim([self.extent[2], self.extent[3]])
        return fig, axs

    def _plot_presence_altamont(self, in_prob, in_val, fig=None, axs=None,
                                plot_turbs=False, wfarm_level=False, normalized=True):
        """Plots a presence density """
        if axs is None:
            fig, axs = plt.subplots(figsize=self.fig_size)
        in_prob[in_prob <= in_val] = 0.
        if normalized:
            vmin = in_val
            vmax = 1.0
        else:
            maxval = np.max(in_prob)
            vmin = in_val * maxval
            vmax = maxval
        _ = axs.imshow(in_prob, extent=self.extent, origin='lower',
                       cmap='Reds', alpha=0.75,
                       norm=LogNorm(vmin=vmin, vmax=vmax))
        if wfarm_level:
            _, _ = create_gis_axis(fig, axs, None, 1.)
        else:
            _, _ = create_gis_axis(fig, axs, None, self.km_bar)
        if plot_turbs:
            self.plot_turbine_locations(axs)
        axs.set_xlim([self.extent[0], self.extent[1]])
        axs.set_ylim([self.extent[2], self.extent[3]])
        return fig, axs

    def _plot_presence_HSSRS(self, in_prob, countmax, in_val, plot_turbs, wfarm_level=False):
        """Plots a presence density """
        fig, axs = plt.subplots(figsize=self.fig_size)
        in_prob[in_prob <= in_val] = 0.
        _ = axs.imshow(in_prob, extent=self.extent, origin='lower',
                       cmap='Reds', alpha=0.75,
                       vmin=0.0, vmax=self.track_count * self.thermals_realization_count / 100.)  # using absolute scale = # birds/100
        # norm=LogNorm(vmin=0, vmax=self.track_count*self.thermals_realization_count/100.)) #note changed vmax
        if wfarm_level:
            _, _ = create_gis_axis(fig, axs, None, 1.)
        else:
            _, _ = create_gis_axis(fig, axs, None, self.km_bar)
        if plot_turbs:
            self.plot_turbine_locations(axs)
        axs.set_xlim([self.extent[0], self.extent[1]])
        axs.set_ylim([self.extent[2], self.extent[3]])

        xtext = self.extent[0] + 0.5 * (self.extent[1] - self.extent[0])
        ytext = self.extent[2] + 0.04 * (self.extent[3] - self.extent[2])
        if self.orographic_model == 'original':
            orogmod = 'DO04'
        if self.orographic_model == 'improved':
            orogmod = 'EVVE'
        axs.text(xtext, ytext, f'move dir (deg) = {self.track_direction:6.1f}\n'
                               f'move model = {self.sim_movement}\n'
                               f'ruleset = {self.movement_ruleset}\n'
                               f'wo model = {orogmod}\n'
                               f'look ahead dist (km) = {self.look_ahead_dist/1000:2.1f}\n'
                               f'thermal intensity scale = {self.thermal_intensity_scale:4.1f}\n'
                               f'wind = {self.sim_mode} {self.uniform_winddirn_h:4.0f} deg {self.uniform_windspeed_h:4.1f} m/s\n'
                               f'random walk freq = {1./self.random_walk_freq:8.6f}\n'
                               f'n tracks = {self.track_count*self.thermals_realization_count:5d}\n'\
                               # f'n thermal realizations = {self.thermals_realization_count:3d}\n'\
                               f'max_count/n_tracks = {100.*countmax/(self.track_count*self.thermals_realization_count):4.2f}',
                 fontsize='xx-small', color='black')

        return fig, axs

    def plot_presence_map_HSSRS(
        self,
        plot_turbs=True,
        radius: float = 250.,
        show=False,
        minval=0.1,
        plot_all: bool = False
    ) -> None:
        """ Plot presence maps """
        print('Plotting presence density map..')
        # Use analysis-resolution information
        elevation = highRes2lowRes(
            self.get_terrain_elevation(), self.resolution_terrain, self.resolution)
        summary_prob = np.zeros_like(elevation)
        krad = min(max(radius / self.resolution, 2), min(self.gridsize) / 2)
        for case_id in self.case_ids:
            updrafts = self.load_updrafts(case_id, apply_threshold=True)
            case_prob = np.zeros_like(elevation)
            for real_id,_ in enumerate(updrafts):
                fname = self._get_tracks_fname(
                    case_id, real_id, self.mode_data_dir)
                with open(f'{fname}.pkl', 'rb') as fobj:
                    tracks = pickle.load(fobj)
                prprob = compute_smooth_presence_counts_HSSRS(
                    tracks, self.gridsize, int(round(krad)))
                #prprob /= np.amax(prprob)
                case_prob += prprob
                if plot_all:
                    fig, ax = self._plot_presence_HSSRS(
                        prprob, minval, plot_turbs)
                    fname = self._get_presence_fname(case_id, real_id,
                                                     self.mode_fig_dir)
                    self.save_fig(fig, f'{fname}.png', show)
            #case_prob /= np.amax(case_prob)
            summary_prob += case_prob
            print('Max summary presence count =', np.amax(summary_prob))
            countmax = np.amax(summary_prob)
            fig, ax = self._plot_presence_HSSRS(
                case_prob, countmax, minval, plot_turbs)
            fname = f'{self._get_id_string(case_id)}_presence.png'
            fpath = os.path.join(self.mode_fig_dir, fname)
            self.save_fig(fig, fpath, show)
        #summary_prob /= np.amax(summary_prob)

        fname = os.path.join(self.mode_data_dir, 'summary_presence')
        np.save(f'{fname}.npy', summary_prob.astype(np.float32))
        if len(self.case_ids) > 1:
            fig, ax = self._plot_presence_HSSRS(
                summary_prob, minval, plot_turbs)
            fpath = os.path.join(self.mode_fig_dir, 'summary_presence.png')
            self.save_fig(fig, fpath, show)

        return fig, ax

    def calc_presence_map(
        self,
        tracks: np.ndarray,
        radius: float = 1000.
    ) -> np.ndarray:
        krad = min(max(radius / self.resolution, 2), min(self.gridsize) / 2)
        prprob = compute_smooth_presence_counts(
            tracks, self.gridsize, int(round(krad)))
        prprob /= np.amax(prprob)
        return prprob

    def plot_presence_map(
        self,
        plot_turbs=True,
        radius: float = 1000.,
        show=False,
        minval=0.1,
        plot_all: bool = False
    ) -> None:
        """ Plot presence maps """
        print('Plotting presence density map..')
        # Use analysis-resolution information
        elevation = highRes2lowRes(
            self.get_terrain_elevation(), self.resolution_terrain, self.resolution)
#        summary_prob = np.zeros_like(elevation)
        summary_prob = None
#        krad = min(max(radius / self.resolution, 2), min(self.gridsize) / 2)
        for case_id in self.case_ids:
            updrafts = self.load_updrafts(case_id, apply_threshold=True)
            case_prob = np.zeros_like(updrafts[0])
            if summary_prob is None:
                summary_prob = np.zeros_like(updrafts[0])
            for real_id, _ in enumerate(updrafts):
                fname = self._get_tracks_fname(
                    case_id, real_id, self.mode_data_dir)
                with open(f'{fname}.pkl', 'rb') as fobj:
                    tracks = pickle.load(fobj)
#                prprob = compute_smooth_presence_counts(
#                    tracks, self.gridsize, int(round(krad)))
#                prprob /= np.amax(prprob)
                prprob = self.calc_presence_map(tracks, radius)
                case_prob += prprob
                if plot_all:
                    fig, _ = self._plot_presence(prprob, minval, plot_turbs)
                    fname = self._get_presence_fname(case_id, real_id,
                                                     self.mode_fig_dir)
                    self.save_fig(fig, f'{fname}.png', show)
            case_prob /= np.amax(case_prob)
            summary_prob += case_prob
            fig, _ = self._plot_presence(case_prob, minval, plot_turbs)
            fname = f'{self._get_id_string(case_id)}_presence.png'
            fpath = os.path.join(self.mode_fig_dir, fname)
            self.save_fig(fig, fpath, show)
        summary_prob /= np.amax(summary_prob)
        fname = os.path.join(self.mode_data_dir, 'summary_presence')
        np.save(f'{fname}.npy', summary_prob.astype(np.float32))
        if len(self.case_ids) > 1:
            fig, _ = self._plot_presence(summary_prob, minval, plot_turbs)
            fpath = os.path.join(self.mode_fig_dir, 'summary_presence.png')
            self.save_fig(fig, fpath, show)

    def plot_presence_map_altamont(
        self,
        plot_turbs=False,
        radius: float = 1000.,
        norm=True,
        show=False,
        minval=0.1,
        plot_all: bool = False
    ) -> None:
        """ Plot presence maps """
        print('Plotting presence density map..')
        #elevation = self.get_terrain_elevation()
        elevation = highRes2lowRes(
            self.get_terrain_elevation(), self.resolution_terrain, self.resolution)
        summary_prob = np.zeros_like(elevation)
        krad = min(max(radius / self.resolution, 2), min(self.gridsize) / 2)
        for case_id in self.case_ids:
            updrafts = self.load_updrafts(case_id, apply_threshold=True)
            case_prob = np.zeros_like(updrafts[0])
            for real_id, _ in enumerate(updrafts):
                fname = self._get_tracks_fname(
                    case_id, real_id, self.mode_data_dir)
                with open(f'{fname}.pkl', 'rb') as fobj:
                    tracks = pickle.load(fobj)
                prprob = compute_smooth_presence_counts(
                    tracks, self.gridsize, int(round(krad)))
                if norm:
                    prprob /= np.amax(prprob)
                case_prob += prprob
                if plot_all:
                    fig, _ = self._plot_presence(
                        prprob, minval, plot_turbs, normalized=norm)
                    fname = self._get_presence_fname(case_id, real_id,
                                                     self.mode_fig_dir)
                    self.save_fig(fig, f'{fname}.png', show)
            if norm:
                case_prob /= np.amax(case_prob)
            summary_prob += case_prob
            # fig, _ = self._plot_presence(case_prob, minval, plot_turbs)
            # fname = f'{self._get_id_string(case_id)}_presence.png'
            # fpath = os.path.join(self.mode_fig_dir, fname)
            # self.save_fig(fig, fpath, show)
        if norm:
            summary_prob /= np.amax(summary_prob)
        fname = os.path.join(self.mode_data_dir, 'summary_presence')
        np.save(f'{fname}.npy', summary_prob.astype(np.float32))
        fig, axs = self._plot_presence(summary_prob, minval, plot_turbs)
        fpath = os.path.join(self.mode_fig_dir, 'summary_presence.png')
        self.save_fig(fig, fpath, show)
        return fig, axs, summary_prob

    def _get_presence_fname(self, case_id: str, real_id: int, dirname: str):
        """ Returns file path for saving presence """
        fname = f'{self._get_id_string(case_id, real_id)}_presence'
        return os.path.join(dirname, fname)

    def plot_windplant_presence_map_w(
        self,
        pname,
        radius: int = 100.,
        plot_turbs=True,
        show=False,
        minval=0.05,
        pad: float = 2000.
    ) -> None:
        """ Plot presence maps """
        print('Plotting presence density map..')
        # Use analysis-resolution information
        elevation = highRes2lowRes(
            self.get_terrain_elevation(), self.resolution_terrain, self.resolution)
        summary_prob = np.zeros_like(elevation)
        xloc, yloc = self.turbines.get_locations_for_this_project(pname)
        krad = min(max(radius / self.resolution, 2), min(self.gridsize) / 2)
        for case_id in self.case_ids:
            with open(self._get_tracks_fpath(case_id), 'rb') as fobj:
                tracks = pickle.load(fobj)
            if self.sim_movement == 'fluid-analogy':
                prprob = compute_smooth_presence_counts(
                    tracks, self.gridsize, self.presence_smoothing_radius)
            elif self.sim_movement == 'heuristics':
                prprob = compute_smooth_presence_counts_HSSRS(
                    tracks, self.gridsize, self.presence_smoothing_radius)
            prprob /= self.track_count
            prprob /= np.amax(prprob)
            fig, axs = plt.subplots(figsize=self.fig_size)
            # _ = axs.imshow(elevation, alpha=0.75, cmap='Greys',
            #                origin='lower', extent=self.extent)
            prprob[prprob <= minval] = 0.
            _ = axs.imshow(prprob, extent=self.extent, origin='lower',
                           cmap='Reds', alpha=0.75,
                           norm=LogNorm(vmin=minval, vmax=1.0))
            # cm = axs.imshow(prprob, extent=self.extent, origin='lower',
            #                 cmap='Reds', alpha=0.75)
            _, _ = create_gis_axis(fig, axs, None, self.km_bar)
            if plot_turbs:
                self.plot_turbine_locations(axs)

            xtext = self.extent[0] + 0.5 * (self.extent[1] - self.extent[0])
            ytext = self.extent[2] + 0.04 * (self.extent[3] - self.extent[2])
            axs.text(xtext, ytext, 'PAM(deg) = %6.1f\nmove model = %s\nruleset = %s\nlook ahead dist (km)= %2.1f'
                     '\nthermal intensity scale =%4.1f\nwind = %s %4.0f %4.1f mps\nrandom walk freq = %6.4f\nn tracks = %5d'
                     % (self.track_direction, self.sim_movement, self.movement_ruleset, self.look_ahead_dist / 1000., self.thermal_intensity_scale,
                        self.sim_mode, self.uniform_winddirn_h, self.uniform_windspeed_h, 1. / self.random_walk_freq, self.track_count),
                     fontsize='xx-small', color='black')

            axs.set_xlim([self.extent[0], self.extent[1]])
            axs.set_ylim([self.extent[2], self.extent[3]])

            fname = f'{case_id}_{int(self.track_direction)}_presence.png'
            self.save_fig(fig, os.path.join(self.mode_fig_dir, fname), show)

    def plot_windplant_presence_map_altamont(
        self,
        pname,
        radius: int = 100.,
        plot_turbs=True,
        show=False,
        minval=0.05,
        pad: float = 2000.,
        axs=None
    ) -> None:
        """ Plot presence maps """
        print('Plotting presence density map..')
        # Use analysis-resolution information
        elevation = highRes2lowRes(
            self.get_terrain_elevation(), self.resolution_terrain, self.resolution)
        summary_prob = np.zeros_like(elevation)
        xloc, yloc = self.turbines.get_locations_for_this_project(pname)
        krad = min(max(radius / self.resolution, 2), min(self.gridsize) / 2)
        for case_id in self.case_ids:
            updrafts = self.load_updrafts(case_id, apply_threshold=True)
            case_prob = np.zeros_like(updrafts[0])
            for real_id, _ in enumerate(updrafts):
                fname = self._get_tracks_fname(
                    case_id, real_id, self.mode_data_dir)
                with open(f'{fname}.pkl', 'rb') as fobj:
                    tracks = pickle.load(fobj)
                prprob = compute_smooth_presence_counts(
                    tracks, self.gridsize, krad)
                prprob /= np.amax(prprob)
                case_prob += prprob
            case_prob /= np.amax(case_prob)
            summary_prob += case_prob
        summary_prob /= np.amax(summary_prob)
        fig, axs = self._plot_presence(summary_prob, minval, plot_turbs,
                                       wfarm_level=True)
        axs.set_xlim([min(xloc) - pad, max(xloc) + pad])
        axs.set_ylim([min(yloc) - pad, max(yloc) + pad])
        fpath = os.path.join(self.mode_fig_dir, f'presence_{pname}.png')
        self.save_fig(fig, fpath, show)

    def plot_windplant_presence_map(
        self,
        pname,
        radius: int = 100.,
        plot_turbs=True,
        show=False,
        minval=0.05,
        pad: float = 2000.
    ) -> None:
        """ Plot presence maps """
        print('Plotting presence density map..')
        # Use analysis-resolution information
        elevation = highRes2lowRes(
            self.get_terrain_elevation(), self.resolution_terrain, self.resolution)
        summary_prob = np.zeros_like(elevation)
        xloc, yloc = self.turbines.get_locations_for_this_project(pname)
        krad = min(max(radius / self.resolution, 2), min(self.gridsize) / 2)
        for case_id in self.case_ids:
            updrafts = self.load_updrafts(case_id, apply_threshold=True)
            case_prob = np.zeros_like(updrafts[0])
            for real_id, _ in enumerate(updrafts):
                fname = self._get_tracks_fname(
                    case_id, real_id, self.mode_data_dir)
                with open(f'{fname}.pkl', 'rb') as fobj:
                    tracks = pickle.load(fobj)
                prprob = compute_smooth_presence_counts(
                    tracks, self.gridsize, krad)
                prprob /= np.amax(prprob)
                case_prob += prprob
            case_prob /= np.amax(case_prob)
            summary_prob += case_prob
        summary_prob /= np.amax(summary_prob)
        fig, axs = self._plot_presence(summary_prob, minval, plot_turbs,
                                       wfarm_level=True)
        axs.set_xlim([min(xloc) - pad, max(xloc) + pad])
        axs.set_ylim([min(yloc) - pad, max(yloc) + pad])
        fpath = os.path.join(self.mode_fig_dir, f'presence_{pname}.png')
        self.save_fig(fig, fpath, show)

    # def get_turbine_presence(self) -> None:
    #     """ Get turbines list where relative presence is high """
    #     print('Plotting presence map for the study area..')
    #     # elevation = self.get_terrain_elevation()
    #     for case_id in self.case_ids:
    #         with open(self._get_tracks_fpath(case_id), 'rb') as fobj:
    #             tracks = pickle.load(fobj)
    #         prprob = compute_smooth_presence_counts(
    #             tracks, self.gridsize, self.presence_smoothing_radius)
    #         print(np.amax(prprob), np.amin(prprob))
    #         prprob /= self.track_count
    #         prprob /= np.amax(prprob)

    #         self.save_fig(fig, os.path.join(self.mode_fig_dir, fname), show)

    def plot_simulated_tracks_w(self, plot_turbs=True, show=False) -> None:
        """ Plots simulated tracks """
        print('Plotting simulated tracks..')
        lwidth = 0.1 if self.track_count > 251 else 0.4
        # Use analysis-resolution information
        elevation = highRes2lowRes(
            self.get_terrain_elevation(), self.resolution_terrain, self.resolution)
        xgrid, ygrid = self.get_terrain_grid()
        for case_id in self.case_ids:
            thermal = np.load(self._get_thermal_fpath(case_id))
            orograph = np.load(self._get_orographicupdraft_fpath(case_id))
            w_tot = thermal + orograph
            fig, axs = plt.subplots(figsize=self.fig_size)
            _ = axs.imshow(w_tot, cmap='viridis',  # cmap was previously 'Greys' and alpha parameter
                           origin='lower', extent=self.extent)
            with open(self._get_tracks_fpath(case_id), 'rb') as fobj:
                tracks = pickle.load(fobj)
                for itrack in tracks:
                    axs.plot(xgrid[itrack[0, 1]], ygrid[itrack[0, 0]], 'b.',
                             markersize=1.0)
                    axs.plot(xgrid[itrack[:, 1]], ygrid[itrack[:, 0]],
                             '-w', linewidth=lwidth, alpha=0.5)  # DB changed color to white
            _, _ = create_gis_axis(fig, axs, None, self.km_bar)
            if plot_turbs:
                self.plot_turbine_locations(axs)
            if self.track_start_region_rotation == 0:
                left = self.extent[0] + self.track_start_region[0] * 1000.
                bottom = self.extent[2] + self.track_start_region[2] * 1000.
                width = self.track_start_region[1] - self.track_start_region[0]
                hght = self.track_start_region[3] - self.track_start_region[2]
                rect = mpatches.Rectangle((left, bottom), width * 1000.,
                                          hght * 1000., alpha=0.2,
                                          edgecolor='none', facecolor='b')
                axs.add_patch(rect)
            else:
                # TODO: transform rectangle (e.g., https://stackoverflow.com/questions/43000288/unable-to-rotate-a-matplotlib-patch-object-about-a-specific-point-using-rotate-d)
                print(
                    f'Starting region is rotated by {self.track_start_region_rotation:g} deg')

            xtext = self.extent[0] + 0.5 * (self.extent[1] - self.extent[0])
            ytext = self.extent[2] + 0.04 * (self.extent[3] - self.extent[2])
            axs.text(xtext, ytext, 'PAM(deg) = %6.1f\nmove model = %s\nruleset = %s\nlook ahead dist (km)= %2.1f'
                     '\nthermal intensity scale =%4.1f\nwind = %s %4.0f %4.1f mps\nrandom walk freq = %6.4f\nn tracks = %5d'
                     % (self.track_direction, self.sim_movement, self.movement_ruleset, self.look_ahead_dist / 1000., self.thermal_intensity_scale,
                        self.sim_mode, self.uniform_winddirn_h, self.uniform_windspeed_h, 1. / self.random_walk_freq, self.track_count),
                     fontsize='xx-small', color='white')

            axs.set_xlim([self.extent[0], self.extent[1]])
            axs.set_ylim([self.extent[2], self.extent[3]])

            fname = f'{case_id}_{int(self.track_direction)}_tracks_wtot.png'
            self.save_fig(fig, os.path.join(self.mode_fig_dir, fname), show)

    def plot_plant_specific_presence_maps(self, show=False,
                                          minval=0.2) -> None:
        """ Plot presence maps for each power plant contained in study area"""
        print('Plotting presence map for each project..')
        smoothing_radius = int(self.presence_smoothing_radius / 2)
        pad = 2000.  # in meters
        for case_id in self.case_ids:
            with open(self._get_tracks_fpath(case_id), 'rb') as fobj:
                tracks = pickle.load(fobj)
            prprob = compute_smooth_presence_counts(
                tracks, self.gridsize, smoothing_radius)
            prprob[prprob <= minval] = 0.
            for pname in self.turbines.get_project_names():
                xloc, yloc = self.turbines.get_locations_for_this_project(
                    pname)
                fig, axs = plt.subplots()
                _ = axs.imshow(prprob, extent=self.extent, origin='lower',
                               cmap='Reds', alpha=0.75,
                               norm=LogNorm(vmin=minval, vmax=1.0))
                _, _ = create_gis_axis(fig, axs, None, 1)
                axs.set_xlim([min(xloc) - pad, max(xloc) + pad])
                axs.set_ylim([min(yloc) - pad, max(yloc) + pad])
                self.plot_turbine_locations(axs)
                fname = f'{case_id}_{int(self.track_direction)}_{pname}_presence.png'
                self.save_fig(fig, os.path.join(
                    self.mode_fig_dir, fname), show)


########### Plotting terrain features ###########

    def plot_terrain_features(self, plot_turbs=True, show=False) -> None:
        """ Plots terrain layers """
        print('Plotting terrain layers..', flush=True)
        self.plot_terrain_elevation(plot_turbs, show)
        self.plot_terrain_slope(plot_turbs, show)
        self.plot_terrain_aspect(plot_turbs, show)

    def plot_terrain_elevation(self, plot_turbs=True, show=False) -> None:
        """ Plotting terrain elevation """
        elevation = self.get_terrain_elevation()
        fig, axs = plt.subplots(figsize=self.fig_size)
        curm = axs.imshow(elevation, cmap='terrain',
                          extent=self.extent, origin='lower')
        cbar, _ = create_gis_axis(fig, axs, curm, self.km_bar)
        cbar.set_label('Elevation (m)')
        if plot_turbs:
            self.plot_turbine_locations(axs)
        self.save_fig(fig, os.path.join(self.fig_dir, 'elevation.png'), show)

    def plot_terrain_elevation_altamont(self, plot_turbs=True, show=False,
                                        fig=None, axs=None, colorbar=True, **kwargs) -> None:
        """ Plotting terrain elevation """
        #elevation = self.get_terrain_elevation()
        elevation = highRes2lowRes(
            self.get_terrain_elevation(), self.resolution_terrain, self.resolution)
        if axs is None:
            fig, axs = plt.subplots(figsize=self.fig_size)
        curm = axs.imshow(elevation / 1000., cmap='terrain',
                          extent=self.extent, origin='lower', **kwargs)
        if colorbar:
            cbar, _ = create_gis_axis(fig, axs, curm, self.km_bar)
            cbar.set_label('Altitude (km)')

        if plot_turbs:
            self.plot_turbine_locations(axs)

        if axs is None:
            self.save_fig(fig, os.path.join(
                self.fig_dir, 'elevation.png'), show)

        return fig, axs

    def plot_terrain_slope(self, plot_turbs=True, show=False, plot='pcolormesh',
                           fig=None, axs=None, figsize=None) -> None:
        """ Plots slope in degrees """
        if figsize is None:
            figsize = self.fig_size

        slope = self.get_terrain_slope()
        if axs is None:
            fig, axs = plt.subplots(figsize=figsize)

        if plot == 'pcolormesh':
            curm = axs.pcolormesh(
                self.xx_terrain, self.yy_terrain, slope.T, cmap='magma_r', rasterized=True)
        else:
            curm = axs.imshow(slope, cmap='magma_r',
                              extent=self.extent, origin='lower')

        cbar, _ = create_gis_axis(fig, axs, curm, self.km_bar)
        cbar.set_label('Slope (deg)')
        if plot_turbs:
            self.plot_turbine_locations(axs)

        if axs is None:
            self.save_fig(fig, os.path.join(self.fig_dir, 'slope.png'), show)

        return fig, axs

    def plot_terrain_aspect(self, plot_turbs=True, show=False, plot='pcolormesh', cmap='twilight', figsize=None) -> None:
        """ Plots terrain aspect """
        if figsize is None:
            figsize = self.fig_size
        aspect = self.get_terrain_aspect()
        fig, axs = plt.subplots(figsize=figsize)
        if plot == 'pcolormesh':
            curm = axs.pcolormesh(self.xx_terrain, self.yy_terrain,
                                  aspect.T, cmap=cmap, vmin=0, vmax=360, rasterized=True)
        else:
            curm = axs.imshow(aspect, cmap=cmap,
                              extent=self.extent, origin='lower', vmin=0, vmax=360.)
        cbar, _ = create_gis_axis(fig, axs, curm, self.km_bar)
        cbar.ax.set_yticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
        cbar.ax.set_yticklabels(
            ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'])
        cbar.set_label('Aspect (Direction)')
        if plot_turbs:
            self.plot_turbine_locations(axs)
        self.save_fig(fig, os.path.join(self.fig_dir, 'aspect.png'), show)

    def plot_terrain_sx(self, plot_turbs=True, show=False, plot='imshow', cmap=None, figsize=None, **kwargs) -> None:
        """ Plots sheltering angle in degrees """
        if cmap is None:
            # Custom scale for Sx (diverging cmap,light red-black-white)
            gray = cm.get_cmap('binary_r', 128)
            red = cm.get_cmap('Reds', 128)
            mydiv = np.vstack((red(np.linspace(0, 1, 128)),
                              gray(np.linspace(0, 1, 128))))
            cmap = ListedColormap(mydiv, name='RdBl')

        if figsize is None:
            figsize = self.fig_size
        sx = self.get_terrain_sx()
        fig, axs = plt.subplots(figsize=figsize)
        if plot == 'pcolormesh':
            curm = axs.pcolormesh(
                self.xx_terrain, self.yy_terrain, sx.T, cmap=cmap, rasterized=True, **kwargs)
        else:
            curm = axs.imshow(
                sx, cmap=cmap, extent=self.extent, origin='lower')
        cbar, _ = create_gis_axis(fig, axs, curm, self.km_bar)
        cbar.set_label('Sx (Degrees)')
        if plot_turbs:
            self.plot_turbine_locations(axs)
        self.save_fig(fig, os.path.join(
            self.fig_dir, f'sx_{int(self.uniform_winddirn)}m.png'), show)


########### Plotting atmospheric conditions from HRRR ###########


    def plot_atmospheric_conditions_HRRR(self, show=False) -> None:
        """ Plots a grid of several atmospheric variables for the site of interest """

        if self.wind_data_source != 'hrrr':
            print(
                'Skipping plotting atmopsheric conditions due to wind source data option selected')
            return

        from matplotlib import colors

        print(
            f'Saving images of atmospheric conditions for: {self.case_ids}, at {self.dtimes}')

        g = 9.81    # m/s^2
        rho = 1.225  # kg/m^3
        cp = 1005   # J/(kg*K)

        for case_id in self.case_ids:
            for time in self.dtimes:

                # Get string of time to pass to HRRR. Time can be passed either a tuple of datetime object
                if isinstance(time, datetime):
                    timestr = f' {time.year}-{time.month:02d}-{time.day:02d} {time.hour:02d}:{time.minute:02d}'
                else:
                    timestr = f'{time[0]}-{time[1]:02d}-{time[2]:02d} {time[3]:02d}:00'

                # Get hrrr data
                hrrr = HRRR(valid_date=timestr)

                SWlonlat = self.southwest_lonlat
                # Extent is given from 0 to the dimensions of the terrain
                extent = [0, 0, self.region_width_km[0] * 1000,
                          self.region_width_km[1] * 1000]  # xmin, ymin, xmax, ymax
                res = self.resolution

                # Compute wstar
                wstar, xx, yy = hrrr.get_convective_velocity(
                    SWlonlat, extent, res)

                # Compute albedo
                albedo, xx, yy = hrrr.get_albedo(SWlonlat, extent, res)

                # Get quantities
                gflux_Wm2, xx, yy = hrrr.get_single_var_on_grid(
                    ':(GFLUX):', SWlonlat, extent, res)   # ground heat flux
                pot, xx, yy = hrrr.get_single_var_on_grid(
                    ':(POT):', SWlonlat, extent, res)   # potential temp at 2m
                zi, xx, yy = hrrr.get_single_var_on_grid(
                    ':(HPBL):', SWlonlat, extent, res)   # boundary layer height
                cloud, xx, yy = hrrr.get_single_var_on_grid(
                    ':(LCDC):', SWlonlat, extent, res)   # cloud coverage
                Lu, xx, yy = hrrr.get_single_var_on_grid(
                    ':ULWRF:surface', SWlonlat, extent, res)   # long wave upward
                Ld, xx, yy = hrrr.get_single_var_on_grid(
                    ':DLWRF:surface', SWlonlat, extent, res)   # long wave downward
                Su, xx, yy = hrrr.get_single_var_on_grid(
                    ':USWRF:surface', SWlonlat, extent, res)   # short wave upward
                Sd, xx, yy = hrrr.get_single_var_on_grid(
                    ':DSWRF:surface', SWlonlat, extent, res)   # short wave downward
                sensible, xx, yy = hrrr.get_single_var_on_grid(
                    ':SHTFL:surface', SWlonlat, extent, res)   # sensible heat flux
                latent, xx, yy = hrrr.get_single_var_on_grid(
                    ':LHTFL:surface', SWlonlat, extent, res)   # latent heat flux
                moisture, xx, yy = hrrr.get_single_var_on_grid(
                    ':MSTAV:', SWlonlat, extent, res)   # moisture
                groundT, xx, yy = hrrr.get_single_var_on_grid(
                    ':TMP:surface', SWlonlat, extent, res)   # ground temp
                snow, xx, yy = hrrr.get_single_var_on_grid(
                    ':SNOWC', SWlonlat, extent, res)   # snow cover

                gflux_Wm2 = gflux_Wm2[list(gflux_Wm2.keys())[0]]
                pot = pot[list(pot      .keys())[0]]
                zi = zi[list(zi       .keys())[0]]
                cloud = cloud[list(cloud    .keys())[0]]
                Lu = Lu[list(Lu       .keys())[0]]
                Ld = Ld[list(Ld       .keys())[0]]
                Su = Su[list(Su       .keys())[0]]
                Sd = Sd[list(Sd       .keys())[0]]
                sensible = sensible[list(sensible .keys())[0]]
                latent = latent[list(latent   .keys())[0]]
                moisture = moisture[list(moisture .keys())[0]]
                groundT = groundT[list(groundT  .keys())[0]]
                snow = snow[list(snow     .keys())[0]]

                # Compute quantities
                # gain of energy by the surface, positive when it is towards the surface
                Lnet = Ld - Lu
                net_rad = Sd - Su + Ld - Lu  # Rn1
                qs = sensible + latent - gflux_Wm2  # minus for sign convention

                # Create plot
                fig, axs = plt.subplots(ncols=4, nrows=4, figsize=(17, 15))
                axs = axs.flatten()

                cmap = 'RdGy_r'
                vmin = -500
                vmax = 500
                norm = colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)

                elevation = highRes2lowRes(
                    self.get_terrain_elevation(), self.resolution_terrain, self.resolution)
                cm = axs[0].pcolormesh(elevation / 1000, cmap='terrain')
                fig.colorbar(cm, ax=axs[0], label='terrain elevation [m]')

                slope = highRes2lowRes(self.get_terrain_slope(
                ), self.resolution_terrain, self.resolution)
                cm = axs[1].pcolormesh(slope, cmap='magma_r')
                fig.colorbar(cm, ax=axs[1], label='terrain slope [deg]')

                aspect = highRes2lowRes(self.get_terrain_aspect(
                ), self.resolution_terrain, self.resolution)
                cm = axs[2].pcolormesh(
                    aspect, cmap='twilight', vmin=0, vmax=360)
                cbar = fig.colorbar(
                    cm, ax=axs[2], label='terrain aspect [direction]')
                cbar.ax.set_yticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
                cbar.ax.set_yticklabels(
                    ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'])

                cm = axs[3].axis('off')

                cm = axs[4].contourf(xx, yy, albedo, cmap='plasma')
                fig.colorbar(cm, ax=axs[4], label='surface albedo [-]')

                cm = axs[5].contourf(xx, yy, net_rad, cmap=cmap, norm=norm)
                fig.colorbar(
                    cm, ax=axs[5], label='net radiation from long/short wave [W/m2]')

                cm = axs[6].contourf(xx, yy, qs, cmap=cmap, norm=norm)
                fig.colorbar(cm, ax=axs[6], label='heat budget [W/m2]')

                vminmaxgroundT = np.round(
                    max(np.min(groundT), np.max(groundT))) - 273
                vminmaxgroundT = np.ceil(vminmaxgroundT / 5) * 5
                cm = axs[7].contourf(xx, yy, groundT - 273, cmap='RdBu_r',
                                     levels=np.arange(-vminmaxgroundT, vminmaxgroundT + 1, 5))
                fig.colorbar(cm, ax=axs[7], label='ground temp [deg C]')

                cm = axs[8].contourf(xx, yy, sensible, cmap='viridis')
                fig.colorbar(cm, ax=axs[8], label='sensible heat [W/m2]')

                cm = axs[9].contourf(xx, yy, latent, cmap='viridis')
                fig.colorbar(cm, ax=axs[9], label='latent heat [W/m2]')

                vminmaxglux = np.round(
                    max(-np.min(gflux_Wm2), np.max(gflux_Wm2)))
                cm = axs[10].contourf(
                    xx, yy, gflux_Wm2, cmap=cmap, vmin=-vminmaxglux, vmax=vminmaxglux)
                fig.colorbar(cm, ax=axs[10], label='gound heat flux [W/m2]')

                cm = axs[11].contourf(
                    xx, yy, moisture, cmap='magma', levels=np.arange(0, 100.1, 10))
                fig.colorbar(cm, ax=axs[11],
                             label='soil moisture availability [%]')

                cm = axs[12].contourf(
                    xx, yy, zi, cmap='gist_ncar', levels=np.arange(0, 3500.1, 100))
                fig.colorbar(cm, ax=axs[12], label='boundary layer height [m]')

                cm = axs[13].contourf(
                    xx, yy, cloud, cmap='Blues_r', levels=np.arange(0, 100.1, 10))
                fig.colorbar(cm, ax=axs[13], label='low cloud cover [%]')

                vmaxwstar = np.ceil(np.mean(wstar))
                # , levels=np.arange(0,1.2*vmaxwstar, vmaxwstar/10))
                cm = axs[14].contourf(xx, yy, wstar, cmap='cividis')
                fig.colorbar(cm, ax=axs[14], label='convective velocity [m/s]')

                cm = axs[15].contourf(
                    xx, yy, snow, cmap='Greens_r', levels=np.arange(0, 100.1, 10))
                fig.colorbar(cm, ax=axs[15], label='snow cover [%]')

                fig.suptitle(f'Conditions at {timestr} UTC', fontsize=14)
                for ax in axs:
                    ax.set_aspect('equal')
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                fig.tight_layout()

                fname = f'{case_id}_atmospheric_quantities.png'
                self.save_fig(fig, os.path.join(self.fig_dir, fname), show)


########### other useful functions ###########


    def plot_turbine_locations(
            self,
            axs,
            set_label: bool = True,
            draw_box: bool = False
    ):
        """ Plot turbine locations on a given axis"""
        if self.turbines.dframe is not None:
            for i, pname in enumerate(self.turbines.get_project_names()):
                mrkr = self.turbine_mrkr_styles[i %
                                                len(self.turbine_mrkr_styles)]
                xlocs, ylocs = self.turbines.get_locations_for_this_project(
                    pname)
                axs.plot(xlocs, ylocs, mrkr, markersize=self.turbine_mrkr_size,
                         alpha=0.75, label=pname if set_label else "")
                if draw_box:
                    width = max(xlocs) - min(ylocs) + 2
                    height = max(ylocs) - min(ylocs) + 2
                    rect = mpatches.Rectangle((min(xlocs) - 1, min(ylocs) - 1),
                                              width, height,
                                              linewidth=1, edgecolor='k',
                                              facecolor='none', zorder=20)
                    axs.add_patch(rect)

    def get_wtk_locs(self):
        """ Returns xlocs and ylocs of wtk data points """
        wtk_lons, wtk_lats = self.wtk.get_coordinates()
        wtk_xlocs, wtk_ylocs = transform_coordinates(
            lonlat_crs, self.projected_crs, wtk_lons, wtk_lats)
        return wtk_xlocs, wtk_ylocs

    def get_seasonal_datetimes(self) -> List[datetime]:
        """ Determine the datetimes for importing seasonal data from WTK """
        print(f'Seasonal: Requested {self.seasonal_count} counts')
        print(f'Seasonal: Starting Month,Day is {self.seasonal_start}')
        print(f'Seasonal: Ending Month,Day is {self.seasonal_end}')
        print(f'Seasonal: Time of day is {self.seasonal_timeofday}')
        random_datetimes = set()
        i = 0
        while i < self.seasonal_count:
            rnd_year = random.choice(self.wtk.years)
            start_date = datetime(rnd_year, *self.seasonal_start)
            end_date = datetime(rnd_year, *self.seasonal_end)
            rnd_date = start_date + random.random() * (end_date - start_date)
            rnd_date = rnd_date.replace(microsecond=0, second=0, minute=0)
            lonlat = self.lonlat_bounds[0:2]
            srise, sset = get_sunrise_sunset_time(lonlat, rnd_date)
            daytime_hours = np.array(range(srise.hour + 1, sset.hour + 1))
            split_hours = np.array_split(daytime_hours, 3)
            if self.seasonal_timeofday.lower() == 'morning':
                chosen_hours = list(split_hours[0])
            elif self.seasonal_timeofday.lower() == 'afternoon':
                chosen_hours = list(split_hours[1])
            elif self.seasonal_timeofday.lower() == 'evening':
                chosen_hours = list(split_hours[2])
            elif self.seasonal_timeofday.lower() == 'daytime':
                chosen_hours = list(daytime_hours)
            else:
                raise ValueError(
                    (f'Invalid time of day:{self.seasonal_timeofday}'
                     '\nOptions: morning, afternoon, evening, daytime'))
            rnd_date = rnd_date.replace(hour=random.choice(chosen_hours))
            if rnd_date not in random_datetimes:
                random_datetimes.add(rnd_date)
                i += 1
        return list(random_datetimes)

    def save_fig(self, fig, fpath: str, show_fig: bool = False):
        """ Saves a fig """
        if not show_fig:
            fig.savefig(fpath, bbox_inches='tight', dpi=self.fig_dpi)
            plt.close(fig)

    def _get_orographicupdraft_fpath(self, case_id: str):
        """ Returns file path for saving orographic updrafts data """
        return os.path.join(self.mode_data_dir, f'{case_id}_orograpicupdraft_{self.orographic_model}Model.npy')

    def _get_thermal_fpath(self, case_id: str):
        """ Returns file path for saving thermal updrafts data """
        return os.path.join(self.mode_data_dir, f'{case_id}_thermal.npy')

    def _get_potential_fpath(self, case_id: str):
        """ Returns file path for saving directional potential data"""
        fname = f'{case_id}_{int(self.track_direction)}_potential.npy'
        return os.path.join(self.mode_data_dir, fname)

    def _get_tracks_fpath(self, case_id: str):
        """ Returns file path for saving simulated tracks """
        fname = f'{case_id}_{int(self.track_direction)}_tracks.pkl'
        return os.path.join(self.mode_data_dir, fname)

    def _get_uniform_id(self):
        """ Returns case id for uniform mode """
        return (f's{int(self.uniform_windspeed_h)}'
                f'd{int(self.uniform_winddirn_h)}'
                f'thInt{int(self.thermal_intensity_scale)}')

    def _interpolate_wtk_vardata(
        self,
        vdata: np.ndarray
    ) -> np.ndarray:
        """ Interpolates wtk data (unstructured) to terrain (structured) grid"""
        xgrid, ygrid = self.get_terrain_grid()
        wtk_xlocs, wtk_ylocs = self.get_wtk_locs()
        points = np.array([wtk_xlocs, wtk_ylocs]).T
        xmesh, ymesh = np.meshgrid(xgrid, ygrid)
        interp_data = griddata(points, vdata, (xmesh, ymesh),
                               method=self.wtk_interp_type)
        return interp_data

    def _get_interpolated_wind_conditions(
        self,
        wspeed: np.ndarray,
        wdirn: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ Interpolates wind speed and direction from wtk to terrain grid """
        ang = np.radians(270. - wdirn)
        ueast = np.multiply(wspeed, np.cos(ang))
        vnorth = np.multiply(wspeed, np.sin(ang))
        interp_ueast = self._interpolate_wtk_vardata(ueast)
        interp_vnorth = self._interpolate_wtk_vardata(vnorth)
        interp_wspeed = np.sqrt(interp_ueast**2 + interp_vnorth**2)
        interp_wdirn = 180. + np.degrees(np.arctan2(interp_ueast,
                                                    interp_vnorth))
        return interp_wspeed, interp_wdirn

    def plot_updraft_threshold_function(self, show=False):
        """Plots the threshold function """
        fig, axs = plt.subplots(figsize=(5, 3))
        uspeed = np.linspace(0, np.ceil(self.updraft_threshold) + 1, 100)
        axs.plot(uspeed, get_above_threshold_speed(
            uspeed, self.updraft_threshold))
        axs.grid(True)
        axs.set_xlabel('Updraft speed (m/s)')
        axs.set_ylabel('Threshold function')
        fname = 'threshold_function.png'
        self.save_fig(fig, os.path.join(self.fig_dir, fname), show)
