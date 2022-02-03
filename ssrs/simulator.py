""" Module for setting up SSRS """

import os
import json
import time
import pickle
import random
from typing import List, Tuple
from datetime import datetime
import pathos.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from contextlib import redirect_stdout
from scipy.interpolate import griddata
from matplotlib.colors import LogNorm
from dataclasses import asdict
from .terrain import Terrain
from .wtk import WTK
from .turbines import TurbinesUSWTB
from .config import Config
from .layers import (compute_orographic_updraft, compute_aspect_degrees,
                     compute_slope_degrees)
from .raster import (get_raster_in_projected_crs,
                     transform_bounds, transform_coordinates)
from .movmodel import (MovModel, get_starting_indices, generate_eagle_track,
                       generate_heuristic_eagle_track,
                       compute_smooth_presence_counts)
from .heuristics import rulesets
from .utils import (makedir_if_not_exists, get_elapsed_time,
                    get_extent_from_bounds, empty_this_directory,
                    create_gis_axis, get_sunrise_sunset_time)


class Simulator(Config):
    """ Class for SSRS simulation """

    lonlat_crs = 'EPSG:4326'
    time_format = 'y%Ym%md%dh%H'

    def __init__(self, in_config: Config = None, **kwargs) -> None:
        # initiate the config parameters
        if in_config is None:
            super().__init__(**kwargs)
        else:
            super().__init__(**asdict(in_config))
        print(f'\n---- SSRS in {self.sim_mode} mode')
        print(f'---- movements based on {self.sim_movement} model')
        print(f'Run name: {self.run_name}')

        # re-init random number generator for results reproducibility
        if self.sim_seed >= 0:
            print('Specified random number seed:',self.sim_seed)
            np.random.seed(self.sim_seed)

        # create directories for saving data and figures
        print(f'Output dir: {os.path.join(self.out_dir, self.run_name)}')
        self.data_dir = os.path.join(self.out_dir, self.run_name, 'data/')
        self.fig_dir = os.path.join(self.out_dir, self.run_name, 'figs/')
        self.mode_data_dir = os.path.join(self.data_dir, self.sim_mode)
        self.mode_fig_dir = os.path.join(self.fig_dir, self.sim_mode)
        for dirname in (self.mode_data_dir, self.mode_fig_dir):
            makedir_if_not_exists(dirname)

        # save the config file
        fpath = os.path.join(self.out_dir, self.run_name,
                             f'{self.run_name}.json')
        with open(fpath, 'w', encoding='utf-8') as cfile:
            json.dump(self.__dict__, cfile, ensure_ascii=False, indent=2)

        # Determine grid size of the region
        xsize = int(round((self.region_width_km[0] * 1000. / self.resolution)))
        ysize = int(round((self.region_width_km[1] * 1000. / self.resolution)))
        self.gridsize = (ysize, xsize)

        # figure out bounds in both lon/lat and in projected crs
        proj_west, proj_south = transform_coordinates(
            self.lonlat_crs, self.projected_crs,
            self.southwest_lonlat[0], self.southwest_lonlat[1])
        proj_east = proj_west[0] + xsize * self.resolution
        proj_north = proj_south[0] + ysize * self.resolution
        self.bounds = (proj_west[0], proj_south[0], proj_east, proj_north)
        self.extent = get_extent_from_bounds(self.bounds)
        self.lonlat_bounds = transform_bounds(
            self.bounds, self.projected_crs, self.lonlat_crs)

        # download terrain layers from USGS's 3DEP dataset
        self.region = Terrain(self.lonlat_bounds, self.data_dir)
        try:
            self.terrain_layers = {
                'Elevation': 'DEM',
                'Slope': 'Slope Degrees',
                'Aspect': 'Aspect Degrees'
            }
            self.region.download(self.terrain_layers.values())
        except Exception as _:
            print('Connection issues with 3DEP WMS service! Trying SRTM1..')
            self.terrain_layers = {'Elevation': 'SRTM1'}
            self.region.download(self.terrain_layers.values())

        # setup turbine data
        self.turbines = TurbinesUSWTB(self.bounds, self.projected_crs,
                                      self.turbine_minimum_hubheight,
                                      self.data_dir)
        fname = os.path.join(self.data_dir, 'turbines_summary.txt')
        with open(fname, 'w') as f:
            with redirect_stdout(f):
                self.turbines.print_details()

        # figure out wtk and its layers to extract
        self.wtk_layers = {
            'wspeed': f'windspeed_{str(int(self.wtk_orographic_height))}m',
            'wdirn': f'winddirection_{str(int(self.wtk_orographic_height))}m',
            'pressure': f'pressure_{str(int(self.wtk_thermal_height))}m',
            'temperature': f'temperature_{str(int(self.wtk_thermal_height))}m',
            'blheight': 'boundary_layer_height',
            'surfheatflux': 'surface_heat_flux'
        }

        # Compute orographic updrafts
        if self.sim_mode.lower() != 'uniform':
            self.wtk = WTK(self.wtk_source, self.lonlat_bounds,
                           self.wtk_layers.values(), self.mode_data_dir)
            if self.sim_mode.lower() == 'seasonal':
                empty_this_directory(self.mode_data_dir)
                self.dtimes = self.get_seasonal_datetimes()
            elif self.sim_mode.lower() == 'snapshot':
                self.dtimes = [datetime(*self.snapshot_datetime)]
            self.wtk.download_data(self.dtimes, self.max_cores)
            self.case_ids = [dt.strftime(self.time_format)
                             for dt in self.dtimes]
            self.compute_orographic_updrafts_using_wtk()
        else:
            self.case_ids = [self._get_uniform_id()]
            self.compute_orographic_updraft_uniform()

        # plotting settings
        fig_aspect = self.region_width_km[0] / self.region_width_km[1]
        self.fig_size = (self.fig_height * fig_aspect, self.fig_height)
        self.km_bar = min([1, 5, 10], key=lambda x: abs(
            x - self.region_width_km[0] // 4))

    def simulate_tracks(self):
        """ Simulate tracks """
        if self.sim_movement == 'fluid-analogy':
            self.compute_directional_potential()
        elif self.sim_movement == 'heuristics':
            if self.movement_ruleset not in rulesets.keys():
                raise ValueError(f'{self.movement_ruleset} is not defined.  Valid rulesets: {rulesets.keys()}')
        # print('Getting starting locations for simulating eagle tracks..')
        starting_rows, starting_cols = get_starting_indices(
            self.track_count,
            self.track_start_region,
            self.track_start_type,
            self.region_width_km,
            self.resolution
        )
        starting_locs = [[x, y] for x, y in zip(starting_rows, starting_cols)]
        num_cores = min(self.track_count, self.max_cores)
        for case_id in self.case_ids:
            tmp_str = f'{case_id}_{int(self.track_direction)}'
            print(f'{tmp_str}: Simulating {self.track_count} tracks..',
                  end="", flush=True)
            orograph = np.load(self._get_orograph_fpath(case_id))
            if self.sim_movement == 'fluid-analogy':
                potential = np.load(self._get_potential_fpath(case_id))
            start_time = time.time()
            if self.sim_movement == 'fluid-analogy':
                with mp.Pool(num_cores) as pool:
                    tracks = pool.map(lambda start_loc: generate_eagle_track(
                        orograph,
                        potential,
                        start_loc,
                        self.track_dirn_restrict,
                        self.track_stochastic_nu
                    ), starting_locs)
            elif self.sim_movement == 'heuristics': 
                with mp.Pool(num_cores) as pool:
                    tracks = pool.map(lambda start_loc: generate_heuristic_eagle_track(
                        self.movement_ruleset,
                        orograph,
                        start_loc,
                        self.track_direction,
                        self.resolution
                    ), starting_locs)
            print(f'took {get_elapsed_time(start_time)}', flush=True)
            with open(self._get_tracks_fpath(case_id), "wb") as fobj:
                pickle.dump(tracks, fobj)

    def compute_directional_potential(self):
        """ Computes the mogration potential by solving sparse linear system"""
        mov_model = MovModel(self.track_direction, self.gridsize)
        bndry_nodes, bndry_energy = mov_model.get_boundary_nodes()
        row_inds, col_inds, facs = mov_model.assemble_sparse_linear_system()
        for case_id in self.case_ids:
            fpath = self._get_potential_fpath(case_id)
            tmp_str = f'{case_id}_{int(self.track_direction)}'
            try:
                potential = np.load(fpath)
                if potential.shape != self.gridsize:
                    raise FileNotFoundError
                print(f'{tmp_str}: Found saved potential')
            except FileNotFoundError as _:
                start_time = time.time()
                print(f'{tmp_str}: Computing potential..', end="", flush=True)
                orograph = np.load(self._get_orograph_fpath(case_id))
                potential = mov_model.solve_sparse_linear_system(
                    orograph,
                    bndry_nodes,
                    bndry_energy,
                    row_inds,
                    col_inds,
                    facs
                )
                print(f'took {get_elapsed_time(start_time)}', flush=True)
                np.save(fpath, potential.astype(np.float32))

    def compute_orographic_updraft_uniform(self) -> None:
        """ Computing orographic updrafts for uniform mode"""
        print('Computing orographic updrafts..')
        slope = self.get_terrain_slope()
        aspect = self.get_terrain_aspect()
        wspeed = self.uniform_windspeed * np.ones(self.gridsize)
        wdirn = self.uniform_winddirn * np.ones(self.gridsize)
        orograph = compute_orographic_updraft(wspeed, wdirn, slope, aspect)
        fpath = self._get_orograph_fpath(self.case_ids[0])
        np.save(fpath, orograph.astype(np.float32))

    def compute_orographic_updrafts_using_wtk(self) -> None:
        """ Computing orographic updrafts using wtk data for all datetimes"""
        print('Computing orographic updrafts..', end="")
        slope = self.get_terrain_slope()
        aspect = self.get_terrain_aspect()
        start_time = time.time()
        for dtime, case_id in zip(self.dtimes, self.case_ids):
            wtk_df = self.wtk.get_dataframe_for_this_time(dtime)
            wspeed, wdirn = self._get_interpolated_wind_conditions(
                wtk_df[self.wtk_layers['wspeed']],
                wtk_df[self.wtk_layers['wdirn']]
            )
            orograph = compute_orographic_updraft(wspeed, wdirn, slope, aspect)
            fpath = self._get_orograph_fpath(case_id)
            np.save(fpath, orograph.astype(np.float32))
        print(f'took {get_elapsed_time(start_time)}', flush=True)

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
        curm = axs.imshow(elevation / 1000., cmap='terrain',
                          extent=self.extent, origin='lower')
        cbar, _ = create_gis_axis(fig, axs, curm, self.km_bar)
        cbar.set_label('Altitude (km)')
        if plot_turbs:
            self.plot_turbine_locations(axs)
        self.save_fig(fig, os.path.join(self.fig_dir, 'elevation.png'), show)

    def plot_terrain_slope(self, plot_turbs=True, show=False) -> None:
        """ Plots slope in degrees """
        slope = self.get_terrain_slope()
        fig, axs = plt.subplots(figsize=self.fig_size)
        curm = axs.imshow(slope, cmap='magma_r',
                          extent=self.extent, origin='lower')
        cbar, _ = create_gis_axis(fig, axs, curm, self.km_bar)
        cbar.set_label('Slope (Degrees)')
        if plot_turbs:
            self.plot_turbine_locations(axs)
        self.save_fig(fig, os.path.join(self.fig_dir, 'slope.png'), show)

    def plot_terrain_aspect(self, plot_turbs=True, show=False) -> None:
        """ Plots terrain aspect """
        aspect = self.get_terrain_aspect()
        fig, axs = plt.subplots(figsize=self.fig_size)
        curm = axs.imshow(aspect, cmap='hsv',
                          extent=self.extent, origin='lower', vmin=0, vmax=360.)
        cbar, _ = create_gis_axis(fig, axs, curm, self.km_bar)
        cbar.set_label('Aspect (Degrees)')
        if plot_turbs:
            self.plot_turbine_locations(axs)
        self.save_fig(fig, os.path.join(self.fig_dir, 'aspect.png'), show)

    def plot_simulation_output(self, plot_turbs=True, show=False) -> None:
        """ Plots oro updraft and tracks """
        self.plot_orographic_updrafts(plot_turbs, show)
        #self.plot_directional_potentials(plot_turbs, show)
        self.plot_simulated_tracks(plot_turbs, show)
        self.plot_presence_map(plot_turbs, show)

    def plot_orographic_updrafts(self, plot_turbs=True, show=False) -> None:
        """ Plot orographic updrafts """
        for case_id in self.case_ids:
            orograph = np.load(self._get_orograph_fpath(case_id))
            fig, axs = plt.subplots(figsize=self.fig_size)
            maxval = min(max(1, int(round(np.mean(orograph)))), 5)
            curm = axs.imshow(orograph, cmap='viridis',
                              extent=self.extent, origin='lower',
                              vmin=0, vmax=maxval)
            cbar, _ = create_gis_axis(fig, axs, curm, self.km_bar)
            cbar.set_label('Orographic updraft (m/s)')
            if plot_turbs:
                self.plot_turbine_locations(axs)
            fname = os.path.join(self.mode_fig_dir, f'{case_id}_orograph.png')
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
                    # cmap = 'hsv' if 'direction' in wtk_lyr else 'viridis'
                    curm = axs.imshow(interp_data, cmap='viridis',
                                      origin='lower', extent=self.extent,
                                      alpha=0.75)
                    cbar, _ = create_gis_axis(fig, axs, curm, self.km_bar)
                    cbar.set_label(wtk_lyr)
                    axs.set_xlim([self.extent[0], self.extent[1]])
                    axs.set_ylim([self.extent[2], self.extent[3]])
                    if plot_turbs:
                        self.plot_turbine_locations(axs)
                    fname = f'{case_id}_{wtk_lyr}.png'
                    self.save_fig(fig, os.path.join(self.mode_fig_dir, fname),
                                  show)
        except AttributeError as _:
            print('No WTK data to plot in uniform mode!')

    def plot_directional_potentials(self, plot_turbs=True, show=False) -> None:
        """ Plot directional potential """
        print('Plotting directional potential..')
        for case_id in self.case_ids:
            potential = np.load(self._get_potential_fpath(case_id))
            fig, axs = plt.subplots(figsize=self.fig_size)
            lvls = np.linspace(0, np.amax(potential), 11)
            curm = axs.contourf(potential, lvls, cmap='cividis', origin='lower',
                                extent=self.extent)
            cbar, _ = create_gis_axis(fig, axs, curm, self.km_bar)
            cbar.set_label('Directional potential')
            if plot_turbs:
                self.plot_turbine_locations(axs)
            axs.set_xlim([self.extent[0], self.extent[1]])
            axs.set_ylim([self.extent[2], self.extent[3]])
            fname = f'{case_id}_{int(self.track_direction)}_potential.png'
            self.save_fig(fig, os.path.join(self.mode_fig_dir, fname), show)

    def plot_simulated_tracks(self, plot_turbs=True, show=False) -> None:
        """ Plots simulated tracks """
        print('Plotting simulated tracks..')
        lwidth = 0.1 if self.track_count > 251 else 0.4
        elevation = self.get_terrain_elevation()
        xgrid, ygrid = self.get_terrain_grid()
        for case_id in self.case_ids:
            fig, axs = plt.subplots(figsize=self.fig_size)
            _ = axs.imshow(elevation, alpha=0.75, cmap='Greys',
                           origin='lower', extent=self.extent)
            with open(self._get_tracks_fpath(case_id), 'rb') as fobj:
                tracks = pickle.load(fobj)
                for itrack in tracks:
                    axs.plot(xgrid[itrack[0, 1]], ygrid[itrack[0, 0]], 'b.',
                             markersize=1.0)
                    axs.plot(xgrid[itrack[:, 1]], ygrid[itrack[:, 0]],
                             '-r', linewidth=lwidth, alpha=0.5)
            _, _ = create_gis_axis(fig, axs, None, self.km_bar)
            if plot_turbs:
                self.plot_turbine_locations(axs)
            left = self.extent[0] + self.track_start_region[0] * 1000.
            bottom = self.extent[2] + self.track_start_region[2] * 1000.
            width = self.track_start_region[1] - self.track_start_region[0]
            hght = self.track_start_region[3] - self.track_start_region[2]
            rect = mpatches.Rectangle((left, bottom), width * 1000.,
                                      hght * 1000., alpha=0.2,
                                      edgecolor='none', facecolor='b')
            axs.add_patch(rect)
            axs.set_xlim([self.extent[0], self.extent[1]])
            axs.set_ylim([self.extent[2], self.extent[3]])
            fname = f'{case_id}_{int(self.track_direction)}_tracks.png'
            self.save_fig(fig, os.path.join(self.mode_fig_dir, fname), show)

    def plot_presence_map(self, plot_turbs=True, show=False,
                          minval=0.25) -> None:
        """ Plot presence maps """
        print('Plotting presence density map..')
        # elevation = self.get_terrain_elevation()
        for case_id in self.case_ids:
            with open(self._get_tracks_fpath(case_id), 'rb') as fobj:
                tracks = pickle.load(fobj)
            prprob = compute_smooth_presence_counts(
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
            axs.set_xlim([self.extent[0], self.extent[1]])
            axs.set_ylim([self.extent[2], self.extent[3]])
            fname = f'{case_id}_{int(self.track_direction)}_presence.png'
            self.save_fig(fig, os.path.join(self.mode_fig_dir, fname), show)

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

    def plot_plant_specific_presence_maps(self, show=False,
                                          minval=0.2) -> None:
        """ Plot presence maps for each power plant contained in study area"""
        print('Plotting presence map for each project..')
        smooting_radius = int(self.presence_smoothing_radius / 2)
        pad = 2000.  # in meters
        for case_id in self.case_ids:
            with open(self._get_tracks_fpath(case_id), 'rb') as fobj:
                tracks = pickle.load(fobj)
            prprob = compute_smooth_presence_counts(
                tracks, self.gridsize, smooting_radius)
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

    def plot_turbine_locations(
            self,
            axs,
            set_label: bool = True,
            draw_box: bool = False
    ):
        """ Plot turbine locations on a given axis"""
        for i, pname in enumerate(self.turbines.get_project_names()):
            mrkr = self.turbine_mrkr_styles[i % len(self.turbine_mrkr_styles)]
            xlocs, ylocs = self.turbines.get_locations_for_this_project(pname)
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

    def get_terrain_grid(self):
        """ Returns xgrid and ygrid for the terrain """
        xgrid = np.linspace(self.bounds[0], self.bounds[0] + self.gridsize[1] *
                            self.resolution, self.gridsize[1])
        ygrid = np.linspace(self.bounds[1], self.bounds[1] + self.gridsize[0] *
                            self.resolution, self.gridsize[0])
        return xgrid, ygrid

    def get_wtk_locs(self):
        """ Returns xlocs and ylocs of wtk data points """
        wtk_lons, wtk_lats = self.wtk.get_coordinates()
        wtk_xlocs, wtk_ylocs = transform_coordinates(
            self.lonlat_crs, self.projected_crs, wtk_lons, wtk_lats)
        return wtk_xlocs, wtk_ylocs

    def get_terrain_elevation(self):
        """ Returns data for terrain layer inprojected crs """
        return self.get_terrain_layer('Elevation')

    def get_terrain_slope(self):
        """ Returns data for terrain layer inprojected crs """
        try:
            slope = self.get_terrain_layer('Slope')
        except:
            elev = self.get_terrain_elevation()
            slope = compute_slope_degrees(elev, self.resolution)
        return slope

    def get_terrain_aspect(self):
        """ Returns data for terrain layer inprojected crs """
        try:
            aspect = self.get_terrain_layer('Aspect')
        except:
            elev = self.get_terrain_elevation()
            aspect = compute_aspect_degrees(elev, self.resolution)
        return aspect

    def get_terrain_layer(self, lname: str):
        """ Returns data for terrain layer inprojected crs """
        ldata = get_raster_in_projected_crs(
            self.region.get_raster_fpath(self.terrain_layers[lname]),
            self.bounds, self.gridsize, self.resolution, self.projected_crs)
        return ldata

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
        fig.savefig(fpath, bbox_inches='tight', dpi=self.fig_dpi)
        if not show_fig:
            plt.close(fig)

    def _get_orograph_fpath(self, case_id: str):
        """ Returns file path for saving orographic updrafts data """
        return os.path.join(self.mode_data_dir, f'{case_id}_orograph.npy')

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
        return (f's{int(self.uniform_windspeed)}'
                f'd{int(self.uniform_winddirn)}')

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
        easterly = np.multiply(wspeed, np.sin(wdirn * np.pi / 180.))
        northerly = np.multiply(wspeed, np.cos(wdirn * np.pi / 180.))
        interp_easterly = self._interpolate_wtk_vardata(easterly)
        interp_northerly = self._interpolate_wtk_vardata(northerly)
        interp_wspeed = np.sqrt(np.square(interp_easterly) +
                                np.square(interp_northerly))
        interp_wdirn = np.arctan2(interp_easterly, interp_northerly)
        interp_wdirn = np.mod(interp_wdirn + 2. * np.pi, 2. * np.pi)
        return interp_wspeed, interp_wdirn * 180. / np.pi
