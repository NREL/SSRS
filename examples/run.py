""" Scipt for running the SSRS simulations """
import os
import sys
from dataclasses import replace, asdict
# sys.path.append('')
# print(os.getcwd())
from ssrs import Simulator
from ssrs import Config


config_base = Config(
    # terrain info
    run_name='default',
    max_cores=16,
    southwest_lonlat=(42.78, -106.21),  # (lon, lat) for southwest pt
    region_width_km=(60., 50.),  # terrain width (xwidth, ywidth) in km
    resolution=100.
)

config_predefined_north = replace(
    config_base,
    sim_mode='predefined',
    predefined_winddirn=90.,
    predefined_windspeed=10.,
    track_direction='north',
    track_start_region=(5, 55, 0, 0)
)

config_predefined_south = replace(
    config_predefined_north,
    track_direction='south',
    track_start_region=(5, 55, 50, 50)
)


config_snapshot_north = replace(
    config_base,
    sim_mode='snapshot',
    snapshot_datetime=(2010, 6, 17, 13),
    track_direction='north',
    track_start_region=(5, 55, 0, 0)
)

config_snapshot_south = replace(
    config_snapshot_north,
    track_direction='south',
    track_start_region=(5, 55, 50, 50)
)

config_seasonal_north = replace(
    config_base,
    sim_mode='seasonal',
    seasonal_start=(3, 1),  # start of season (month, day)
    seasonal_end=(12, 1),  # end of season (month, day)
    seasonal_timeofday='daytime',  # morning, afternoon, evening, daytime
    seasonal_count=8,
    track_direction='north',
    track_start_region=(5, 55, 0, 0)
)

if __name__ == '__main__':

    sim = Simulator(**asdict(config_snapshot_north))
    sim.simulate_tracks()
    if
    sim.plot_terrain_layers()
    sim.plot_wtk_layers()
    sim.plot_ssrs_output()

    # sim = SimulatorSSRS(**asdict(config_snapshot_south))
    # sim.simulate_tracks()
    # sim.plot_directional_potentials()
    # sim.plot_simulated_tracks()
