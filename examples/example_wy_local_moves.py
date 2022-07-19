"""Script for running the HSSRS simulation in Wyoming
around Top of the World wind power plant at 50-m resolution"""

from dataclasses import replace
from ssrs import Simulator, Config

config_jem_base = Config(
    run_name='wy_heuristic_local',
    sim_movement='heuristics',
    movement_ruleset='local_moves', # dir_random_walk,step_ahead_drw,step_ahead_look_ahead,lookahead,mixed
    random_walk_freq=200, # if > 0, how often random walks will randomly occur -- approx every 1/random_walk_freq steps
    random_walk_step_range=(30,60), # when a random walk does occur, the number of random steps will occur in this range
    look_ahead_dist = 2000.0, #distance outward that bird will scan for strong updrafts
    thermal_intensity_scale=2.5, #1 gives weak random field, 3 gives v strong random field
    max_cores=16,
    out_dir='./output',
    southwest_lonlat=(-106.21, 42.78),  # (lon, lat) for southwest pt
    region_width_km=(50., 50.),  # terrain width (xwidth, ywidth) in km
    resolution=50.,
    track_direction=180.,
    track_start_region=(24, 26, 24, 26),
    track_count=3
)

config_jem_uniform = replace(
    config_jem_base,
    sim_mode='uniform',
    uniform_winddirn=270.,
    uniform_windspeed=15.,
)


config_jem_snapshot_north = replace(
    config_jem_base,
    sim_mode='snapshot',
    snapshot_datetime=(2010, 6, 17, 13),
)


config_jem_seasonal_north = replace(
    config_jem_base,
    sim_mode='seasonal',
    seasonal_start=(3, 1),  # start of season (month, day)
    seasonal_end=(6, 1),  # end of season (month, day)
    seasonal_timeofday='daytime',  # morning, afternoon, evening, daytime
    seasonal_count=8,
)

if __name__ == '__main__':

    configs_to_run = (
        config_jem_uniform,
        #config_jem_snapshot_north,
        # config_jem_seasonal_north
    )
    for i, cfg in enumerate(configs_to_run):
        sim = Simulator(cfg)
        sim.simulate_tracks()
        sim.plot_terrain_features()
        sim.plot_wtk_layers()
        sim.plot_simulation_output()
