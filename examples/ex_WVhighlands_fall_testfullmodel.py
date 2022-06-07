"""Script for running the SSRS simulation in West Virginia
over the Appalachian Mountains at 30-m resolution"""

from dataclasses import replace
from ssrs import Simulator, Config
import os

config_base = Config(
    run_name='WVhighlands__fall_testfullmodel',
    sim_movement='heuristics',
    movement_model='heuristics',
    
    out_dir='./output',
    max_cores=16,
    
    movement_ruleset='step_ahead_look_ahead', # dir_random_walk,step_ahead_drw,step_ahead_look_ahead,lookahead,mixed
    random_walk_freq=300, # if > 0, how often random walks will randomly occur -- approx every 1/random_walk_freq steps
    random_walk_step_range=(30,60), # when a random walk does occur, the number of random steps will occur in this range
    look_ahead_dist = 2000.0, #distance outward that bird will scan for strong updrafts
    
    thermals_realization_count=10,
    thermal_intensity_scale=2.5, #1 gives weak random field, 3 gives v strong random field    
    updraft_threshold=0.85,
    
    southwest_lonlat=(-79.35, 39.1),  # (lon, lat) for southwest pt, no integers!
    region_width_km=(30., 40.),  # terrain width (xwidth, ywidth) in km
    resolution=30., # meters
    
    track_direction=180., #202.5,
    track_start_region=(1, 29, 39.5, 39.5),  #xmin, xmax, ymin, ymax
    #track_start_region=(12, 18, 15, 25),  #xmin, xmax, ymin, ymax. placed centrally for random walk case
    track_count=100,  #per thermals realization
    
    # plotting related
    fig_height=6.,
    fig_dpi=300
)

config_uniform_north = replace(
    config_base,
    sim_mode='uniform',
    uniform_winddirn=135.,
    uniform_windspeed=6.,
)


config_snapshot_north = replace(
    config_base,
    sim_mode='snapshot',
    snapshot_datetime=(2010, 6, 17, 13),
)


config_seasonal_north = replace(
    config_base,
    sim_mode='seasonal',
    seasonal_start=(3, 1),  # start of season (month, day)
    seasonal_end=(6, 1),  # end of season (month, day)
    seasonal_timeofday='daytime',  # morning, afternoon, evening, daytime
    seasonal_count=8,
)

if __name__ == '__main__':
 
    configs_to_run = (
        config_uniform_north,
        #config_snapshot_north,
        # config_seasonal_north
    )
    for i, cfg in enumerate(configs_to_run):

#        for j in range(2):  #allows us to run for multiple realizations of the thermal field
#            sim = Simulator(cfg)
#            sim.simulate_tracks()
#            #sim.plot_terrain_features()
#            #sim.plot_wtk_layers()
#            sim.plot_simulation_output()
#            os.rename(sim.mode_fig_dir, f'{sim.mode_fig_dir}_{j}')
#            os.rename(sim.mode_data_dir, f'{sim.mode_data_dir}_{j}')
        
        sim = Simulator(cfg)
        sim.simulate_tracks_HSSRS()
        sim.plot_terrain_features()
        #sim.plot_wtk_layers()
        #sim.plot_directional_potentials()
        sim.plot_simulated_tracks_HSSRS()
        sim.plot_presence_map_HSSRS()
