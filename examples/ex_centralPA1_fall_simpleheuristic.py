"""Script for running the simple heuristic simulation over Stone/Jacks convergence and gap in central PA at 30-m resolution"""

from dataclasses import replace
from ssrs import Simulator, Config
import os

config_base = Config(
    run_name='centralPA1_fall_simple',
    sim_movement='heuristics',
    movement_model='heuristics',
    movement_ruleset='simple_step_ahead_drw_orog', 
    #options are: random_walk, dir_random_walk, simple_step_ahead_drw_orog, step_ahead_drw_mixedlift, step_ahead_look_ahead_mixedlift
    
    out_dir='./output',
    max_cores=16,
    
    random_walk_freq=10000, # if > 0, how often random walks will randomly occur -- approx every 1/random_walk_freq steps
                            #use large number to eliminate randomwalks
    random_walk_step_range=(30,60), # when a random walk does occur, the number of random steps will occur in this range
    
    look_ahead_dist = 3000.0, #distance outward (m) that bird will scan for strong updrafts
    updraft_threshold=0.85,  
    
    orographic_model='original',  # original, improved
    thermal_model = 'naive',      # naive, 'improvedAllen'
    wind_data_source='wtk',

    thermals_realization_count=1,
    thermal_intensity_scale=0,    # 1 gives weak random field, 3 gives v strong random field    
    
    southwest_lonlat=(-78.1, 40.35),  # (lon, lat) for southwest pt, no integers!
    region_width_km=(30., 40.),  # terrain width (xwidth, ywidth) in km
    resolution=30., # meters
    
    track_direction=202.5, #202.5 = SSW
    track_start_region=(10, 29.5, 39.5, 39.5),  #xmin, xmax, ymin, ymax
    #track_start_region=(12, 18, 15, 25),  #xmin, xmax, ymin, ymax. placed centrally for random walk case
    track_start_type='structured',  # structured, random
    track_count=100,  #50,  #per thermals realization
    
    # plotting related
    fig_height=6.,
    fig_dpi=150
)

config_uniform_northwest_bo04Orog_naiveTher = replace(
    config_base,
    run_name='centralPA1_fall_simple_bo04Orog_naiveTher',
    sim_mode='uniform',
    orographic_model='original',  # original, improved
    thermal_model = 'naive',      # naive, 'improvedAllen'
    thermal_intensity_scale=0,    # 1 gives weak random field, 3 gives v strong random field    
    wind_data_source = 'wtk',
    uniform_winddirn_href=315.,
    uniform_windspeed_href=8.,
    uniform_winddirn_h=315.,
    uniform_windspeed_h=8.,
    uniform_winddirn=315.,
    uniform_windspeed=8.,
)

config_uniform_northwest_evveOrog_naiveTher = replace(
    config_base,
    run_name='centralPA1_fall_simple_evveOrog_naiveTher',
    sim_mode='uniform',
    orographic_model='improved',  # original, improved
    h = 80,
    href = 80,
    thermal_model = 'naive',      # naive, 'improvedAllen'
    thermal_intensity_scale=0,    # 1 gives weak random field, 3 gives v strong random field    
    wind_data_source = 'wtk',
    uniform_winddirn_href=315.,
    uniform_windspeed_href=8.,
    uniform_winddirn_h=315.,
    uniform_windspeed_h=8.,
    uniform_winddirn=315.,
    uniform_windspeed=8.,
)

config_snapshot_northwest_evveOrog_evveTher = replace(
    config_base,
    run_name='centralPA1_fall_simple_evveOrog_evveTher',
    sim_mode='snapshot',
    orographic_model='improved',  # original, improved
    h = 200,
    href = 80,
    thermal_model = 'improvedAllen',      # naive, 'improvedAllen'
    wind_data_source = 'hrrr',
    #region_width_km=(50., 60.),  # terrain width (xwidth, ywidth) in km
    #resolution=50., # meters
    thermals_realization_count=1,
    snapshot_datetime=(2021, 6, 13, 22),
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
        config_uniform_northwest_bo04Orog_naiveTher,
        config_uniform_northwest_evveOrog_naiveTher,
        config_snapshot_northwest_evveOrog_evveTher,
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

        sim.plot_terrain_features()
        sim.plot_updrafts(plot='pcolormesh', vmax=2)
        sim.plot_thermal_updrafts()
        sim.plot_atmospheric_conditions_HRRR()
        #sim.plot_wtk_layers()
        #sim.plot_directional_potentials()
        sim.simulate_tracks_HSSRS(PAM_stdev=15.)
        sim.plot_simulated_tracks_HSSRS()
        sim.plot_presence_map_HSSRS()

