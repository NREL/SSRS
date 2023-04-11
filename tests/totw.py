import os
from ssrs import Config

defaults = {
    # general parameters
    #'max_cores': 4,
    'max_cores': 1,
    'sim_seed': 42, # for 1-to-1 reproducibility, need to set max_cores=1
    'out_dir': os.getcwd(),
    'sim_mode': 'snapshot',
    
    # terrain
    # - orig inputs
    'terrain_data_source': 'SRTM1',
    'resolution': 50.,  # desired terrain resolution [m]
    #'southwest_lonlat': (-106.030792, 42.899494), # SW corner of ToTW, Campbell Hill, Rolling Hills
    #'region_width_km': (50., 50.),
    # - old code inputs:
    #'terrain_width': [50., 50.], 
    #'terrain_southwest_pad': [12.,12], # to ensure a grid with `terrain_width` dimensions can be formed
    # - old code calculated SW corner (padded):
    #'southwest_lonlat': (-106.19979819,42.7717746), # padded
    #'region_width_km': (60., 55.), # adjusted to center ToTW
    'southwest_lonlat': (-106.15,42.80), # padded
    'region_width_km': (50., 50.),
    
    # atmospheric conditions
    'wtk_source': 'HRRR', # 'AWS', 'EAGLE', 'EAGLE_LED' or 'HRRR'
    
    # updraft calculations
    'orographic_model': 'original', # 'original', 'improved'
    'updraft_threshold': 0.0, # no thresholding applied [m/s]
    'smooth_threshold_cutoff': True,
    
    # track simulation
    'track_count': 500,  # number of simulated eagle tracks
    'track_start_type': 'structured',  # structured, random
    'track_stochastic_nu': 1., # scaling of move probs, 0 = random walk
    'track_dirn_restrict': 2,  # restrict within 45 deg of previous # moves
}

############ Case 1 ##########
case1 = Config(
    run_name='TOTW_case1_2020-03-18_12Z',
    snapshot_datetime=(2020,3,18,12),
    track_direction=180., # southbound
    track_start_region_origin=(-105.84346, 43.161419),
    track_start_region_origin_xy=False,
    track_start_region_width=0.5,
    track_start_region_depth=0.5,
    **defaults
)

############ Case 2 ##########
case2 = Config(
    run_name='TOTW_case2_2020-03-04_13Z',
    snapshot_datetime=(2020,3,4,13),
    track_direction=135., # southeast-bound
    track_start_region_origin=(-105.929413, 42.997623),
    track_start_region_origin_xy=False,
    track_start_region_width=0.5,
    track_start_region_depth=0.5,
    **defaults
)

############ Case 3 ##########
case3 = Config(
    run_name='TOTW_case3_2019-07-03_12Z',
    snapshot_datetime=(2019,7,3,12),
    track_direction=0., # northbound
    track_start_region_origin=(-105.749725, 42.84803),
    track_start_region_origin_xy=False,
    track_start_region_width=0.5,
    track_start_region_depth=0.5,
    **defaults
)

############ Case 4 ##########
case4 = Config(
    run_name='TOTW_case4_2020-04-18_10Z',
    snapshot_datetime=(2020,4,18,10),
    track_direction=270., # westbound
    track_start_region_origin=(-105.620247, 42.981663),
    track_start_region_origin_xy=False,
    track_start_region_width=0.25,
    track_start_region_depth=0.25,
    **defaults
)
