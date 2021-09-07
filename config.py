default = {
    'run_name': 'default',  # name of this run, determines directory names
    'max_cores': 8,  # maximum of cores to use
    'mode': 'predefined',  # snapshot, seasonal, predefined
    'terrain_southwest_lonlat': (-106.19, 42.78),
    'terrain_width': (50., 50.),  # terrain width (xwidth, ywidth) in km
    'terrain_res': 100.,  # desired terrain resolution (meters)
    'wfarm_minimum_hubheight': 75.,  # for extracting turbine locations
    'predefined_winddirn': 270.,  # northerly = 0., easterly = 90, westerly=270
    'predefined_windspeed': 10.,  # uniform wind speed in m/s
    'wtk_data_source': 'WTK_AWS',  # 'WTK_EAGLE', 'WTK_AWS', 'WTK_LED_EAGLE'
    'wtk_aws_key': 'Q6UcSUDu0mf1xk9nUUUi200HyERI6ZsaSzhKHvVY',  # to access AWS
    'wtk_seasonal_months': [3, 4, 5],  # list of months, order not important
    'wtk_seasonal_timeofday': 'daytime',  # morning, afternoon, evening, daytime
    'wtk_seasonal_count': 8,  # number of seasonal updraft computations
    'wtk_snapshot_datetime': (2010, 6, 17, 13),  # (year, month, date, hour)
    'wtk_orographic_height': 100,  # WTK wind conditions at this height
    'wtk_thermal_height': 100,  # WTK pressure, temperature, at this height
    'wtk_thermals_agl': 100.,  # altitude at which thermals are computed
    'wtk_interpolation_type': 'linear',  # 'nearest' 'linear' 'cubic'
    'wtk_use_saved_data': True,  # set to False when changing wtk_* pars,
    'track_direction': 'north',  # south,north,east,west,nwest,neast,swest,seast
    'track_count': 1000,  # number of simulated eagle tracks
    'track_start_region': (5, 45, 0, 0),  # [xmin, xmax, ymin, ymax] in km
    'track_start_type': 'uniform',  # uniform, random
    'track_stochastic_nu': 4.,  # scaling of move probs, 0 = random walk
    'track_dirn_restrict': 2,  # consideration of previous moves, options: 0,1,2
    'track_use_saved_data': True,  # set to False when changing the track_* pars
    'presence_smoothing_radius': 1000, # radius of smoothing kernel in meters
}


JEM = {**default,
       'run_name': 'JEM',
       'terrain_southwest_lonlat': (-106.19, 42.78),
       'terrain_width': (50, 50.),
       'terrain_res': 50.,
       'track_direction': 'north',
       'track_count': 1000,
       'track_start_region': (2, 48, 0, 0),
       }


WY_TOTW_big = {**default,
           'run_name': 'default_big',
           'terrain_southwest_lonlat': (-106.15, 42.74),
           'terrain_width': (80, 60.),
           'track_direction': 'north',
           'track_count': 1000,
           'track_start_region': (10, 70, 0, 0),
           }


CA_AltamontPass = {**default,
                   'run_name': 'CA_AltamontPass',
                   'terrain_southwest_lonlat': (-122.0, 37.5),
                   'terrain_width': (60, 50.),
                   'terrain_res': 100.,  
                   'track_direction': 'north',
                   'track_count': 1000,
                   'track_start_region': (2, 48, 0, 0),
                   }

PA_Appalachian = {**default,
                  'run_name': 'PA_Appalachian',
                  'terrain_southwest_lonlat': (-79.37, 39.52),
                  'terrain_width': (60, 50.),
                  'terrain_res': 100., 
                  'track_direction': 'north',
                  'track_count': 1000,
                  'track_start_region': (2, 48, 0, 0),
                  }


# initialization function
import json
import os
import argparse as ap
from types import SimpleNamespace
from tools.common import *


def setup_config() -> SimpleNamespace:

    print('\n--- Initializing')
    parser = ap.ArgumentParser()
    parser.add_argument("-c", "--configuration", action='store',
                        dest='config', help="configuration name")
    parser.add_argument("-o", "--override", action='store',
                        dest='override', help="override base config")
    args = parser.parse_args()
    try:
        config = eval(args.config) if args.config is not None else default
    except:
        exit('No default configuration found')
    if args.override is not None:
        config = parse_config_from_args(args.override, config)
    run_name = config['run_name'] if 'run_name' in config.keys() else 'noname'
    output_path = os.path.join(os.path.abspath(os.path.curdir), 'output')
    output_path = os.path.join(output_path, run_name)
    mode = config['mode'] if 'mode' in config.keys() else 'noname'
    config['output_dir'] = output_path
    config['data_dir'] = os.path.join(output_path, 'data/')
    config['fig_dir'] = os.path.join(output_path, 'figs/')
    config['terrain_data_dir'] = os.path.join(config['data_dir'], 'terrain/')
    config['terrain_fig_dir'] = os.path.join(config['fig_dir'], 'terrain/')
    config['mode_data_dir'] = os.path.join(config['data_dir'], mode + '/')
    config['mode_fig_dir'] = os.path.join(config['fig_dir'], mode + '/')
    config['dtime_format'] = 'y%Ym%md%dh%H'
    print('run_name:', run_name)
    print('mode:', mode)
    print('data_dir:', config['mode_data_dir'])
    print('fig_dir:', config['mode_fig_dir'])
    for key in ['output_dir', 'data_dir', 'fig_dir', 'mode_data_dir',
                'mode_fig_dir', 'terrain_data_dir', 'terrain_fig_dir']:
        makedir_if_not_exists(config[key])
    config_filename = os.path.join(output_path, run_name + '.json')
    with open(config_filename, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2, sort_keys=True)
    return SimpleNamespace(**config)
