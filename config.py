default = {
    # Common parameters
    'run_name': 'default',  # full name of this run
    'max_cores': 8,  # limits the maximum number of cores the code exploits
    'output_dir': './output/',
    'fig_size': (5, 5),  # figure size
    'fig_dpi': 200,  # dpi of saved png files
    'datetime_format': 'y%Ym%md%dh%H',  # for timestamping file names

    # Wind farm parameters (extract_data.py)
    'wfarm_fname': './data/uswtdb_v3_1_20200717.csv',
    'wfarm_state': 'WY',  # state in which wind farms located
    'wfarm_names': ('Top of the World', 'Campbell Hill', 'Rolling Hills'),
    'wfarm_labels': ('TOTW', 'Campbell Hill', 'Rolling Hills'),
    'wfarm_timezone': 'US/Mountain',  # MST, U

    # Terrain parameters (extract_data.py)
    'terrain_source': 'SRTM3',  # SRTM1: 30 m res, SRTM3: 90 m res
    'terrain_southwest_pad': [12., 12.],  # padding at southwest corner (Km)
    'terrain_width': [50., 50.],  # terrain width (Km)
    'terrain_res': 100. / 1000.,  # desired resolution (Km)

    # WIND ToolKit parameters (extract_data.py)
    'wtk_source': 'AWS',  # 'EAGLE' , 'AWS', 'EAGLE_new'
    'AWS_api_key': 'gKkIcs6Mf1KXK4djQZvJ65TFCoo5cTq1KxPWSxDb',  # to access AWS
    'wtk_mode': 'snapshot',  # snapshot: certain time, seasonal: range of times
    'wtk_snapshot_datetime': (2010, 6, 26, 11),  # year, month, date, time
    'wtk_seasonal_months': [3, 4, 5],  # list of months, order not important
    'wtk_seasonal_timeofday': 'daytime',  # morning, afternoon, evening, daytime

    'oro_varnames': ('windspeed_100m', 'winddirection_100m'),
    'thermal_varnames': ('pressure_100m', 'temperature_100m',
                         'boundary_layer_height', 'surface_heat_flux'),
    'var_labels': ('Wind speed (m/s)',
                   'Wind direction (Deg)',
                   'Pressure (Pa)', 'Temperature (C)',
                   'Boundary layer height (m)',
                   'Surface heat flux (K.m/s)',
                   'Potential temperature (C)'),

    # Updraft computation parameter (compute_updrafts.py)
    'seasonal_updraft_count': 4,  # number of random updraft computations
    # 'updraft_threshold': 0.85,  # threshold updraft speed m/s
    # 'thermal_altitude': 100.,  # altitude at which thermal updraft is computed
    'interp_type': 'cubic',  # WTK data interpolation 'nearest' 'linear' 'cubic'
    # 'weibull_k': 1.5,  # k parameter for the weibull distribution

    # Behaviour model parameters (run_tcmodel.py)
    'bndry_condition': 'mnorth',  # options: msouth,mnorth,meast,mwest
    'number_of_eagles': 100,
    'region_of_eagle_entry': [1., 49.],
    'type_of_eagle_entry': 'random',  # uniform, random
    'nu': 1.,  # scaling of move probabilities, 0 = random walk
    'dirn_restrict': 2,  # 45 degree from previous {dirn_restrict} moves
}

default_seasonal = {**default,
                    'run_name': 'default_seasonal',
                    'wtk_mode': 'seasonal',
                    }

# additioal configurations for special cases
# for_eliot = {**default,
#              'run_name': 'for_eliot',
#              'max_cores': 8,
#              'updraft_count': 1,
#              'wtk_source': 'AWS',
#              'wfarm_names': ('Top of the World',),
#              'wfarm_labels': ('TOTW', ),
#              'terrain_southwest_pad': [0.0, 0.0],
#              'terrain_width': [6., 6.],  # terrain width (Km)
#              'terrain_res': 100. / 1000.,
#              'number_of_eagles': 200,
#              'region_of_eagle_entry': [24.9, 25.1]
#              }

# for_paper = {**default,
#              'run_name': 'for_paper',
#              'max_cores': 8,
#              'updraft_count': 8,
#              'terrain_res': 50. / 1000.,
#              'number_of_eagles': 200,
#              'region_of_eagle_entry': [24.9, 25.1]
#              }

# for_paper_2 = {**default,
#                'run_name': 'for_paper_2',
#                'max_cores': 8,
#                'updraft_count': 8,
#                'terrain_res': 50. / 1000.,
#                'number_of_eagles': 2000,
#                'region_of_eagle_entry': [1., 49.]
#                }

# for_paper_3 = {**default,
#                'run_name': 'for_paper_3',
#                'max_cores': 8,
#                'updraft_count': 8,
#                'terrain_res': 50. / 1000.,
#                'number_of_eagles': 500,
#                'region_of_eagle_entry': [24.9, 25.1]
#                }

# base = {**default,
#         'run_name': 'base',
#         'max_cores': 8,
#         'terrain_res': 50. / 1000.,
#         'number_of_eagles': 1000,
#         }

# trial1 = {**base,
#           'run_name': 'trial1',
#           'region_of_eagle_entry': [25., 26.],
#           }

# trial2 = {**base,
#           'run_name': 'trial2',
#           'region_of_eagle_entry': [5., 6.],
#           }

# trial3 = {**base,
#           'run_name': 'trial3',
#           'region_of_eagle_entry': [45., 46.],
#           }

# trial4 = {**base,
#           'run_name': 'trial4',
#           'interp_type': 'cubic'
#           }

# trial5 = {**base,
#           'run_name': 'trial5',
#           'type_of_eagle_entry': 'random'
#           }

# trial6 = {**base,
#           'run_name': 'trial6',
#           'number_of_eagles': 2000,
#           }

# # Spring migration configurations
# spring_mnorth_morning = {**default,
#                          'run_name': 'spring_mnorth_morning',
#                          'wtk_months': [3, 4, 5],
#                          'wtk_timeofday': 'morning',
#                          'bndry_condition': 'mnorth',
#                          'updraft_count': 300, }

# spring_mnorth_afternoon = {**default,
#                            'run_name': 'spring_mnorth_afternoon',
#                            'wtk_months': [3, 4, 5],
#                            'wtk_timeofday': 'afternoon',
#                            'bndry_condition': 'mnorth',
#                            'updraft_count': 300, }


# spring_mnorth_evening = {**default,
#                          'run_name': 'spring_mnorth_evening',
#                          'wtk_months': [3, 4, 5],
#                          'wtk_timeofday': 'evening',
#                          'bndry_condition': 'mnorth',
#                          'updraft_count': 300, }


# # Fall migration configurations
# fall_msouth_morning = {**default,
#                        'run_name': 'fall_msouth_morning',
#                        'wtk_months': [9, 10, 11],
#                        'wtk_timeofday': 'morning',
#                        'bndry_condition': 'msouth',
#                        'updraft_count': 300, }


# fall_msouth_afternoon = {**default,
#                          'run_name': 'fall_msouth_afternoon',
#                          'wtk_months': [9, 10, 11],
#                          'wtk_timeofday': 'afternoon',
#                          'bndry_condition': 'msouth',
#                          'updraft_count': 300, }


# fall_msouth_evening = {**default,
#                        'run_name': 'fall_msouth_evening',
#                        'wtk_months': [9, 10, 11],
#                        'wtk_timeofday': 'evening',
#                        'bndry_condition': 'msouth',
#                        'updraft_count': 300, }
