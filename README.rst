# Stochastic Soaring Raptor Simulator (SSRS)

## A Stochastic agent-based model for predicting raptor movements during updraft-subsidized directional flights

The SSRS tool simulates multiple updraft-subsidized tracks for soaring raptors for a given set of atmospheric conditions, with the goal of predicting potential conflict with wind turbines. This tool has been programmed, verified, and validated for soaring golden eagles within the conterminous US, but could be easily extended for other regions and other soaring raptors. 


SSRS uses the following data sources for setting up the simulation of eagle tracks:
- Terrain altitude data for the user supplied longitude and latitude ranges (defined hereby as the study area) is extracted from NASA's Shuttle Radar Topography Mission (SRTM) dataset. More details at https://www2.jpl.nasa.gov/srtm/
-  The wind turbine locations within the study area are extracted from USGS's United States Wind Turbine Database (USWTDB). More details at https://eerscmap.usgs.gov/uswtdb/
- The atmospheric conditions are extracted from NREL's Wind ToolKit (WTK) dataset hosted on AWS. The dataset contains instantaneous 1-hour resolution model output data for 7 years (2007-2014) on a uniform 2-km grid that covers the continental US. More details at https://www.nrel.gov/grid/wind-toolkit.html.

SSRS operates under three modes: predefined, snapshot, and seasonal, as detailed below:
- In the predefined mode, uniform wind speed and direction is assumed for the entire study area. 
- In the snapshot mode, the focus is on a particular time instant for which the atmospheric data is extracted from WTK and multiple tracks are simulated based on the estimated orographic updrafts at that time instant. This mode is useful for predicting real-time soaring routes of an eagle approaching the study area. 
- In the seasonal mode, the focus is on a range of months and time of day. Multiple time instances are randomly selected for the given range of months and simulated tracks are produced for each time instant. This mode is useful for predicting the average behavior of soaring eagles for the range of atmospheric conditions experienced during the user-supplied month/time window.  

For each mode, a relative eagle presence density is produced using the simulated tracks that provides an indication of likely eagle presence while traversing the study area. This tool implements and extends the capability of the fluid-flow model from 'Brandes, D., & Ombalski, D. (
2004). Modelling raptor migration pathways using a fluid-flow analogy. The Journal of Raptor Research, 38(3), 195-207.'

## Keywords
Behavior modeling, Stochastic modeling, agent--based movement model, wind--wildlife interactions, raptor conservation, golden eagles

## Citation
Sandhu, Rimple, Tripp, Charles Edison, Thedin, Regis, Quon, Eliot, Lawson, Michael, Doubrawa, Paula, Draxl, Caroline, and Williams, Lindy. NREL/SSRS. Computer Software. https://github.com/NREL/SSRS. USDOE Office of Energy Efficiency and Renewable Energy (EERE), Renewable Power Office. Wind Energy Technologies Office. Web.

## Files and directory description

- tools/: module that contains useful functions associated with the SSRS tool
- output/: contains output from SSRS runs
- README.md: readme file
- enviroment.yml: used for setting up conda/python environment
- config.py: contains the parameters that dictate the SSRS output/results
- LICENSE.txt: Licence file
- preprocess.py: python script for extracting the wind turbine data, atmospheric data, and terrain altitude data.
- simulator.py: python script for simulating eagle tracks
- plotting.py: python script for plotting SSRS output
- run.sh: bash utility for running batch simulations


## Instructions on running the code


- To get started, install [Anaconda](https://www.anaconda.com/products/individual) or [miniconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/macos.html). Then, install the SSRS conda environement using environment.yml and activate the installed ssrs_env environment

```bash
conda deactivate
conda env create --file environment.yml
conda activate ssrs_env
```

- To access NREL's WTK dataset, the user needs to get an API key from https://developer.nrel.gov/signup/ and assign it to the parameter 'wtk_aws_key' in config.py. The example API key in config.py is for demonstation and is rate-limited per IP.

- Run python scripts for the default configuration defined in config.py as
```bash
python {script_name}.py
```
where {script_name} can be preprocess, simulator or plotting. Or, run a parameter configuration named {config_name} defined in config.py as
```bash
python {script_name}.py -c {config_name}
```
Or, run a parameter configuration named {config_name} with parameter {par_name} changed to {new_val} as
```bash
python {script_name}.py -c {config_name} -o {par_name1}:{new_val1},{par_name2}:{new_val2}
```
Note that {par_name}={new_val} will not work, it has to be {par_name1}:{new_val1} separated by commas. 

- Minimalistic set of commands to run after cd into the SSRS directory:
```bash
conda deactivate
conda env create --file environment.yml
conda activate ssrs_env
python preprocess.py
python simulator.py
python plotting.py
```

- In addition to the default configuration (which focuses on a study area in Wyoming), the config.py also provides additional configurations, namely CA_AltamontPass (focuses on Altamont Pass area in California) and PA_Appalachian (focuses on an area in Appalachian mountain). To run the snapshot mode for CA_AltamontPass case, use following commands (ensure you are in the SSRS directory):
```bash
conda activate ssrs_env
python preprocess.py -c CA_AltamontPass -o mode:snapshot
python simulator.py -c CA_AltamontPass -o mode:snapshot
python plotting.py -c CA_AltamontPass -o mode:snapshot
```
The corresponding figures will be in ./output/CA_AltamontPass/figs/snapshot/

## Parameter configuration options: 
SSRS relies on a set of parameter values  (defined as configuration) declared in config.py. Here is a brief description and recommended options for each parameter:
- Name of the SSRS simulation. The output data and figs will be stored in ./output/{run_name}/. Options: any string.
```bash
'run_name': 'default'
```
- Maximum number of cores to use when importing data from WTK and when simulating multiple eagle tracks. Only relevant for preprocess.py and simulator.py scripts. Options: positive integer, preferrably a multiple of 4.
```bash
'max_cores': 8
```
- Mode of the SSRS simulation. Options: 'snapshot', 'seasonal', 'predefined'. 
```bash
'mode': 'predefined'
```
- The longitude and latitude of the southwestern point of the study area. Options: any Longitute/Latitude combination within the contiguous United States.
```bash
'terrain_southwest_lonlat': (-106.19, 42.78),
```
- Width of the study area in UTM projection along longitude and latitude directions (in km).
```bash
'terrain_width': (50., 50.)
```
- Uniform terrain resolution in meters, considered same along longitude and latitude directions. The terrain altitude data extracted from SRTM is interpolated onto this resolution to create a structured grid within the study area with altitude known at each grid point.
```bash
'terrain_res': 100.
```
- The wind turbine locations within the study area are extracted from USGS USWTB ((https://eerscmap.usgs.gov/uswtdb/)) for turbines with hub height greater than the value of 'wfarm_minimum_hubheight' parameter in meters.
```bash
'wfarm_minimum_hubheight': 75.
```
- For predefined mode of SSRS, these parameters sets the wind speed and direction across the entire study area. Speed value in meters per second and direction value in degrees (measured clockwise from north). 90 = easterly wind, 180 = southery wind, 270 = westerly wind, 0 or 360 = northerly wind.
```bash
'predefined_winddirn': 270.
'predefined_windspeed': 10.
```
- The WTK file location for extracting atmospheric data (wind conditions, pressure temperature, etc). Options: 'WTK_EAGLE', 'WTK_AWS', 'WTK_LED_EAGLE'.
Both 'WTK_EAGLE' and 'WTK_LED_EAGLE' require access to NREL's HPC machine EAGLE, while 'WTK_AWS' uses the WTK data hosted on AWS. For 'WTK_AWS', the user should get their own key from https://developer.nrel.gov/signup/ and assign it to the 'wtk_aws_key' parameter.
```bash
'wtk_data_source': 'WTK_AWS'
'wtk_aws_key': 'Q6UcSUDu0mf1xk9nUUUi200HyERI6ZsaSzhKHvVY'
```
- The range of months for extracting atmosheric data from WTK for the seasonal mode of SSRS. Options: any subset and combination of [1,2,3,4,5,6,7,8,9,10,11,12], order not important. 
```bash
'wtk_seasonal_months': [3, 4, 5]
```
- The time of day for extracting the atmospehric data from WTK for the seasonal mode of SSRS. Options: 'morning', 'afternoon', 'evening', 'daytime'.
```bash
'wtk_seasonal_timeofday': 'daytime'
```
- The count of randomly selected hourly instances within the months defined in 'wtk_seasonal_months' and the time of day defined in 'wtk_seasonal_timeofday'. Options: any integer greater than 1.
```bash
'wtk_seasonal_count': 8
```
- The (year, month, date, hour) for the snapshot mode of SSRS. Note that for 'wtk_data_source':'WTK_AWS', the year has to be within [2007-2014], while for 'wtk_data_source': 'WTK_EAGLE' or 'WTK_LED_EAGLE' the year has to be within [2018-2019].
```bash
'wtk_snapshot_datetime': (2010, 6, 17, 13)
```
- The height above ground level (in meters) at which atmospheric data is extracted from WTK to compute the updrafts. 'wtk_orographic_height' is the height AGL at which wind conditions are imported for computing orographic updrafts, its value should be one of (10, 40, 60, 80, 100, 120, 140, 160, 200) meters. 'wtk_thermal_height' is the height at which pressure and temperature are imported for computing thermal updrafts, its value should be among (0, 100, 200). Note that thermal updrafts are only computed when 'wtk_data_source' is either 'WTK_EAGLE' or 'WTK_LED_EAGLE' which requires access to NREL's EAGLE HPC system. 
```bash
'wtk_orographic_height': 100
'wtk_thermal_height': 100
```
- The height above ground level (in meters) at which thermal updrafts are computed.
Options: Any value greater than zero and less than the expected boundary layer height at that time instant. Assigning a value higher than BLH will result in zero thermal updraft value across the study area.
```bash
'wtk_thermals_agl': 100.
```
- Type of interpolation for 2 km resolution atmospheric data from WTK onto the terrain resolution of 'terrain_res'. Options: 'nearest', 'linear', 'cubic'
```bash
'wtk_interpolation_type': 'linear'
```
- Set to False when changing wtk_* parameters, or else it uses the WTK data saved from previous runs. This helps avoid making unnecassery calls to the WTK server and saves computational time. Note that anytime terrain_* parameters are changed the code deletes all the previously saved data so no need to set this parameter to False when changing terrain_* parameters, only set it to False when changing wtk_* parameters without changing the terrain_* parameters. Options: True, False.
```bash
'wtk_use_saved_data': True
```
- The preferred direction of movement for simulated eagle tracks. Options: 'south', 'north', 'east', 'west', 'northwest', 'northeast', 'southwest', 'southeast'
```bash
'track_direction': 'north'
```
- Number of eagle tracks to be simulated for a given time instant. Options: positive non-zero integer
```bash
'track_count': 1000
```
- (xmin, xmax, ymin, ymax) of the rectangular region inside the study area where the simulated eagle tracks initiate. Ensure that terrain_width[0]>=xmax>=xmin>=0 and terrain_width[1]>=ymax>=ymin>=0. Note that this parameter assumes the southwest corner of the study area is at (0,0).
```bash
 'track_start_region': (5, 45, 0, 0)
```
- Determines how the starting grid location for simulating an eagle track is found within the rectangular region defined by 'track_start_region'. 'uniform' ensure equally spaced starting locations within 'track_start_region', while 'random' picks the starting location randomly within 'track_start_region'. Options: 'random', 'uniform'. 
```bash
 'track_start_type': 'uniform'
```
- Determines the level of stochasticity in eagle decision making. A value of zero means the movement will be purely random walk. A very high value will simulate increasingly deterministic moves with many simulated tracks following the same minimum energy expenditure path. Options: a positive floating number. 
```bash
'track_stochastic_nu': 4.
```
- The number of previous moves to consider to restrict the next move so as to avoid u-turn and sharp turning behaviour. Options: 0, 1, 2
```bash
'track_dirn_restrict': 2
```
- Set to False when changing track_* parameters, or else it uses the simulated track data saved from previous runs. This parameter is mainly useful in seasonal mode when hundreds of time instances need to be considered, setting this parameter to True will avoid rerunning previously ran time instances. Note that anytime terrain_* parameters are changed the code deletes all the previously saved data so no need to set this parameter to False when changing terrain_* parameters. Options: True, False.
```bash
'track_use_saved_data': True
```
-  Radius of the smoothing kernel in meters used for computing presence density from simulated tracks. A higher value will generate smoother presence maps.  Options: Any vaule greater than 'terrain_res' and less than the minimum width of the study area. Typically, this is chosen between 500 meters and 2000 meters.
```bash
'presence_smoothing_radius': 1000
```

## Credit

This software is currently developed and maintained by Rimple Sandhu (rimple.sandhu@nrel.gov) and Charles Tripp (charles.tripp@nrel.gov).

## Additional information:

- Variables available from WTK_AWS:
coordinates
inversemoninobukhovlength_2m
meta
precipitationrate_0m
pressure_0m
pressure_100m
pressure_200m
relativehumidity_2m
temperature_100m
temperature_10m
temperature_120m
temperature_140m
temperature_160m
temperature_200m
temperature_2m
temperature_40m
temperature_60m
temperature_80m
time_index
winddirection_100m
winddirection_10m
winddirection_120m
winddirection_140m
winddirection_160m
winddirection_200m
winddirection_40m
winddirection_60m
winddirection_80m
windspeed_100m
windspeed_10m
windspeed_120m
windspeed_140m
windspeed_160m
windspeed_200m
windspeed_40m
windspeed_60m
windspeed_80m

