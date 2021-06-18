# Stochastic Soaring Raptor Simulator (SSRS)

## A Stochastic agent-based model for predicting raptor movements during updraft-subsidized directional flights

This project implements and extends the capability of the terrain conductance model from 'Brandes, D., & Ombalski, D. (
2004). Modelling raptor migration pathways using a fluid-flow analogy. The Journal of Raptor Research, 38(3), 195-207.'
The code provides an eagle presence map for a given setting of wind conditions and movement direction.

## Authors

+ Rimple Sandhu (National Wind Technology Center, National Renewable Energy Laboratory)
+ Charles Tripp (Computational Science Center, National Renewable Energy Laboratory)
+ Eliot Quon (National Wind Technology Center, National Renewable Energy Laboratory)
+ Regis Thedin (National Wind Technology Center, National Renewable Energy Laboratory)
+ Lindy Williams (Computational Science Center, National Renewable Energy Laboratory)
+ Paula Doubrawa (National Wind Technology Center, National Renewable Energy Laboratory)
+ Caroline Draxl (National Wind Technology Center, National Renewable Energy Laboratory)
+ Mike Lawson (National Wind Technology Center, National Renewable Energy Laboratory)

### Keywords
Behavior modeling, Stochastic modeling, agent--based movement model, wind--wildlife interactions, raptor conservation, golden eagles


### Files and directory description

- tcfuncs/: module that contains useful functions associated with the terrain conductance model
- data/ : contain raw turbine location data
- output/: contains output from model runs
- config.py: contains all parameters that drive the model
- extract_data.py: python script for extracting the wind turbine, terrain and WTK data
- compute_updrafts.py: python script for computing orographic updrafts
- run_tcmodel.py: python script for running the model
- generate_plots.py: plots the output from previous three scripts

### Instructions on running the code

- Get the mmctools module: Clone the mmctools repository by running

```bash
git clone --single-branch --branch dev https://github.com/a2e-mmc/mmctools.git
```

then cd into the mmctools directory and run

```bash
git checkout 5c2cbd32a1ca7ea23e8a67d7d93a653facb15296
```

Code is tested for this commit. Then add the following line in your ~/.bash_profile file:

```bash
export PYTHONPATH="$MMC_DIR:$PYTHONPATH" 
```

where $MMC_DIR is the direcotry containing the mmctools directory

- Install the conda environement: Activate conda and install using environments.yml file. The environment.yml is
  included in the repo.

```bash 
conda deactivate
conda env create --file environment.yml
conda activate tc_env
```

The last command should activate tc_env

- Run python scripts: run individually for default configurationusing

```bash
python extract_data.py default
python compute_updrafts.py default
python run_tcmodel.py default
python generate_plots.py default
```

- To run for a configuration other than the default config, do

```bash
python {script_name}.py {config_name}
```

where {config_name} is a configuration defined in config.py. The default config is the 'snapshot' mode. Try replacing '
default' with 'default_seasonal' to run the model in seasonal mode.

<!-- - To rerun a script for any of the previous case with run_name of {sample_run}, do
```bash
python {script_name}.py run_name {sample_run}
``` -->
<!-- In this case the code reads the configuration from file ./output/{sample_run}/{sample_run}.json -->

### Additional information:

- Wind farm data available at https://eerscmap.usgs.gov/uswtdb/data/
- Change 'wtk_filesource':'EAGLE' when running on NREL's EAGLE supercomputer
- timezones pytz.all_timezones
  'US/Alaska',
  'US/Aleutian',
  'US/Arizona',
  'US/Central',
  'US/East-Indiana',
  'US/Eastern',
  'US/Hawaii',
  'US/Indiana-Starke',
  'US/Michigan',
  'US/Mountain',
  'US/Pacific',
  'US/Samoa',
