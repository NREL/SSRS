Stochastic Soaring Raptor Simulator (SSRS)
===========================================

The goal of SSRS is to predict movements of soaring raptors (such as
Golden Eagles) for a given wind conditions with the aim of determining
potential negative conflict between soaring raptors and wind
turbines. SSRS uses a stochastic agent-based model for predicting raptor
movements through an orographic updraft field estimated using the
spatially varying wind conditions and ground features (altitude, slope, aspect).
SSRS can be applied to any rectangular region within the US without the
need for any eagle-centric or atmosphere-related data collection efforts, using
only the publically available data sources. SSRS implements and extends the
capability of the fluid-flow model from 'Brandes, D., & Ombalski, D. (2004). 
Modelling raptor migration pathways using a fluid-flow analogy. The Journal
of Raptor Research, 38(3), 195-207.'


SSRS uses the following publically available data sources:

* USGS's `3D Elevation Program (3DEP) <https://www2.jpl.nasa.gov/srtm/>`_ dataset for terrain altitude, slope and aspect at a spatial resolution of 10 meters within the US.
* USGS's `United States Wind Turbine Database (USWTDB) <https://eerscmap.usgs.gov/uswtdb/>`_ for up-to-date turbine locations within the US.
* NREL's `Wind ToolKit (WTK) <https://www.nrel.gov/grid/wind-toolkit.html>`_ dataset for atmospheric conditions such as wind speed and direction at 1-hour temporal resolution and 2 km spatial resolution within the US.

SSRS operates under three modes: 

* **Uniform**: Uses uniform wind speed and direction across the target region.
* **Snapshot**: Uses wind conditions for a specific time imported from WTK dataset.
* **Seasonal**: Uses wind conditions randomly sampled from a range of dates or months or time of day from the WTK dataset.


Installation
--------------
Without Anaconda/miniconda (requires python>=3.8 and pip>21.3):

.. code-block:: bash

    pip install git+https://github.com/NREL/SSRS.git#egg=ssrs

With Anaconda/miniconda:

.. code-block:: bash

    conda create -n my_env_name python=3.8 pip
    conda activate my_env_name
    pip install git+https://github.com/NREL/SSRS.git#egg=ssrs

Alternatively, clone the GitHub repository on local machine,
cd into the SSRS directory and run following commands in the terminal

.. code-block:: bash

    conda env create -f environment.yml
    conda activate ssrs_env
    pip install .

For running conda environment ssrs_env in Jupyter Notebook,

.. code-block:: bash

    conda install ipykernel
    ipython kernel install --user --name=ssrs_env

For SSRS to access NREL's WTK dataset in the snapshot mode, need to get an
API key from https://developer.nrel.gov/signup/ and save it in .hscfg file provided

Usage
--------------

The Jupyter Notebooks in examples/ show the usage of SSRS for a given region.
For instance, see this notebook_ for ssrs simulation in uniform mode for a region
in Wyoming, simulating 1000 eagles travelling north, which will generate
following figures:

.. _notebook: notebooks/sample_ssrs_uniform.ipynb

Ground elevation and turbine locations:

.. image:: notebooks/output/run_wy/figs/elevation.png
    :width: 2000 px
    :scale: 20 %
    :align: left
    :alt: Ground elevation and turbine locations

Orographic updrafts:

.. image:: notebooks/output/run_wy/figs/uniform/s10d270_orograph.png
    :width: 2000 px
    :scale: 20 %
    :align: left
    :alt: Orographic updrafts

1000 simulated tracks travelling towards north:

.. image:: notebooks/output/run_wy/figs/uniform/s10d270_north_tracks.png
    :width: 2000 px
    :scale: 20 %
    :align: right
    :alt: 

Relative eagle presence density

.. image:: notebooks/output/run_wy/figs/uniform/s10d270_north_presence.png
    :width: 2000 px
    :scale: 20 %
    :align: right
    :alt: Relative eagle presence density


Configuration
--------------

Parameter configuration options: 
SSRS relies on a set of parameter values  (defined as configuration) 
general parameters for the SSRS simulation

.. code-block:: python

    run_name: str = 'default'  # name of this run, determines directory names
    
    out_dir: str = os.path.join(os.path.abspath(os.path.curdir), 'output')
    max_cores: int = 8  # maximum of cores to use
    sim_mode: str = 'uniform'  # snapshot, seasonal, uniform

    southwest_lonlat: Tuple[float, float] = (42.78, -106.21)
    projected_crs: str = 'ESRI:102008'  # ESRI, EPSG, PROJ4 or WKT string
    region_width_km: Tuple[float, float] = (30., 20.)
    resolution: int = 100.  # desired terrain resolution (meters)


    uniform_winddirn: float = 270.  # northerly = 0., easterly = 90, westerly=270
    uniform_windspeed: float = 10.  # uniform wind speed in m/s

    # parameters for seasonal mode
    seasonal_start: Tuple[int, int] = (3, 1)  # start of season (month, day)
    seasonal_end: Tuple[int, int] = (12, 1)  # end of season (month, day)
    seasonal_timeofday: str = 'daytime'  # morning, afternoon, evening, daytime
    seasonal_count: int = 8  # number of seasonal updraft computations

    # parameters for snapshot mode
    snapshot_datetime: Tuple[int, int, int, int] = (2010, 6, 17, 13)

    # downloading data from WTK
    wtk_source: str = 'AWS'  # 'EAGLE', 'AWS', 'EAGLE_LED'
    wtk_orographic_height: int = 100  # WTK wind conditions at this height
    wtk_thermal_height: int = 100  # WTK pressure, temperature, at this height
    wtk_interp_type: str = 'linear'  # 'nearest' 'linear' 'cubic'

    # parameters for simulating tracks
    track_direction: str = 'north'  # south,north,east,west,nwest,neast,swest,seast
    track_count: str = 100  # number of simulated eagle tracks
    track_start_region: Tuple[float, float, float, float] = (5, 45, 0, 0)
    track_start_type: str = 'random'  # uniform, random
    track_stochastic_nu: float = 1.  # scaling of move probs, 0 = random walk
    track_dirn_restrict: int = 2  # consideration of previous moves, options: 0,1,2

    # plotting related
    fig_height: float = 6.
    fig_dpi: int = 200  # increase this to get finer plots
    turbine_minimum_hubheight: float = 50.  # for select turbine locations
    turbine_mrkr_styles = ('1k', '2k', '3k', '4k',
                           '+k', 'xk', '*k', '.k', 'ok')
    turbine_mrkr_size: float = 3.
    turbine_box_around_wfarm: bool = False
    presence_smoothing_radius: bool = 10  # smoothing radius in meters


Keywords
--------------
Behavior modeling, Stochastic modeling, agent--based movement model, wind--wildlife interactions, raptor conservation, golden eagles

Citation
--------------
Sandhu, Rimple, Tripp, Charles Edison, Thedin, Regis, Quon, Eliot, Lawson, Michael, Doubrawa, Paula, Draxl, Caroline, and Williams, Lindy. NREL/SSRS. Computer Software. https://github.com/NREL/SSRS. USDOE Office of Energy Efficiency and Renewable Energy (EERE), Renewable Power Office. Wind Energy Technologies Office. Web.



