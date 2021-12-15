""" Scipt for running the SSRS simulation in Wyoming
around Top of the World wind power plane at 50 m resolution"""

from dataclasses import replace, asdict
from ssrs import Simulator, Config

config_jem_base = Config(
    run_name='appalachian',
    max_cores=16,
    out_dir='./output',
    southwest_lonlat=(39., -79.7),  # (lon, lat) for southwest pt
    region_width_km=(70., 60.),  # terrain width (xwidth, ywidth) in km
    resolution=100.,
    track_direction='north',
    track_start_region=(2, 68, 0, 0),
    track_count=1000
)

config_jem_uniform_north = replace(
    config_jem_base,
    sim_mode='uniform',
    uniform_winddirn=270.,
    uniform_windspeed=10.,
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
        config_jem_uniform_north,
        config_jem_snapshot_north,
        # config_jem_seasonal_north
    )
    for i, cfg in enumerate(configs_to_run):
        sim = Simulator(**asdict(cfg))
        sim.simulate_tracks()
        sim.plot_wtk_layers()
        sim.plot_ssrs_output()
        sim.plot_terrain_layers()
