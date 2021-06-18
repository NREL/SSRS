import importlib
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import utm

pd.set_option('display.float_format', lambda x: '%.1f' % x)


def create_hscfg_file(api_key: str):
    print('Writing .hscfg file .. ', end="")
    with open(".hscfg", "w") as f:
        f.write('hs_endpoint = https://developer.nrel.gov/api/hsds\n')
        f.write('hs_username = None\n')
        f.write('hs_password = None\n')
        f.write('hs_api_key = ' + api_key + '\n')
    print('done.')


# def get_wtk_sourceinfo(source: str,):
#     """ Returns list of years, WTK filename and associated package to load """

#     if source == 'AWS':
#         years = list(range(2007, 2015))
#         pname = 'h5pyd'
#         fname = '/nrel/wtk/conus/wtk_conus_'
#     elif source == 'EAGLE':
#         years = list(range(2007, 2015))
#         pname = 'h5py'
#         f_ext = '/datasets/WIND/conus/v1.0.0/wtk_conus_'
#     elif source == 'EAGLE_new':
#         years = list(range(2017, 2019))
#         pname = 'h5py'
#         f_ext = '/lustre/eaglefs/shared-projects/MSS/projects/' + \
#             'tap/ERA5_En1/WTK/h5/wtk_ERA5_En1_'
#     return years, pname, f_ext


def get_wtk_sourceinfo(
        source: str
):
    """ Returns list of years, WTK filename and associated package to load """

    if source == 'AWS':
        years = list(range(2007, 2015))
        pname = 'h5pyd'

        def fname(yr):
            return '/nrel/wtk/conus/wtk_conus_' + str(yr) + '.h5'
    elif source == 'EAGLE':
        years = list(range(2007, 2015))
        pname = 'h5py'

        def fname(yr):
            return '/datasets/WIND/conus/v1.0.0/wtk_conus_' + \
                   str(yr) + '.h5'
    elif source == 'EAGLE_new':
        years = list(range(2018, 2019))
        pname = 'h5py'

        def fname(yr):
            return '/lustre/eaglefs/shared-projects/wtk-led/' + \
                   'ERA5_En1/wtk_ERA5_En1_' + str(yr) + '.h5'
    return years, pname, fname


def extract_wtk_locations(
        lon_bnd: Tuple[float, float],
        lat_bnd: Tuple[float, float],
        modname: str,
        fname: str
) -> pd.DataFrame:
    """ Extracts WTK grid locations for a given lat/lon bounds """

    print('\n--- Extracting WTK data source locations')
    hsds = importlib.import_module(modname)
    print('WTK source: ', fname)
    with hsds.File(fname, mode='r') as f:
        # print('Got these fields: \n', list(f))
        ts_lat_all = f['coordinates'][:, 0]
        lat_bool_all = np.logical_and(ts_lat_all > lat_bnd[0],
                                      ts_lat_all < lat_bnd[1])
        lat_index_all = np.where(lat_bool_all)[0]
        ts_lon = f['coordinates'][min(lat_index_all):max(lat_index_all), 1]
    ts_lat = ts_lat_all[min(lat_index_all):max(lat_index_all)]
    lat_bool = np.logical_and(ts_lat > lat_bnd[0], ts_lat < lat_bnd[1])
    lon_bool = np.logical_and(ts_lon > lon_bnd[0], ts_lon < lon_bnd[1])
    wtk_ind = np.where(np.logical_and(lon_bool, lat_bool))[0]
    out = np.vectorize(utm.from_latlon)(ts_lat[wtk_ind], ts_lon[wtk_ind])
    df_locs = pd.DataFrame(list(zip(ts_lon[wtk_ind], ts_lat[wtk_ind],
                                    out[0][:] / 1000., out[1][:] / 1000.)),
                           columns=['xlong', 'ylat', 'xkm', 'ykm'],
                           index=min(lat_index_all) + wtk_ind)
    print(df_locs.describe(percentiles=[]))
    return df_locs


def extract_wtk_data(
        timestamp: datetime,
        wtk_indices: np.ndarray,
        wtk_columns: List[str],
        modname: str,
        fname: str
) -> pd.DataFrame:
    """Extracts WTK data for the wtk_columns and return a dataframe"""

    # Intialize
    print('{0:s}'.format(timestamp.strftime('%I %p, %x')), flush=True)
    hsds = importlib.import_module(modname)

    # Figure out the time index for requested WTK data
    base_time = datetime(timestamp.year, 1, 1, 0)
    time_diff = timestamp - base_time
    time_index = time_diff.days * 24 + time_diff.seconds // 3600

    # Extract the data
    df_wtk = pd.DataFrame(columns=wtk_columns, index=wtk_indices)
    wtk_units = []
    with hsds.File(fname, mode='r') as f:
        for varname in wtk_columns:
            inorm = f[varname].attrs['scale_factor']
            wtk_units.append(f[varname].attrs['units'])
            if modname == 'h5pyd':
                wtk_data_raw = f[varname][time_index, min(
                    wtk_indices):max(wtk_indices) + 1] / inorm
                df_wtk.loc[:, varname] = wtk_data_raw[wtk_indices -
                                                      min(wtk_indices)]
            else:
                df_wtk.loc[:, varname] = f[varname][time_index,
                                                    wtk_indices] / inorm
    df_wtk.columns = pd.MultiIndex.from_tuples(zip(df_wtk.columns, wtk_units),
                                               names=('Variable', 'Unit'))
    return df_wtk
