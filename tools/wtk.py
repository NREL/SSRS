import importlib
import random
import utm
import numpy as np
import pandas as pd
from datetime import datetime
from calendar import monthrange
from typing import List, Tuple
from timezonefinder import TimezoneFinder
from astral import sun, LocationInfo


def create_hscfg_file(api_key: str):
    """ creates a .hscfg file required to read WTK data from AWS"""

    #print('Writing .hscfg file .. ', end="", flush=True)
    with open(".hscfg", "w") as f:
        f.write('hs_endpoint = https://developer.nrel.gov/api/hsds\n')
        f.write('hs_username = None\n')
        f.write('hs_password = None\n')
        f.write('hs_api_key = ' + api_key + '\n')
    # print('done')


def extract_wtk_locations(
        extent_lonlat: Tuple[float, float, float, float],
        extent_km: Tuple[float, float, float, float],
        modname: str,
        fname: str
):
    """ Extracts WTK grid locations for a given terrain bounds """

    hsds = importlib.import_module(modname)
    lonmin, lonmax, latmin, latmax = extent_lonlat
    xmin, xmax, ymin, ymax = extent_km
    _, _, zone_n, zone_l = utm.from_latlon(latmin, lonmin)
    tr_pad = 5000.
    latmin, lonmin = utm.to_latlon(xmin * 1000. - tr_pad,
                                   ymin * 1000. - tr_pad,
                                   zone_n, zone_l)
    latmax, lonmax = utm.to_latlon(xmax * 1000. + tr_pad,
                                   ymax * 1000. + tr_pad,
                                   zone_n, zone_l)
    with hsds.File(fname, mode='r') as f:
        wtk_fields = list(f)
        ts_lat_all = f['coordinates'][:, 0]
        lat_bool_all = np.logical_and(ts_lat_all > latmin,
                                      ts_lat_all < latmax)
        lat_index_all = np.where(lat_bool_all)[0]
        ts_lon = f['coordinates'][min(lat_index_all):max(lat_index_all), 1]
    ts_lat = ts_lat_all[min(lat_index_all):max(lat_index_all)]
    lat_bool = np.logical_and(ts_lat > latmin, ts_lat < latmax)
    lon_bool = np.logical_and(ts_lon > lonmin, ts_lon < lonmax)
    wtk_ind = np.where(np.logical_and(lon_bool, lat_bool))[0]
    out = np.vectorize(utm.from_latlon)(ts_lat[wtk_ind], ts_lon[wtk_ind])
    wtk_inds = min(lat_index_all) + wtk_ind
    wtk_xylocs_km = np.vstack((out[0][:] / 1000., out[1][:] / 1000.))
    return wtk_inds, wtk_xylocs_km.astype(np.float32), wtk_fields


def get_hours(
        lonlat: Tuple[float, float],
        timeofday: str,
        curdt: datetime
):
    """ returns list of hours based on coordinates """

    tf = TimezoneFinder()
    tzone = tf.timezone_at(lng=lonlat[0], lat=lonlat[1])
    aloc = LocationInfo(name='name', region='region', timezone=tzone,
                        longitude=lonlat[0], latitude=lonlat[1])
    sunloc = sun.sun(aloc.observer, date=curdt.now().date(),
                     tzinfo=aloc.timezone)
    srise = sunloc['sunrise'].hour + 1
    sset = sunloc['sunset'].hour + 1
    hours = np.array_split(np.array(range(srise, sset)), 3)
    if timeofday == 'morning':
        chosen_hours = list(hours[0])
    elif timeofday == 'afternoon':
        chosen_hours = list(hours[1])
    elif timeofday == 'evening':
        chosen_hours = list(hours[2])
    elif timeofday == 'daytime':
        chosen_hours = list(hours[0]) + list(hours[1]) + list(hours[2])
    else:
        raise ValueError('Incorrect timeofday string\n \
                        Options: morning, afternoon, evening, daytime')
    return chosen_hours, (srise, sset)


def get_random_datetimes(
        count: int,
        lonlat: Tuple[float, float],
        years: List[int],
        months: List[int],
        timeofday: str,
) -> List[datetime]:
    """ returns random list of datetimes given a 
    choice of years,months,timeofday"""
    datetime_list = set()
    i = 0
    while i < count:
        rnd_year = random.choice(years)
        rnd_month = random.choice(months)
        rnd_day = random.choice(range(*monthrange(rnd_year, rnd_month))) + 1
        rnd_date = datetime(rnd_year, rnd_month, rnd_day)
        hours, _ = get_hours(lonlat, timeofday, rnd_date)
        rnd_hour = random.choice(hours)
        rnd_datetime = datetime(rnd_year, rnd_month, rnd_day, rnd_hour)
        if rnd_datetime not in datetime_list:
            datetime_list.add(rnd_datetime)
            i += 1
    return list(datetime_list)


def extract_wtk_data(
        timestamp: datetime,
        wtk_indices: np.ndarray,
        wtk_columns: List[str],
        modname: str,
        fname: str
) -> pd.DataFrame:
    """Extracts WTK data for the wtk_columns and return a dataframe"""

    print('{0:s}'.format(timestamp.strftime('%I %p, %d %b %Y')), flush=True)
    hsds = importlib.import_module(modname)
    base_time = datetime(timestamp.year, 1, 1, 0)
    time_diff = timestamp - base_time
    time_index = time_diff.days * 24 + time_diff.seconds // 3600
    df_wtk = pd.DataFrame(columns=wtk_columns, index=wtk_indices)
    wtk_units = []
    with hsds.File(fname, mode='r') as f:
        for varname in wtk_columns:
            try:
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
            except:
                print(varname, 'not found!\nAvailable variables:\n', list(f))
                exit()
    df_wtk.columns = pd.MultiIndex.from_tuples(zip(df_wtk.columns, wtk_units),
                                               names=('Variable', 'Unit'))
    return df_wtk
