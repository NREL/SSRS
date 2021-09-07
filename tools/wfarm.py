from typing import List
import pandas as pd
import utm


def extract_wfarm_data_using_names(
        wf_state: str,
        wf_names: List[str]
) -> pd.DataFrame:
    """Extracts turbine data for specific projects using USTDB dataset"""

    url_ustdb = ('https://eersc.usgs.gov/api/uswtdb/v1/turbines?&t_cap=gt.0&'
                 'select=t_state,p_name,p_year,t_cap,t_hh,t_rd,xlong,ylat')
    print('Using user-supplied wfarm_names')
    df = pd.read_json(url_ustdb)
    wfarms_all = df.loc[df['t_state'] == wf_state, 'p_name'].unique()
    for wf in wf_names:
        if wf not in wfarms_all:
            print(*df['p_name'].unique())
            raise ValueError('[' + wf + '] not present in ' + wf_state + '!')
    wfbool = (df['t_state'] == wf_state) & (df['p_name'].isin(wf_names))
    dfsmall = df.loc[wfbool, :]
    xy_km = utm.from_latlon(dfsmall['ylat'].values,
                            dfsmall['xlong'].values)[0:2]
    dfsmall = dfsmall.assign(xkm=xy_km[0] / 1000., ykm=xy_km[1] / 1000.)
    dfsmall.reset_index(drop=True, inplace=True)
    return dfsmall


def extract_wfarm_data_using_lonlat(
    lonlat_bnds: List[float],
    min_hh: int
) -> pd.DataFrame:
    """Extracts turbine data given lon/lat bounds using USTDB dataset"""

    url_ustdb = ('https://eersc.usgs.gov/api/uswtdb/v1/turbines?&t_cap=gt.0&'
                 'select=t_state,p_name,p_year,t_cap,t_hh,t_rd,xlong,ylat')
    # print('Using user-supplied terrain lon/lat bounds')
    df = pd.read_json(url_ustdb)
    margin = 0.005
    lonmin = df.xlong > (lonlat_bnds[0] + margin)
    lonmax = df.xlong < (lonlat_bnds[1] - margin)
    latmin = df.ylat > (lonlat_bnds[2] + margin)
    latmax = df.ylat < (lonlat_bnds[3] - margin)
    print('Minimum Hub Height set to {0:5.1f} meters'.format(min_hh))
    minhh = df.t_hh >= min_hh
    dfsmall = df.loc[lonmin & latmin & lonmax & latmax & minhh, :]
    if dfsmall.shape[0] > 0:
        xy_km = utm.from_latlon(dfsmall['ylat'].values,
                                dfsmall['xlong'].values)[0:2]
        dfsmall = dfsmall.assign(xkm=xy_km[0] / 1000., ykm=xy_km[1] / 1000.)
    dfsmall.reset_index(drop=True, inplace=True)
    return dfsmall


def print_windfarm_details(df: pd.DataFrame):
    """ print basic details of the wind turbines """

    if df.shape[0] > 0:
        print('Wind farms found:')
        print('      Project         State  Year  Count Hub_hght(m) Rotor_dia(m)')
        for i, wf in enumerate(df['p_name'].unique()):
            wf_state = df.loc[df['p_name'] == wf, 't_state'].iloc[0]
            wf_year = int(df.loc[df['p_name'] == wf, 'p_year'].iloc[0])
            wf_count = df[df['p_name'] == wf].shape[0]
            wf_hh = df.loc[df['p_name'] == wf, 't_hh'].median()
            wf_rd = df.loc[df['p_name'] == wf, 't_rd'].median()
            print('{0:s}. {1:20s}{4:4s}{5:6d}{2:6d}{3:10.1f}{6:10.1f}'
                  .format(str(i + 1), wf, wf_count, wf_hh, wf_state, 
                  wf_year, wf_rd))
    else:
        print('No wind turbines found within the Lon/Lat bounds!')
