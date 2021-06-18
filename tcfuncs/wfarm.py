from typing import List

import numpy as np
import pandas as pd
import utm

pd.set_option('display.float_format', lambda x: '%.1f' % x)


def get_all_wfarms(
        wf_state: str,
        wf_csv: str,
        print_verbose=False
) -> List[str]:
    """List all wind farms in a particular state using the USGS provided csv"""

    # Read raw turbine data and print out the columns
    df = pd.read_csv(wf_csv)
    if print_verbose:
        print('Got these columns in the dataframe: \n', df.columns)

    # List all windfarms in the given state, grouped by county
    wf_counties = df.loc[df.t_state == wf_state, 't_county'].unique()
    wf_names = []
    if print_verbose:
        print('Got these wfarms in {:s} (grouped by county):'.format(wf_state))
    for icounty in wf_counties:
        inames = list(df.loc[df.t_county == icounty, 'p_name'].unique())
        wf_names = np.concatenate((wf_names, inames))
        if print_verbose:
            print(icounty, ': \n', inames)
    return wf_names


def extract_turbine_data(
        wf_state: str,
        wf_names: List[str],
        wf_csv: str
) -> pd.DataFrame:
    """Extracts wind farm data (in lat/lon and Kms) using USGS provided csv"""

    # Check if wfarm names accurate
    print('\n--- Extracting wind turbine locations')
    print('CSV file used: {:s}'.format(wf_csv))
    print('State selected: {:s}'.format(wf_state))
    for wf in wf_names:
        if wf not in get_all_wfarms(wf_state, wf_csv):
            raise Exception('[' + wf + '] not present in ' + wf_state + '!')

    # Extract location data for turbines in km and lat/lon
    col_names = ['p_name', 't_cap', 't_hh', 't_rd', 'xlong', 'ylat']
    df_raw = pd.read_csv(wf_csv)
    df = df_raw.loc[(df_raw.t_state == wf_state) & df_raw.p_name.isin(wf_names),
                    col_names]
    xy_km = np.vectorize(utm.from_latlon)(df.ylat, df.xlong)[0:2]
    df = df.assign(xkm=xy_km[0] / 1000., ykm=xy_km[1] / 1000.)
    df.reset_index(drop=True, inplace=True)
    print('Wind farms selected: \n', df.p_name.unique())
    print('Summary of wind farm data: \n', df.describe(percentiles=[]))
    return df
