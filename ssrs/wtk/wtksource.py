""" Module for dealing with various data sources of the WindToolKit dataset"""

import os
import importlib


class WtkSource:
    """ Class for defining various sources of WindToolKit data """

    valid_sources = ('AWS', 'EAGLE', 'EAGLE_LED')

    def __init__(
        self,
        sname: str
    ):
        self.api_website = None
        if sname == 'AWS':
            self.years = list(range(2007, 2015))
            fname = '/nrel/wtk/conus/wtk_conus_$YEAR.h5'
            self.module_name = 'h5pyd'
            self.api_website = 'https://developer.nrel.gov/signup/'
        elif sname == 'EAGLE':
            self.years = list(range(2007, 2015))
            fname = '/datasets/WIND/conus/v1.0.0/wtk_conus_$YEAR.h5'
            self.module_name = 'h5py'
        elif sname == 'EAGLE_LED':
            self.years = list(range(2018, 2020))
            fname = '/lustre/eaglefs/shared-projects/wtk-led/ERA5_En1/wtk_ERA5_En1_$YEAR.h5'
            self.module_name = 'h5py'
        else:
            raise ValueError((f'Invalid WindToolKit source: {sname}\nOptions:'
                              f'\n{chr(10).join(self.valid_sources)}\n'))

        print(f'Considering WindToolKit source: {sname}')
        self.file_names = [fname.replace('$YEAR', str(yr))
                           for yr in self.years]
        self.hsds = importlib.import_module(self.module_name)
        try:
            with self.hsds.File(self.file_names[0], mode='r') as f_obj:
                self.valid_layers = list(f_obj)
        except FileNotFoundError as _:
            if sname in ('EAGLE', 'EAGLE_LED'):
                tmp_str = (f'WTK source {sname} requires access to NREL '
                           f'EAGLE system, choose AWS instead!')
            else:
                tmp_str = 'Connection issues! Try again.'
            raise FileNotFoundError(
                f'Cannot find {self.file_names[0]}\n{tmp_str}') from None
        if sname == 'AWS':
            self.validate_aws_source()

    def validate_aws_source(self):
        """ Check if AWS source for WTK data is connectable """
        hscfg_fpath = os.path.join(os.getcwd(), '.hscfg')
        try:
            with self.hsds.File(self.file_names[0], mode='r') as f_obj:
                _ = list(f_obj)
        except OSError as _:
            raise ValueError(
                f'AWS: Invalid or Nonexistent file at {hscfg_fpath}') from None


def create_hscfg_file(api_key: str, fpath: str) -> None:
    """ creates a .hscfg file required to read WTK data from AWS"""

    with open(fpath, "w", encoding='UTF-8') as f_obj:
        f_obj.write('hs_endpoint = https://developer.nrel.gov/api/hsds\n')
        f_obj.write('hs_username = None\n')
        f_obj.write('hs_password = None\n')
        f_obj.write('hs_api_key = ' + api_key + '\n')
