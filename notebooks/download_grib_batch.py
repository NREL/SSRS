from herbie import Herbie
import pandas as pd
from os.path import exists

log_filename = '/home/akey/Eagles/log_download_grib_batch.log'
input_filename = '/home/akey/Eagles/app_telemetry_data_unique_timestamps.csv'
start_row = 0

if not exists(log_filename):
    with open(log_filename, 'w') as f:
        f.write('row, timestamp_utc, status\n')

input_df = pd.read_csv(input_filename)
timestamps_utc = input_df['timestamp_utc']

for index, timestamp_utc in enumerate(timestamps_utc):
    try:
        H = Herbie(timestamp_utc, model='hrrr', product='sfc', fxx=0)
        H.download()
        with open(log_filename, 'a') as f:
            f.write(f'{index}, {timestamp_utc}, success\n')
    except ValueError as e:
        with open(log_filename, 'a') as f:
            f.write(f'{index}, {timestamp_utc}, fail\n')
        print(e)
