import os
import pickle
import random
import time
from calendar import monthrange
from datetime import datetime
from math import log10, floor
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astral import sun, LocationInfo
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


# import matplotlib as mpl
# mpl.use('Agg')
# plt.style.use('seaborn-notebook')
# from pprint import pprint
# plt.rcParams.update({'font.size': 8})


def round_to_1(x):
    return round(x, -int(floor(log10(abs(x)))))


def round_to_2(x):
    return round(x, -int(floor(log10(abs(x)))) + 1)


def get_min_mid_max(data, ext):
    width = np.amax(data) - np.amin(data)
    if np.all(data <= 1) and np.all(data >= 0):
        xmax = min(1, round_to_1(np.amax(data) + ext * width))
        xmin = max(0, round_to_1(np.amin(data) - ext * width))
        xmid = 0.5 * (xmin + xmax)
        if xmid != 0:
            xmid = round_to_2(xmid)
    else:
        xmax = round(np.amax(data) + ext * width)
        xmin = round(np.amin(data) - ext * width)
        if np.all(data > 0):
            xmin = max(0, xmin)
        xmid = round(0.5 * (xmin + xmax))
        if xmid == xmax or xmid == xmin:
            xmid = round_to_1(0.5 * (xmin + xmax))
    if xmin == xmax:
        xmax = xmin + 1
    return xmin, xmid, xmax


def create_pdf_axis(cur_fig, cur_ax, xmin, xmid, xmax):
    cur_ax.set_yticks([])
    cur_ax.set_xticks([xmin, xmid, xmax])
    cur_ax.set_xlim([xmin, xmax])
    cur_ax.grid(True)
    _, ymax = cur_ax.get_ylim()
    # print(ymin, ymax)
    cur_ax.set_ylim([0, 1.1 * ymax])
    cur_fig.tight_layout()


def initiate_timer(in_str: str):
    """ Returns the currne time and prints the string """
    print(in_str, end="", flush=True)
    return time.time()


def print_elapsed_time(start):
    hours, rem = divmod(time.time() - start, 3600)
    mins, secs = divmod(rem, 60)
    if hours == 0:
        if mins == 0:
            print(" .. took {0:d} sec".format(int(secs) + 1))
        else:
            print(" .. took {0:d} min {1:d} sec".format(int(mins), int(secs)))
    else:
        print(" .. took {0:d} hr {1:d} min".format(int(hours), int(mins)))


def file_exists(dirname: str, fname: str):
    return os.path.exists(dirname + fname)


def delete_files_of_type(dirname: str, fstring: str):
    for path, _, files in os.walk(dirname):
        for name in files:
            if name.endswith(fstring):
                os.remove(os.path.join(path, name))


def create_gis_axis(cur_fig, cur_ax, cur_cm=None, b_kms=10):
    """ Creates GIS axes """
    cur_ax.axis('scaled')
    plt.tick_params(axis='both', which='both', bottom=False, top=False,
                    labelbottom=False, right=False, left=False, labelleft=False)
    b_txt = str(int(b_kms)) + ' Km'
    my_arrow = AnchoredSizeBar(cur_ax.transData, b_kms, b_txt, 3,
                               pad=0.1, size_vertical=0.1, frameon=False)
    cur_ax.add_artist(my_arrow)
    arrowpr = dict(fc="k", ec="k", alpha=0.9, lw=2.1,
                   arrowstyle="<-,head_length=1.0")
    cur_ax.annotate('N', xy=(0.03, 0.925), xycoords='axes fraction',
                    xytext=(0.03, 0.99), textcoords='axes fraction',
                    arrowprops=arrowpr,
                    bbox=dict(pad=-4, facecolor="none", edgecolor="none"),
                    ha='center', va='top', alpha=0.9)
    if cur_cm:
        cur_cb = cur_fig.colorbar(cur_cm, ax=cur_ax, pad=0.01,
                                  shrink=0.82, aspect=40)
        cur_cb.outline.set_visible(False)
        cur_cb.ax.tick_params(size=0)
    else:
        cur_cb = None
    _, labels = cur_ax.get_legend_handles_labels()
    if labels:
        cur_lg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.07),
                            frameon=False, ncol=4, markerscale=2,
                            handletextpad=0.2, columnspacing=1.0)
    else:
        cur_lg = None
    cur_fig.tight_layout()
    return cur_cb, cur_lg


# Saving data
def save_data(dname: str, fname: str, data, **kwrgs) -> None:
    if fname.endswith('.npy'):
        np.save(dname + fname, data, **kwrgs)
    elif fname.endswith('.pkl'):
        pickle.dump(data, open(dname + fname, "wb"))
    elif fname.endswith('.csv'):
        data.to_csv(dname + fname, **kwrgs)
    else:
        np.savetxt(dname + fname, data, **kwrgs)


# Load data
def load_data(dname: str, fname: str, **kwrgs) -> np.ndarray:
    if fname.endswith('.npy'):
        data = np.load(dname + fname, **kwrgs)
    elif fname.endswith('.csv'):
        data = pd.read_csv(dname + fname, index_col=0, **kwrgs)
    elif fname.endswith('.pkl'):
        data = pickle.load(open(dname + fname, "rb"))
    else:
        data = np.loadtxt(dname + fname, **kwrgs)
    return data


# Saving figures
def save_fig(fig: Figure,
             dname: str,
             fname: str,
             fig_dpi: int = 200,
             fig_bbox: str = 'tight',
             **kwrgs) -> None:
    fig.savefig(dname + fname, dpi=fig_dpi,
                bbox_inches=fig_bbox, **kwrgs)
    plt.close(fig)


# get a file name using wfarm name
def get_wfarm_savename(
        iname: str
) -> str:
    return 'wfarm_' + iname.replace(" ", "_").lower() + '.txt'


# new colormap
def get_transparent_cmap(cmp: str, cmin: float, bnds: List[float]):
    icmp = cm.get_cmap(cmp)
    newcolors = icmp(np.arange(icmp.N))
    ind = int(icmp.N * cmin / (bnds[1] - bnds[0]))
    newcolors[:ind, :] = (1, 0, 0, 0)
    newcolors[ind:, -1] = np.linspace(0.6, 0.6, icmp.N - ind)
    newcmp = ListedColormap(newcolors)
    return newcmp


def get_saved_datetimes(
        data_dir: str,
        dformat: str,
        fstring: str
) -> List[datetime]:
    """ get datetimes from saved files that contain fstring"""
    run_ids = set()
    dnames = [f for f in os.scandir(data_dir) if f.is_dir()]
    for dname in dnames:
        for fname in os.listdir(dname):
            if fstring in fname:
                run_ids.add(datetime.strptime(dname.name, dformat))
    return list(run_ids)


def get_hours(
        lonlat: Tuple[float, float],
        timeofday: str,
        tzone: str,
        cur_day: datetime
) -> List[int]:
    """ returns list of hours based on coordinates """

    aloc = LocationInfo(name='name', region='region', timezone=tzone,
                        longitude=lonlat[0], latitude=lonlat[1])
    sunloc = sun.sun(aloc.observer, date=cur_day.now().date(),
                     tzinfo=aloc.timezone)
    srise = sunloc['sunrise'].hour + 1
    sset = sunloc['sunset'].hour + 1
    hours = np.array_split(np.array(range(srise, sset)), 3)
    if timeofday == 'morning':
        return list(hours[0])
    elif timeofday == 'afternoon':
        return list(hours[1])
    elif timeofday == 'evening':
        return list(hours[2])
    elif timeofday == 'daytime':
        return list(hours[0]) + list(hours[1]) + list(hours[2])
    else:
        raise Exception('Incorrect timeofday string\n \
                        options: morning, afternoon, evening, daytime')


def get_random_datetimes(
        count: int,
        lonlat: Tuple[float, float],
        years: List[int],
        months: List[int],
        timeofday: str,
        timezone: str
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
        hours = get_hours(lonlat, timeofday, timezone, rnd_date)
        rnd_hour = random.choice(hours)
        rnd_datetime = datetime(rnd_year, rnd_month, rnd_day, rnd_hour)
        if rnd_datetime not in datetime_list:
            datetime_list.add(rnd_datetime)
            i += 1
    return list(datetime_list)


def standardize_matrix(
        inA: np.ndarray
) -> np.ndarray:
    tol = 1e-5
    if np.amax(inA) - np.amin(inA) < 1e-5:
        outA = np.multiply(np.ones(inA.shape), 1.)
    else:
        outA = np.divide(np.subtract(inA, np.amin(inA)),
                         np.amax(inA) - np.amin(inA))
        outA = outA.clip(min=tol)
    # outA += np.random.rand(inA.shape[0],inA.shape[1])*tol
    return outA
