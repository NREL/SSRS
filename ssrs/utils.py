""" Module for commonly used functions """

from typing import Tuple
import errno
import os
import time as tm
import shutil
from datetime import date, time
from timezonefinder import TimezoneFinder
from astral import sun, LocationInfo
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def get_sunrise_sunset_time(
    this_lonlat: Tuple[float, float],
    this_date: date
) -> Tuple[time, time]:
    """ Get the sunrise and sunset time in local time zone
    for a given lonlat and date """

    if not isinstance(this_date, date):
        raise ValueError('Provide a valid datetime.date object')
    tfinder = TimezoneFinder()
    tzone = tfinder.timezone_at(lng=this_lonlat[0], lat=this_lonlat[1])
    aloc = LocationInfo(name='name', region='region', timezone=tzone,
                        longitude=this_lonlat[0], latitude=this_lonlat[1])
    sunloc = sun.sun(aloc.observer, date=this_date, tzinfo=aloc.timezone)
    return sunloc['sunrise'].time(), sunloc['sunset'].time()


def create_gis_axis(
    cur_fig,
    cur_ax,
    cur_cm=None,
    km_bar: float = 10.
):
    """ Creates GIS axes """

    plt.tick_params(axis='both', which='both', bottom=False, top=False,
                    labelbottom=False, right=False, left=False,
                    labelleft=False)
    b_txt = str(int(km_bar)) + ' km'
    my_arrow = AnchoredSizeBar(cur_ax.transData, km_bar * 1000., b_txt, 3,
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
                                  shrink=0.8, aspect=40)
        cur_cb.outline.set_visible(False)
        cur_cb.ax.tick_params(size=0)
    else:
        cur_cb = None
    _, labels = cur_ax.get_legend_handles_labels()
    if labels:
        w = cur_fig.get_size_inches()[0]
        cur_lg = cur_ax.legend(bbox_to_anchor=(0, 1.005), ncol=int(w // 2),
                               loc='lower left', markerscale=2,
                               columnspacing=1.0, handletextpad=0.0,
                               borderaxespad=0., fontsize='small')
    else:
        cur_lg = None
    cur_ax.set_aspect('equal', adjustable='box')
    return cur_cb, cur_lg


def get_extent_from_bounds(
    bounds: Tuple[float, float, float, float],
    from_origin: bool = False,
    in_km: bool = False
) -> Tuple[float, float, float, float]:
    """ Get extent from bounds """
    extent = (bounds[0], bounds[2], bounds[1], bounds[3])
    if from_origin:
        extent = (0., extent[1] - bounds[0], 0., extent[3] - extent[2])
    if in_km:
        extent = [ix / 1000. for ix in extent]
    return extent


def makedir_if_not_exists(filename: str) -> None:
    """ Create the directory if it does not exists"""
    try:
        os.makedirs(filename)
    except OSError as e_name:
        if e_name.errno != errno.EEXIST:
            raise


def get_elapsed_time(start) -> str:
    "returns the elapsed time as string"
    hours, rem = divmod(tm.time() - start, 3600)
    mins, secs = divmod(rem, 60)
    if hours == 0:
        if mins == 0:
            xstr = f'{int(secs) + 1} sec'
        else:
            xstr = f'{int(mins)} min {int(secs)} sec'
    else:
        xstr = f'{int(hours)} hr {int(mins)} min'
    return xstr


def remove_all_dirs_in_this_dir(dname: str) -> None:
    """ remove all the subdirectories in the given directory"""
    if os.path.isdir(dname):
        dirnames = [f for f in os.scandir(dname) if f.is_dir()]
        for dirname in dirnames:
            shutil.rmtree(dirname)


def empty_this_directory(dirname: str):
    """ Delete the contents of this directory """
    filelist = list(os.listdir(dirname))
    for f in filelist:
        os.remove(os.path.join(dirname, f))
