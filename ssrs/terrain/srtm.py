""" Module for dealing with NASA's Shuttle Radar Topography Mission (SRTM)
elevation data """

import os
from typing import Tuple
import elevation


class SRTM:
    """ Class for dealing with Shuttle Radar Topography Mission (SRTM) data

    Parameters:
    ----------
    bounds: Tuple[float, float, float, float]
        Bounds for the region in latlon coordinate system
        (min_lon, min_lat, max_lon, max_lat)
    layer: str
        layer requested, should be among valid_layerr
    fpath: str
        path of file where the raster data is saved in GeoTIFF format
    cleanup: bool, optional
        Remove cached data if true
    """

    valid_layers = ('SRTM1', 'SRTM3')

    def __init__(
        self,
        layer: str,
        bnds: Tuple[float, float, float, float],
        fpath: str,
        cleanup: bool = True
    ) -> None:
        # check if requested layer is valid
        if layer in self.valid_layers:
            self.layer = layer
        else:
            raise ValueError(f'SRTM: {layer} is invalid layer\n'
                             + f'Options:{self.valid_layers}\n')

        # check if requested bounds are valid
        if (bnds[2] > bnds[0]) and (bnds[3] > bnds[1]):
            self.bnds = bnds
        else:
            raise ValueError('SRTM: bounds should be northing and easting!')

        # check output file path
        out_dir = os.path.dirname(os.path.abspath(fpath))
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        self.fpath = fpath
        self.cleanup = cleanup

    def download(self) -> None:
        """ Downloading data from SRTM in GeoTIFF format """
        try:
            print(self.fpath)
            elevation.clip(self.bnds, product=self.layer,
                           output=self.fpath)
            if self.cleanup:
                elevation.clean()
        except Exception as _:
            raise Exception(
                f'SRTM: something went wrong with {self.layer}!') from None
