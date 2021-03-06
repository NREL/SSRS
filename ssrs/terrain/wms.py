""" Module for using Web Map Service (WMS) tool to download GIS data """

import os
import errno
from typing import Tuple, List
import numpy as np
from owslib.wms import WebMapService
from rasterio.merge import merge
import rasterio as rs


class WMS:
    """ Class for accessing data through Web Map Service (WMS)

    Parameters:
    -----------
    bnds: Tuple[float, float, float, float]
        Bounds of the rectangular region [min_x, min_y, max_x, max_y]
    crs_str: string
        EPSG or ESRI string that defines the projected crs
    url : str
        The base url for the WMS service
    version : str, optional, defaults to 1.3.0.
        The WMS service version which should be either 1.1.1 or 1.3.0,
    max_gridsize: int, optional, defaults to 2000
        The maximum grid size that can be requested from the server,
        used to segment region into tiles. WMS server calls only
        allow requests that has grid size smaller than some fixed value
    """
    fmt = 'image/tiff'
    version: str = '1.3.0'

    def __init__(
        self,
        bnds: Tuple[float, float, float, float],
        crs_str: str,
        url: str,
        max_gridsize: int = 2000
    ) -> None:

        self.max_gridsize = max_gridsize
        self.bnds = bnds
        self.crs_str = crs_str

        try:
            self.wms = WebMapService(url, version=self.version)
        except Exception as _:
            raise Exception('WMS: Connection issues, try again!\n') from None
        self.layers = list(self.wms.contents)
        #self.operations = [op.name for op in self.wms.operations]
        formats = self.wms.getOperationByName("GetMap").formatOptions
        self.crss = {lyr: [s.lower() for s in self.wms[lyr].crsOptions]
                     for lyr in self.layers}
        self.bboxs = {lyr: self.wms[lyr].boundingBoxWGS84
                      for lyr in self.layers}
        if self.fmt not in formats:
            raise ValueError(f'WMS: Invalid format {self.fmt}\nOptions:\n'
                             + f'{chr(10).join(formats)}\n')
        if not (bnds[2] > bnds[0]) & (bnds[3] > bnds[1]):
            raise ValueError(f'WMS: Invalid bounds {self.bnds}\n')

    def validate_layer(self, layer: str) -> None:
        """ Check if requested layer is valid, and available in crs_str """
        if layer not in self.layers:
            raise ValueError(f"WMS: Invalid layer {layer}\nOptions:\n"
                             + f"{chr(10).join(self.layers)}\n")
        if self.crs_str.lower() not in self.crss[layer]:
            raise ValueError(f'WMS: Invalid crs string: {self.crs_str}'
                             + f'\nOptions for {layer}: \n'
                             + f'{chr(10).join(self.crss[layer])}\n')

    def segment_region_into_tiles(
        self,
        res: float
    ) -> List[Tuple[float, float, float, float]]:
        """ Returns list of bounds for tiles within the region """
        width = int(round((self.bnds[2] - self.bnds[0]) / res))
        height = int(round((self.bnds[3] - self.bnds[1]) / res))
        count_xtiles = int(width // self.max_gridsize) + 1
        count_ytiles = int(height // self.max_gridsize) + 1
        xbnds = np.linspace(self.bnds[0], self.bnds[2], count_xtiles + 1)
        ybnds = np.linspace(self.bnds[1], self.bnds[3], count_ytiles + 1)
        tiles_bnd_list = []
        for i in range(count_xtiles):
            for j in range(count_ytiles):
                k_bnd = [xbnds[i], ybnds[j], xbnds[i + 1], ybnds[j + 1]]
                tiles_bnd_list.append(k_bnd)
        return tiles_bnd_list

    def __download_tile_data(
        self,
        bnds_list: List[Tuple[float, float, float, float]],
        layer: str,
        res: float,
        fpath: str
    ) -> None:
        """ Download layer data for all rectangular tiles """
        out_dir = os.path.dirname(os.path.abspath(fpath))
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        print(f'WMS: Downloading data for {len(bnds_list)} tiles..')
        for k, bnds in enumerate(bnds_list):
            width = int(round((bnds[2] - bnds[0]) / res))
            height = int(round((bnds[3] - bnds[1]) / res))
            kk = 0
            while True:
                try:
                    imap = self.wms.getmap(
                        layers=[layer],
                        srs=self.crs_str,
                        bbox=bnds,
                        format=self.fmt,
                        size=(width, height)
                    )
                    with open(self.get_wms_file_path(out_dir, k), 'wb') as out_file:
                        out_file.write(imap.read())
                except Exception as _:
                    if kk < 3:
                        kk += 1
                        continue
                    else:
                        raise Exception(
                            f'WMS: Connection issues! Try again') from None
                else:
                    break
            # except Exception as _:
            #     raise Exception(f'WMS: Connection issues! Try again') from None

    def __merge_tile_data(
        self,
        bnds_list: List[Tuple[float, float, float, float]],
        fpath: str,
        cleanup: bool = True
    ):
        """ Merging saved tile data """

        # load the raw tile data
        out_dir = os.path.dirname(os.path.abspath(fpath))
        src_files_to_mosaic = []
        for k, _ in enumerate(bnds_list):
            src = rs.open(self.get_wms_file_path(out_dir, k))
            src_files_to_mosaic.append(src)

        # merge the tile data
        mosaic, out_trans = merge(src_files_to_mosaic)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": mosaic.shape[1],
                         "width": mosaic.shape[2],
                         "transform": out_trans,
                         "crs": self.crs_str
                         })
        #print(f'WMS: Shape is ({mosaic.shape[1]},{mosaic.shape[2]})')
        with rs.open(fpath, "w", **out_meta) as dest:
            dest.write(mosaic)

        # close the opened raster files and do cleanup
        for src in src_files_to_mosaic:
            src.close()
            if cleanup:
                try:
                    os.remove(src.name)
                except OSError as e_obj:
                    if e_obj.errno != errno.ENOENT:
                        raise

    def download_raster(
            self,
            layer: str,
            res: float,
            fpath: str
    ) -> None:
        """ Download raster data from WMS server """

        self.validate_layer(layer)
        max_res = min(self.bnds[2] - self.bnds[0], self.bnds[3] - self.bnds[1])
        if not 0. < res < max_res:
            raise ValueError(f'WMS: Invalid resolution {res} for bnds'
                             + f'{self.bnds} in crs {self.crs_str}\n')

        bnds_list = self.segment_region_into_tiles(res)
        self.__download_tile_data(bnds_list, layer, res, fpath)
        self.__merge_tile_data(bnds_list, fpath)

    @classmethod
    def get_wms_file_path(cls, out_dir: str, k: int):
        """ returns file name for saving GeoTif tile data """
        return os.path.join(out_dir, f'wms_raw_{k}.tif')
