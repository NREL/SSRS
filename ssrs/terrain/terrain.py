""" Module for downlading terrain features within a rectangular region
from USGS's 3DEP or NASA's SRTM dataset"""

import os
import errno
from typing import Tuple, Union, List
import rasterio as rs
from .srtm import SRTM
from .threedep import ThreeDEP


class Terrain:
    """ Class for downloading terrain features in GeoTiff file for a
    given rectangular region defined by bounds in lon/lat coordinate system

    Parameters:
    ----------
    lonlat_bounds: Tuple[float, float, float, float]
        Defines the bounds = (min_lon, min_lat, max_lon, max_lat) of the
        rectangular terrain region in lon/lat crs
    out_dir: string
        Directory where raster data is saved and read from
    """

    valid_layers = ThreeDEP.valid_layers + SRTM.valid_layers

    def __init__(
            self,
            lonlat_bounds: Tuple[float, float, float, float],
            out_dir: str,
            verbose: bool = True
    ):

        self.lonlat_bounds = lonlat_bounds
        self.out_dir = out_dir
        self.verbose = verbose
        makedir_if_not_exists(self.out_dir)
        ilist = [round(ix, 2) for ix in lonlat_bounds]
        self.printit(f'Bounds set to {ilist}')
        # self.downloaded = {}

    def get_raster_fpath(self, lyr: str) -> str:
        """ Get filename for saving a terrain layer """
        fname = f'{lyr.lower().replace(" ","_")}.tif'
        return os.path.join(self.out_dir, fname)

    def download(
        self,
        layers: Union[List[str], str],
        pad: float = 0.01
    ) -> None:
        """ Downloads the Geotiff data for the specific layer and saves it """
        layers = [layers] if isinstance(layers, str) else layers
        for layer in layers:
            self.validate_layer_name(layer)
            fpath = self.get_raster_fpath(layer)
            padding = [-pad, -pad, pad, pad]
            pad_bnds = [ix + iy for ix, iy in zip(self.lonlat_bounds, padding)]
            try:
                # print(f'Trying to load terrain layer {layer}...')
                self.validate_saved_layer_data(layer)
            except (FileNotFoundError, ValueError) as err:
                if isinstance(err, FileNotFoundError):
                    self.printit(f'Layer {layer} not found..')
                elif isinstance(err, ValueError):
                    self.printit(f'Layer {layer} found with invalid bounds..')
                if layer in ThreeDEP.valid_layers:
                    self.printit(f'Downloading {layer} from 3DEP..')
                    src_object = ThreeDEP(
                        layer=layer,
                        bnds=pad_bnds,
                        fpath=fpath,
                        verbose=self.verbose
                    )
                elif layer in SRTM.valid_layers:
                    self.printit(f'Downloading {layer} data from SRTM..')
                    src_object = SRTM(
                        layer=layer,
                        bnds=pad_bnds,
                        fpath=fpath
                    )
                src_object.download()
            else:
                self.printit(f'Found saved raster data for {layer}')

    def validate_layer_name(self, layer: str) -> None:
        """ check if layer name is valid """
        if layer not in self.valid_layers:
            raise ValueError(f'Terrain: Invalid layer name: {layer}\nOptions:'
                             + f'\n{chr(10).join(self.valid_layers)}')

    def validate_saved_layer_data(self, layer: str) -> None:
        """ Validate the saved layer data"""
        layerfile = self.get_raster_fpath(layer)
        if os.path.isfile(layerfile) is False:
            # layer <layer>.<format> not found
            raise FileNotFoundError

        with rs.open(layerfile) as src_img:
            src_bounds = src_img.bounds
            within_bounds = (
                (src_bounds[0] <= self.lonlat_bounds[0] <= src_bounds[2]) &
                (src_bounds[1] <= self.lonlat_bounds[1] <= src_bounds[3]) &
                (src_bounds[0] <= self.lonlat_bounds[2] <= src_bounds[2]) &
                (src_bounds[1] <= self.lonlat_bounds[3] <= src_bounds[3])
            )
            #print(within_bounds, flush=True)
            if not within_bounds:
                # layer found, but bounds are different
                raise ValueError

    def printit(self, istr: str):
        """Print function"""
        if self.verbose:
            print(f'{self.__class__.__name__}: {istr}', flush=True)
        # try:
        #     with rs.open(self.get_raster_fpath(layer)) as src_img:
        #         src_bounds = src_img.bounds
        #     if not (
        #         (src_bounds[0] <= self.lonlat_bounds[0] <= src_bounds[2]) &
        #         (src_bounds[1] <= self.lonlat_bounds[1] <= src_bounds[3]) &
        #         (src_bounds[0] <= self.lonlat_bounds[2] <= src_bounds[2]) &
        #         (src_bounds[1] <= self.lonlat_bounds[3] <= src_bounds[3])
        #     ):
        #         raise FileNotFoundError
        # except rs.errors.RasterioIOError:
        #     # layer <layer>.<format> not found
        #     raise FileNotFoundError from None


def makedir_if_not_exists(dirname: str) -> None:
    """ Create the directory if it does not exists"""
    try:
        os.makedirs(dirname)
    except OSError as e_name:
        if e_name.errno != errno.EEXIST:
            raise


# JUNK
    # def update_registry(self, layer: str) -> None:
    #     """ Update the registry of what has been downloaded so far """
    #     fpath = os.path.join(self.out_dir, self.get_filename(layer))
    #     self.downloaded = {key: val for key,
    #                        val in self.downloaded.items() if val != fpath}
    #     self.downloaded[layer] = fpath

    # def get_all_registered_layers_in_projected_crs(
    #     self,
    #     proj_bounds: Tuple[float, float, float, float],
    #     proj_crs_string: str,
    #     resolution: Union[float, Tuple[float, float]]
    # ):
    #     """ Compute projected data for all the downloaded rasters"""
    #     print('Terrain: Reprojecting layers', end=" ")
    #     proj_data = {}
    #     ibounds = proj_bounds
    #     print(self.downloaded)
    #     for ilayer, ifpath in self.downloaded.items():
    #         print(ilayer, end=", ")
    #         idata, ibounds = get_raster_data_in_proj_crs(
    #             ifpath, proj_bounds, resolution, proj_crs_string)
    #         proj_data[ilayer] = idata
    #     print('done.', flush=True)
    #     return proj_data, ibounds

    # def get_layer_in_projected_crs(
    #     self,
    #     layer: str,
    #     proj_bounds: Tuple[float, float, float, float],
    #     proj_gridsize: Tuple[int, int],
    #     proj_res: float,
    #     proj_crs: str
    # ):
    #     """ Get a specific layer in projected crs """
    #     self.validate_layer_name(layer)
    #     try:
    #         self.validate_saved_layer_data(layer)
    #     except FileNotFoundError:
    #         print('Terrain: Need to download first!')
    #         self.download(layer)
    #     lyrdata = get_raster_in_projected_crs(
    #         self.get_raster_fpath(layer),
    #         [proj_bounds[3], proj_bounds[0]],
    #         proj_gridsize,
    #         proj_res,
    #         proj_crs
    #     )
    #     return lyrdata
