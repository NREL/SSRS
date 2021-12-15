""" Module for dealing with USGS's 3DEP dataset """

from typing import Tuple
from .wms import WMS


class ThreeDEP(WMS):
    """ Class for accessing 3DEP data from USGS through WMS

    Parameters:
    ----------
    layer: str
        Requested layer, should be among valid_layers
    bnds: Tuple[float, float, float, float]
        Bounds of the region in lat/lon coordinate system
        [min_lon, min_lat, max_lon, max_lat]
    fpath: str
        file path to save the raster data, should be .tif
    res: float, optional, defaults to 1/3 arc second = 1/3600/3 degrees
        Resolution in degrees for extracting the data. Can ask for even lower
        resolution but I think it will be reprojected in some way using the
        1/3 arc second data
    """
    wms_url = ('https://elevation.nationalmap.gov/arcgis/services/'
               '3DEPElevation/ImageServer/WMSServer')
    wms_version = '1.3.0'
    valid_layers = (
        "DEM",
        "Hillshade Gray",
        "Aspect Degrees",
        "Aspect Map",
        "GreyHillshade_elevationFill",
        "Hillshade Multidirectional",
        "Slope Map",
        "Slope Degrees",
        "Hillshade Elevation Tinted",
        "Height Ellipsoidal",
        "Contour 25",
        "Contour Smoothed 25",
    )
    crs_str = 'EPSG:4326'

    def __init__(
        self,
        layer: str,
        bnds: Tuple[float, float, float, float],
        fpath: str,
        res=1 / 3600. / 3.  # 1/3 arc second = 1/3600/3 degrees
    ):
        if layer in self.valid_layers:
            if layer == 'DEM':
                layer = 'None'
        else:
            raise ValueError(f'ThreeDEP:{layer} not a valid layer!\nOptions:' +
                             f'\n{chr(10).join(self.valid_layers)}\n')
        self.layer_name = f'3DEPElevation:{layer}'
        self.res = res
        self.fpath = fpath
        super().__init__(bnds, self.crs_str, self.wms_url, 2000)

    def download(self):
        """ Download the data """
        self.download_raster(self.layer_name, self.res, self.fpath)
