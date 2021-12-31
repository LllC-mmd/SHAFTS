# 3DBuildingInfoMap

Deep-learning-based simultaneous extraction of building height and footprint from Sentinel imagery: development, evaluation and application

## Package Description

This project focuses on patch-level building height and footprint mapping from Sentinel imagery. **SHAFT** is an abbreviation for ***S**imultaneous building **H**eight **A**nd **F**ootprin**T** extraction from Sentinel Imagery*.

### Installation

SHAFT requires 64-bit `python` 3.9+ and can be installed with `pip` in command line prompt:

```
python3 -m pip install shaft --upgrade
```

### Data Download

The input data of SHAFT may include:

- [Sentinel-1](https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-1): VH band, VV band.

- [Sentinel-2](https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-2): RGB band, NIR band.

- [SRTM](https://www2.jpl.nasa.gov/srtm/): DEM (optional). 

SHAFT contains some functions which can download above data directly from [Google Earth Engine](https://earthengine.google.com/). 

Note that according to the [guidance](https://developers.google.com/earth-engine/guides/exporting) for exporting data from Google Earth Engine, we can not export data to any local devices directly. Thus, Google Drive is recommended as a destination where data are export and then we can download exported data to our local devices.

An example for downloading Sentinel-2's image via `sentinel2_download_by_extent` is given as follows:

```python {cmd}
from shaft.utils.GEE_ops import sentinel2_download_by_extent

# ---specify the spatial extent and year for Sentinel-2's images
lon_min = -87.740
lat_min = 41.733
lon_max = -87.545
lat_max = 41.996
year = 2018

# ---define the output settings
dst = "Drive"
dst_dir = "Sentinel-2_export"
file_name = "Chicago_2018_sentinel_2.tif"

# ---start data downloading
sentinel2_download_by_extent(lon_min=lon_min, lat_min=lat_min, lon_max=lon_max, lat_max=lat_max,
                                year=year, dst_dir=dst_dir, file_name=file_name, dst=dst)
```

Also, SHAFT gives functions such as `sentinel1_download`, `sentinel2_download` and `srtm_download` to download images in a batch way by a `.csv` file.

## Building Height and Footprint prediction

After preparing above necessary images, building height and footprint information can be predicted by:

- `pred_height_from_tiff_DL_patch`: using deep-learning-based (DL) models trained by Single-Task-Learning (STL).

- `pred_height_from_tiff_DL_patch_MTL`: using deep-learning-based (DL) models trained by Multi-Task-Learning (MTL).

Since the total amount of relevant parameter settings are relatively more than data downloading, a sample script for prediction can be found under the `example` directory.

## Pretrained DL models

Pretrained DL models named for building height and footprint prediction can be downloaded from this [link](https://drive.google.com/drive/folders/19FNXK6N3-nWfHJZPgUOyltdhBIdJlLLo?usp=sharing) on Google Drive. All of pretrained DL models are stored as `checkpoint.pth.tar`.

Note that all of models offered in the above link requires SRTM images as one of input variables, though more pretrained DL models during package development are collected for performance comparison.