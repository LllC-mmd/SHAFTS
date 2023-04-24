# SHAFTS
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7717080.svg)](https://doi.org/10.5281/zenodo.7717080)


SHAFTS is a deep-learning-based Python package for **S**imultaneous extraction of building **H**eight **A**nd **F**ootprin**T** from **S**entinel Imagery

More details can be found in [the model descrition paper](https://gmd.copernicus.org/articles/16/751/2023/gmd-16-751-2023.html).

## Package Description

This project focuses on patch-level building height and footprint mapping from Sentinel imagery. **SHAFT** is an abbreviation for **S**imultaneous building **H**eight **A**nd **F**ootprin**T** extraction from **S**entinel Imagery.

### Installation

SHAFT requires 64-bit `python` 3.7+ and can be installed with `pip` in command line prompt:

```
python3 -m pip install shaft --upgrade
```

We recommend users to install `gdal>=3.2.0` using `conda` first.
Otherwise, installation may raise error about the environment variables of `gdal`.

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

### Building Height and Footprint prediction

After preparing above necessary images, building height and footprint information can be predicted by:

- `pred_height_from_tiff_DL_patch`: using deep-learning-based (DL) models trained by Single-Task-Learning (STL).

- `pred_height_from_tiff_DL_patch_MTL`: using deep-learning-based (DL) models trained by Multi-Task-Learning (MTL).

Since the total amount of relevant parameter settings are relatively more than data downloading, potential users can ref to the sample script for prediction named `minimum_case_run.py` under the `example` directory.
If batch processing is desired, users can ref to the sample script for prediction named `case_run.py` under the `example` directory.

Here, we offer pretrained DL models (based on [PyTorch](https://pytorch.org/)) for building height and footprint prediction via the [link](https://drive.google.com/drive/folders/148KZKDVOHOh6VOlZ9bqLby2twQd4UcM9?usp=sharing) on Google Drive. All of pretrained DL models are stored as `checkpoint.pth.tar`.

Note that for each target resolution, we can use STL/MTL models with(out) SRTM data to make predictions:

- For STL models, models with SRTM data are stored under `experiment_1` and models without SRTM data are stored under `experiment_2`.

- For MTL models, models with SRTM data are stored under `experiment_1` and models without SRTM data are stored under `experiment_2`.
Since MTL models can give both building height and footprint predictions, we only offer full sets of MTL models under the directory of models for building height prediction named `height`.

Note that all of models offered in the above link requires SRTM images as one of input variables, though more pretrained DL models during package development are collected for performance comparison.

### Integration with Google Cloud Ecosystem

If users want to generate building height and footprint maps without downloading Sentinel data to a local machine, SHAFTS offers the function named `GBuildingMap` which streamlines the workflow of satellite image preprocessing, TFRecord-based dataset management and DL model inference.

An example usage can be given as follows:

```python {cmd}
from shaft import GBuildingMap

# ---specify the mapping extent by the minimum/maximum of longitude and latitude
lon_min = -0.50
lat_min = 51.00
lon_max = 0.4
lat_max = 51.90

# ---specify the year (ranging between 2018 and 2022)
year = 2020

# ---specify the path to the pretrained SHAFTS's Tensorflow-based models
pretrained_weight = './dl-models/height/check_pt_senet_100m_MTL_TF_gpu'

# ---specify the output folder for storing building height and footprint maps
output_folder = './results'

# ---specify the Google Cloud Service configuration
GCS_config = {
    'SERVICE_ACCOUNT': '***** Google Cloud Service Account Name *****',
    'GS_ACCOUNT_JSON': '***** Parh to Google Cloud Service Account Credential *****',
    'BUCKET': '***** name of the bucket set for dataset exporting in Google Cloud Storage *****',
    'DATA_FOLDER': '*****  name of the folder which stores the exported dataset under the `BUCKET` *****',
}

# ---launch building height and footprint mapping
GBuildingMap(
    lon_min,
    lat_min,
    lon_max,
    lat_max,
    year,
    dx=0.09,
    dy=0.09,
    precision=3,
    batch_size=256,
    pretrained_model=pretrained_weight,
    GCS_config=GCS_config,
    target_resolution=100,
    num_task_queue=30,
    num_queue_min=2,
    file_prefix='_',
    padding=0.01,
    patch_size_ratio=1,
    s2_cloud_prob_threshold=20,
    s2_cloud_prob_max=80,
    MTL=True,
    removed=True,
    output_folder=output_folder,
)
```

The execution of this function requires following pre-steps:

1. Install and initialize the `gcloud` command-line interface. Users can ref to this [link](https://cloud.google.com/sdk/docs/install-sdk) for details.

2. Create a [Google Cloud Project](https://cloud.google.com/resource-manager/docs/creating-managing-projects) in the [Google Cloud console](https://console.cloud.google.com/cloud-resource-manager) for building height and footprint prediction.

3. Enable the [Earth Engine API](https://console.cloud.google.com/apis/library/earthengine.googleapis.com) for the project.

4. Set up a bucket in the [Google Cloud Storage (GCS)](https://cloud.google.com/storage) prepared for the storage of some intermediate exported datasets from Google Earth Engine. Please note that **the names of the created bucket, its folder for storing intermediate datasets** correspond to the `BUCKET` and `DATA_FOLDER` in the `GCS_config` required by the `GBuildingMap` function. An example of the structure of the GCS's bucket can be given as follows:

```bash
    BUCKET/
    |-- DATA_FOLDER/
    |   |-- tmp-exported-dataset.tfrecord.gz
    |   |-- ...
```

5. Create a [Google Cloud Service's account](https://console.cloud.google.com/iam-admin/serviceaccounts/) for the project. If there is already an account, you can keep it without creating an additional one. Please note that **the e-mail name of the service account** corresponds to the `SERVICE_ACCOUNT` in the `GCS_config` required by the `GBuildingMap` function.

6. Create a private key in the format of JSON for the service account by clicking the menu for that account via **:** > **key** > **JSON**. Please download the JSON key file locally and the **path to the JSON key for the service account** corresponds to the `GS_ACCOUNT_JSON` in the `GCS_config` required by the `GBuildingMap` function.

Here we should note that

- The pretrained models should be based on [Tensorflow](https://www.tensorflow.org/install)'s implementation. And we offer pretrained MTDL models for building height and footprint mapping at the resolution of 100 m via the [link](https://drive.google.com/drive/folders/1ziJzhrk6w9D9Q3uruCbq-lXGOdYw_haV?usp=sharing) on Google Drive. If your system has CUDA-supported GPUs, please download `check_pt_senet_100m_MTL_TF_gpu`. Otherwise, please download `check_pt_senet_100m_MTL_TF`.

- Based on preliminary tests, building height and footprint mapping for a area of $0.9^\circ\times 0.9^\circ$ might take 20-40 minutes where the majority of time is spent on exporting satellite images. So please control the size of target areas when you are using a laptop for this functionality.
