# 3DBuildingInfoMap

Deep-learning-based simultaneous extraction of building height and footprint from Sentinel imagery: development, evaluation and application

## Task Definition

This project focuses on patch-level building height and footprint mapping from Sentinel imagery.

### Boundary Definition

For cities not located in China, we use the administrative boundary provided by [GADM](https://gadm.org/index.html), which are in Shapefile format. The name of city is stored in *NAME_2* field. A simple GDAL command can help us extract the boundary of a specific city from the whole Shapefile 
```
ogr2ogr -where "NAME_2 = 'CityName'" city_boundary.shp gadm36.shp
```
For cities located in China, we use the administrative boundary from [DATAV.GeoAtlas](http://datav.aliyun.com/tools/atlas/), which are in GeoJSON format. <br>
We would also summarized our study area in Tsinghua Cloud: [link](https://cloud.tsinghua.edu.cn/d/1ec6ac5009f042e181d3/).

For practical use, the projection of these Shapefile could be unified into WGS84 via a GDAL command as follows
```
ogr2ogr output.shp -t_srs "EPSG:4326" input.shp
```

### Reference Data

The reference building data are mainly from [ArcGIS Hub](https://hub.arcgis.com/) and they are expected to be in the form of ESRI Shapefile.

## Dataset Preparation

### Vector Rasterization

The reference raw vector datasets are expected to be in the form of ESRI Shapefile, which needs to be rasterized into GeoTiff format before further operation. Use "FishNet+Intersection" tools in common open-source GIS can help achieve this functionality, see examples from QGIS: [FishNet](https://docs.qgis.org/3.10/en/docs/user_manual/processing_algs/qgis/vectorcreation.html#create-grid), [Intersection](https://docs.qgis.org/3.10/en/docs/user_manual/processing_algs/qgis/vectoroverlay.html#intersection)

We provide two functions `GetHeightFromCSV` and `GetFootprintFromCSV` in the Python script `gdal_ops.py` under the `utils` directory, which allows accelerating rasterization process by using multi-core CPUs.

The rasterized results under different resolution can be downloaded from this [link](https://drive.google.com/drive/folders/1nQhXijvLe90ImiToA5PD_IiK_N2vcu0n?usp=sharing).

### Dataset Storage

To simplify data retrieval, we further store and transform the input patches for model training, validation and test into three formats mainly:

- HDF5 (`.h5`): input patches are organized into different groups based on city location and the constraint which specifies maximum number of patches under one group. See `dataset.py` for details.

- NumPy array (`.npy`): since the input of ML models is actually a N-dimensional feature vector, we store the features and corresponding target value into `.npy` files after initial computation for further reuse. See `train.py` for details.

- Lightning Memory-Mapped Database (`.lmdb`): considering that HDF5 dataset may have some [concurrency issues](https://stackoverflow.com/questions/46045512/h5py-hdf5-database-randomly-returning-nans-and-near-very-small-data-with-multi/52249344#52249344) with PyTorch, we transform original `.h5` files into `.lmdb` files. See `DL_dataset.py` for details.

## Model Training

This project compares the performance of machine-learning-based (ML) models and deep-learning-based (DL) models on building footprint and height mapping.

For ML models, we select Random Forest Regression (RFR), Supporting Vector Regression (SVR) and its bagging version (BaggingSVR), XGBoost Regression (XGBoostR). See ML model implementation in `model.py` and ML model training in `train.py`.

For DL models, we select ResNet, SENet and CBAM as backbones and further compare model performance under Single-Task Learning (STL) and Multi-Task Learning (MTL). See DL model implementation in `DL_model.py` and DL model training in `DL_train.py`.

## Results

Pretrained ML models for building height and footprint prediction can be downloaded from this [link](https://drive.google.com/drive/folders/1YdK3DCDVcsCdUv2A0Pf6lMTl3q6yZTJD?usp=sharing) on Google Drive.

Pretrained DL models for building height and footprint prediction can be downloaded from this [link](https://drive.google.com/drive/folders/1s7c3GxJfLdQ_QICtSbkStSX_3cIHD6Al?usp=sharing) on Google Drive.

Mapping results of Glasgow, Chicago and Beijing be downloaded from this [link](https://drive.google.com/drive/folders/1b7VH8jMa2kiLdR2rt-2voHIK8aNc6giO?usp=sharing) on Google Drive.

## Inference

To predict building footprint and height for other area, we should:

1. prepare Sentinel-1/2's images as input.

    Sentinel-1/2's images can be downloaded from Google Earth Engine using a shell script `ee_download.sh` under the `utils` directory.

    Here we are expected to prepare a sheet in `.csv` format for city name, city boundary and representative year specification. See `GEE_Download_2021.csv` under the `utils` directory for examples.

    Note that according to the [guidance](https://developers.google.com/earth-engine/guides/exporting) for exporting data from Google Earth Engine, we can not export data to any local devices directly. Thus, Google Drive is recommended as a destination where data are export and then we can download exported data to our local devices.

2. make building footprint and height predictions.

    Predictions can be executed using shell scripts. We offer two shell script examples `infer_STL.sh` and `infer_MTL.sh` for the usage of STL models and Multi-Task Learning models, respectively.
