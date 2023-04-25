import ee
from typing import Union, List, Dict
import datetime
import re
from osgeo import gdal
import time
import ee
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import rasterio
import rasterio.transform as rtransform

from google.cloud import storage

from .DL_model_tf import *


# FEATURES = ["VV_p50", "VH_p50", "B4", "B3", "B2", "B8", "elevation"]
FEATURES = ["VV", "VH", "B4", "B3", "B2", "B8", "elevation"]
cloudFreeKeepThresh = 2
SR_BAND_SCALE = 1e4
cloudScale = 30


def add_cloud_bands_infer(img: ee.Image, cloud_prob_threshold=50):
    cloud_prob = ee.Image(img.get("s2cloudless")).select("probability").rename("cloudScore")
    # ------1 for cloudy pixels and 0 for non-cloudy pixels
    is_cloud = cloud_prob.gt(cloud_prob_threshold).rename("cloudFlag")

    return img.addBands(ee.Image([cloud_prob, is_cloud]))


def computeQualityScore_infer(img: ee.Image):
    # ------QualityScore is calculated by selecting the maximum value between the Cloud Score and Shadow Score for each pixel
    score = img.select(['cloudScore']).max(img.select(['shadowFlag']))

    score = score.reproject('EPSG:4326', None, 30).reduceNeighborhood(reducer=ee.Reducer.mean(), kernel=ee.Kernel.square(1))

    score = score.multiply(-1)

    return img.addBands(score.rename('cloudShadowScore'))


# ------recommended for exporting cloud-free Sentinel-2's images during inference
def add_shadow_bands_infer(img: ee.Image, nir_dark_threshold=0.15, cloud_proj_distance=1.5):
    # ------identify the water pixels from the SCL band
    not_water = img.select("SCL").neq(6)

    # ------identify dark NIR pixels that are not water (potential cloud shadow pixels)
    dark_pixels = img.select("B8").lt(nir_dark_threshold * SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')

    # ------determine the direction to project cloud shadow from clouds
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))

    # ------project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    # ---------`directionalDistanceTransform` function is used to calculate the distance between 
    # ---------the nearest cloudy pixels and current non-cloudy pixels along `shadow_azimuth` (<= CLD_PRJ_DIST*10)
    cld_proj = img.select('cloudFlag').directionalDistanceTransform(shadow_azimuth, cloud_proj_distance * 10) \
                .reproject(**{'crs': img.select(0).projection(), 'scale': 100}) \
                .select('distance') \
                .mask() \
                .rename('cloud_transform')

    # ------identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('shadowFlag')

    # ------add dark pixels, cloud projection, and identified shadows as image bands.
    img = img.addBands(ee.Image([cld_proj, shadows]))

    return img


# ------recommended for exporting cloud-free Sentinel-2's images during inference
def add_cloud_shadow_mask_infer(img: ee.Image, cloud_prob_threshold=50):
    img_cloud = add_cloud_bands_infer(img, cloud_prob_threshold)
    img_cloud_shadow = add_shadow_bands_infer(img_cloud)

    #roi = ee.Geometry(img.get('ROI'))
    imgPoly = ee.Algorithms.GeometryConstructors.Polygon(
            ee.Geometry(img.get("system:footprint")).coordinates()
    )

    cloudMask = img_cloud_shadow.select("cloudFlag").add(img_cloud_shadow.select("shadowFlag")).gt(0)
    cloudMask = (cloudMask.focalMin(1.5).focalMax(1.5).reproject(**{'crs': img.select([0]).projection(), 'scale': cloudScale}).rename('cloudmask'))
    cloudAreaImg = cloudMask.multiply(ee.Image.pixelArea())

    stats = cloudAreaImg.reduceRegion(reducer=ee.Reducer.sum(), scale=cloudScale, maxPixels=1e14)

    cloudPercent = ee.Number(stats.get('cloudmask')).divide(imgPoly.area()).multiply(100)  
    img_cloud_shadow = img_cloud_shadow.set('CLOUDY_PERCENTAGE', cloudPercent)

    return img_cloud_shadow


def export_satData_GEE(DEM_dta: ee.Image, S1_dataset: ee.ImageCollection, S2_dataset: List[ee.ImageCollection], S2_cloudProb_dataset: ee.ImageCollection, lon_min: Union[int, float], lat_min: Union[int, float], lon_max: Union[int, float], lat_max: Union[int, float], year: int, target_resolution: int, GCS_config: Dict[str, str], dst_dir: str, precision=2, destination="CloudStorage", file_prefix=None, padding=0.02, patch_size_ratio=1, s2_cloud_prob_threshold=20, s2_cloud_prob_max=80):
    """Export satellite images and data to Google Cloud Storage (GCS).

    Parameters
    ----------

    lon_min : float
        Minimum longitude of target region.
    lat_min : float
        Minimum latitude of target region.
    lon_max : float
        Maximum longitude of target region.
    lat_max : float
        Maximum latitude of target region.
    year : int
        Year of satellite images to be downloaded.
    dst_dir : str
        Directory on the destination device for saving the output image.
    file_prefix : str
        Name of the output image.
    padding : float
        Padding size outside the target region (in degrees).
        The default is `0.04`.

    """
    BUCKET = GCS_config["BUCKET"]
    DATA_FOLDER = GCS_config["DATA_FOLDER"]

    if any([not isinstance(lon_min, int), not isinstance(lon_max, int), not isinstance(lat_min, int), not isinstance(lat_max, int)]):
        lon_min_str = str(round(lon_min, precision))
        lat_min_str = str(round(lat_min, precision))
        lon_max_str = str(round(lon_max, precision))
        lat_max_str = str(round(lat_max, precision))
        file_suffix = "_".join([lon_min_str, lon_max_str, lat_min_str, lat_max_str])
    else:
        file_suffix = "_".join([str(lon_min), str(lon_max_str), str(lat_min_str), str(lat_max_str)])

    point_left_top = [lon_min, lat_max]
    point_right_top = [lon_max, lat_max]
    point_right_bottom = [lon_max, lat_min]
    point_left_bottom = [lon_min, lat_min]

    dx = padding
    dy = padding
    point_left_top = (point_left_top[0]-dx, point_left_top[1]+dy)
    point_right_top = (point_right_top[0]+dx, point_right_top[1]+dy)
    point_right_bottom = (point_right_bottom[0]+dx, point_right_bottom[1]-dy)
    point_left_bottom = (point_left_bottom[0]-dx, point_left_bottom[1]-dy)
    target_bd = ee.Geometry.Polygon([point_left_top, point_right_top, point_right_bottom, point_left_bottom, point_left_top])

    # ------set the sample point
    spc = degree_ref[target_resolution]
    '''
    xloc = np.arange(lon_min + spc * 0.5, lon_max, spc)
    yloc = np.arange(lat_max - spc * 0.5, lat_min, -spc)
    x, y = np.meshgrid(xloc, yloc)
    xy_coords = np.concatenate([x.flatten().reshape(-1, 1), y.flatten().reshape(-1, 1)], axis=1)
    target_pt = ee.FeatureCollection([ee.Geometry.Point(cr[0], cr[1]) for cr in xy_coords])
    '''
    xloc = ee.List.sequence(lon_min + spc * 0.5, lon_max, spc)
    yloc = ee.List.sequence(lat_max - spc * 0.5, lat_min, -spc)
    xy_coords = yloc.map(lambda y: xloc.map(lambda x: ee.Feature(ee.Geometry.Point([x, y])))).flatten()
    target_pt = ee.FeatureCollection(xy_coords)

    # ------set the name of the exported TFRecords file
    if file_prefix is not None:
        records_name = file_prefix + "_" + file_suffix + "_" + "H{0}".format(100) + "W{0}".format(100)
    else:
        records_name = "_" + file_suffix + "_" + "H{0}".format(100) + "W{0}".format(100)

    # ------filter the NASADEM data
    # DEM_dta = ee.Image("NASA/NASADEM_HGT/001").select(["elevation"]).float()
    DEM_dta = DEM_dta.clip(target_bd)
    
    # ------filter the Sentinel-1 data
    S1_dataset = S1_dataset.filterBounds(target_bd)
    # ------note that if we use difference aggregation ops, exported bands would have different suffix with "_p50"
    s1_img = S1_dataset.mean().clip(target_bd)

    # ------filter the Sentinel-2 cloud-free data
    s2_cloudless_col =  S2_cloudProb_dataset.filterBounds(target_bd)

    s2_cld_prob_step = 5
    cld_prb_set = np.arange(s2_cloud_prob_threshold, s2_cloud_prob_max + s2_cld_prob_step, s2_cld_prob_step)
    num_cld_prb = len(cld_prb_set)
    
    s2_img_found_flag = False
    s2_img_cloudFree = None
    for s2_ds_tmp in S2_dataset:
        if not s2_img_found_flag:
            s2_sr_col = s2_ds_tmp.filterBounds(target_bd)
            s2_combined_col = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
                    'primary': s2_sr_col,
                    'secondary': s2_cloudless_col,
                    'condition': ee.Filter.equals(**{
                        'leftField': 'system:index',
                        'rightField': 'system:index'
                    })
            }))
            s2_combined_col = s2_combined_col.map(lambda x: x.clip(target_bd))

            for cld_prb_id in range(0, num_cld_prb):
                if not s2_img_found_flag:
                    cld_prb_v = int(cld_prb_set[cld_prb_id])
                    s2_combined_col_tmp = s2_combined_col.map(lambda x: add_cloud_shadow_mask_infer(x, cld_prb_v)).map(computeQualityScore_infer)
                    s2_cloudless_col_BEST = s2_combined_col_tmp.filter(ee.Filter.lt('CLOUDY_PERCENTAGE', cloudFreeKeepThresh))

                    if s2_cloudless_col_BEST.size().gt(0):
                        s2_img_found_flag = True
                        s2_cloudless_col_BEST = s2_cloudless_col_BEST.sort("CLOUDY_PERCENTAGE", False)
                        s2_filtered = s2_combined_col_tmp.qualityMosaic('cloudShadowScore')
                        newC = ee.ImageCollection.fromImages([s2_filtered, s2_cloudless_col_BEST.mosaic()])
                        s2_img_cloudFree = ee.Image(newC.mosaic()).clip(target_bd)
                else:
                    break
        else:
            break
    
    s2_img_cloudFree = s2_img_cloudFree.select(["B4", "B3", "B2", "B8"])

    kernelSize = int(target_resolution / 10.0)
    overlapSize = overlapSize_ref[target_resolution]
    sentinel_image = ee.Image.cat([s1_img, s2_img_cloudFree]).float()
    
    # ---------generate the neighborhood patches from Sentinel imagery
    sentinel_psize = kernelSize + overlapSize
    sentinel_value_default = ee.List.repeat(-1e4, sentinel_psize)
    sentinel_value_default = ee.List.repeat(sentinel_value_default, sentinel_psize)
    sentinel_kernel = ee.Kernel.fixed(sentinel_psize, sentinel_psize, sentinel_value_default)

    sentinel_patched = sentinel_image.neighborhoodToArray(sentinel_kernel)
    target_pt = sentinel_patched.sampleRegions(collection=target_pt, scale=10, geometries=True)
    
    # ---------generate the neighborhood patches from DEM data
    dem_psize = int(sentinel_psize * patch_size_ratio)
    dem_value_default = ee.List.repeat(-1e4, dem_psize)
    dem_value_default = ee.List.repeat(dem_value_default, dem_psize)
    dem_kernel = ee.Kernel.fixed(dem_psize, dem_psize, dem_value_default)
    
    dem_patched = DEM_dta.neighborhoodToArray(dem_kernel)
    target_pt = dem_patched.sampleRegions(collection=target_pt, scale=30)
    
    # ---------export center points for sampling patches
    '''
    shp_task_config = {
        "collection": target_pt,
        "description": records_name,
        "folder": dst_dir,
        "fileFormat": "SHP",
    }

    shp_task = ee.batch.Export.table.toDrive(**shp_task_config)
    shp_task.start()
    '''
    
    '''
    # ---------export the image where patches are sampled
    s1_img_task_config = {
        "image": s1_img.float(),
        "description": "S1" + records_name,
        "folder": dst_dir,
        "scale": 10,
        "maxPixels": 1e13,
        "crs": "EPSG:4326"
    }
    s1_img_task = ee.batch.Export.image.toDrive(**s1_img_task_config)
    s1_img_task.start()

    s2_img_task_config = {
        "image": s2_img_cloudFree.float(),
        "description": "S2" + records_name,
        "folder": dst_dir,
        "scale": 10,
        "maxPixels": 1e13,
        "crs": "EPSG:4326"
    }
    s2_img_task = ee.batch.Export.image.toDrive(**s2_img_task_config)
    s2_img_task.start()

    nasaDEM_img_task_config = {
        "image": DEM_dta.float(),
        "description": "nasaDEM" + records_name,
        "folder": dst_dir,
        "scale": 30,
        "maxPixels": 1e13,
        "crs": "EPSG:4326"
    }
    nasaDEM_img_task = ee.batch.Export.image.toDrive(**nasaDEM_img_task_config)
    nasaDEM_img_task.start()
    '''
    
    # ---------export the patched dataset into the format of TFRecord
    if destination == "Drive":
        records_task_config = {
            "collection": target_pt,
            "description": records_name,
            "folder": dst_dir,
            "fileFormat": "TFRecord",
        }
        records_task = ee.batch.Export.table.toDrive(**records_task_config)
        records_task.start()
    elif destination == "CloudStorage":
        records_task_config = {
            "collection": target_pt,
            "description": records_name,
            "bucket": BUCKET,
            "fileNamePrefix": DATA_FOLDER + "/" + records_name,
            "fileFormat": "TFRecord",
        }
        records_task = ee.batch.Export.table.toCloudStorage(**records_task_config)
        records_task.start()
    else:
        raise NotImplementedError("Unknown destination: {0}. Supported destination: ['Drive', 'CloudStorage']".format(destination))
    
    print("[Submit] TFRecords is to be exported at [{0}, {1}, {2}, {3}]".format(lon_min, lon_max, lat_min, lat_max))
    return records_task
    
    
def parseSatTFRecord(example_proto: tf.train.Example, target_resolution: int, patch_size_ratio=1):
    kernelSize = int(target_resolution / 10.0)
    overlapSize = overlapSize_ref[target_resolution]

    sentinel_psize = kernelSize + overlapSize
    dem_psize = int(sentinel_psize * patch_size_ratio)

    COLUMNS = [tf.io.FixedLenFeature(shape=[sentinel_psize, sentinel_psize], dtype=tf.float32) if k != "elevation" else tf.io.FixedLenFeature(shape=[dem_psize, dem_psize], dtype=tf.float32) for k in FEATURES]
    FEATURES_DICT = dict(zip(FEATURES, COLUMNS))
    featParsed_ref = tf.io.parse_single_example(example_proto, FEATURES_DICT)

    return featParsed_ref


def rgb_rescale_tf(dta: tf.Tensor, q1=0.98, q2=0.02, vmin=0.0, vmax=1.0):
    vmin = vmin * 1.0
    vmax = vmax * 1.0
    val_min = tf.experimental.numpy.min(dta, axis=(0, 1))
    val_min = tf.expand_dims(tf.expand_dims(val_min, axis=0), axis=0)

    val_high = tfp.stats.percentile(dta, q1 * 100, axis=[0, 1])
    # val_high = tf.expand_dims(tf.expand_dims(val_high, axis=0), axis=0)
    val_high = tf.reshape(val_high, shape=[1, 1, -1])
    val_low = tfp.stats.percentile(dta, q2 * 100, axis=[0, 1])
    # val_low = tf.expand_dims(tf.expand_dims(val_low, axis=0), axis=0)
    val_low = tf.reshape(val_low, shape=[1, 1, -1])
    
    dta_rescale = (dta - val_min) * (vmax - vmin) / (val_high - val_low) + vmin

    dta_clipped = tf.clip_by_value(dta_rescale, vmin, vmax)
    # ------set NaN value to be vmin
    dta_clipped = tf.where(~tf.math.is_finite(dta_clipped), vmin, dta_clipped)

    return dta_clipped


def get_backscatterCoef_tf(raw_s1_dta: tf.Tensor):
    # coef = tf.math.pow(10.0, raw_s1_dta / 10.0)
    # coef = tf.where(coef > 1.0, 1.0, coef)
    coef = tf.clip_by_value(raw_s1_dta, 0.0, 1.0)
    return coef


def toTupleImage(feature_dict: Dict[str, tf.Tensor]):
    featList = [feature_dict.get(k) for k in FEATURES]
    feat_stacked = tf.stack(featList, axis=0)
    # ------convert CHW to HWC
    feat_stacked = tf.transpose(feat_stacked, [1, 2, 0])

    # ------the y-axis of exported patches should be flipped
    feat_sentinel = tf.experimental.numpy.flip(feat_stacked[:, :, :-1], axis=0)
    # feat_sentinel = feat_stacked[:, :, :-1]
    # ---------Note that for SHAFTS v202203, the required data type of input Sentinel's bands are UINT8 -> tf.float32.
    feat_s1 = tf.cast(tf.cast(get_backscatterCoef_tf(feat_stacked[:, :, 0:2]) * 255, tf.uint8), tf.float32) / 255.0
    feat_s2 = tf.cast(tf.cast(rgb_rescale_tf(feat_sentinel[:, :, 2:], vmin=0, vmax=255), tf.uint8), tf.float32) / 255.0

    feat_aux = tf.expand_dims(feat_stacked[:, :, -1], -1)

    feat = tf.concat([feat_s1, feat_s2, feat_aux], axis=-1)

    return feat


def createInferDataset(recordPathList: List[str], target_resolution: int, batch_size=1) -> tf.data.TFRecordDataset:
    imgDataset = tf.data.TFRecordDataset(recordPathList, compression_type="GZIP")
    imgDataset = imgDataset.map(lambda x: parseSatTFRecord(x, target_resolution), num_parallel_calls=8)

    imgDataset = imgDataset.map(toTupleImage, num_parallel_calls=8).batch(batch_size)
    '''
    dta = list(imgDataset.as_numpy_iterator())
    print(dta[0][0][0, :, :, 0])
    print(dta[0][0].shape)
    plt.imshow(dta[0][0][0, :, :, 0], vmin=-11, vmax=-5, cmap="turbo")
    plt.show()
    '''
    print("Dataset parsed: {0}".format([" ".join(recordPathList)]))

    return imgDataset
    

def GBuildingMap(lon_min: Union[int, float], lat_min: Union[int, float], lon_max: Union[int, float], lat_max: Union[int, float], year: int, pretrained_model: Union[Dict[str, str], str], GCS_config: Dict[str, str], target_resolution: int, dx=0.09, dy=0.09, precision=2, batch_size=16, padding=0.02, patch_size_ratio=1, s2_cloud_prob_threshold=20, s2_cloud_prob_max=80, MTL=True, removed=True, num_task_queue=10, num_queue_min=2, file_prefix=None, output_folder="."):
    """Perform the 3D building information mapping using Google Cloud Service.

    Parameters
    ----------

    lon_min : Union[int, float]
        Minimum longitude of target mapping.
    lat_min : Union[int, float]
        Minimum latitude of target mapping.
    lon_max : Union[int, float]
        Maximum longitude of target mapping.
    lat_max : Union[int, float]
        Maximum latitude of target mapping.
    year : int
        Year of target mapping.
    pretrained_model : Union[Dict[str, str], str]
        Dictionary or string which specifies the path to pretrained models. An example for the dictionary is given as:
        {"height": "check_pt_senet_100m_MTL_TF_gpu", "footprint": "check_pt_senet_100m_MTL_TF_gpu"}
    GCS_config : Dict[str, str]
        Dictionary which specifies the configuration information of Google Cloud Service (GCS) for 3D building information mapping. An example is given as:
        GCS_config = {
            "SERVICE_ACCOUNT": "e-mail name of the GCS account",
            "GS_ACCOUNT_JSON": "path to the JSON key for the GCS service account",
            "BUCKET": "name of the Google Cloud Storage bucket",
            "DATA_FOLDER": "name of the folder under the Google Cloud Storage bucket prepared for storing intermediate datasets",
        }
    target_resolution : int
        Target resolution (in degrees) of the output mapping.
    dx : float
        Longitude coverage (in degrees) of sub-regions used for inference.
        The default is `0.09`.
    dy : float
        Latitude coverage (in degrees) of sub-regions used for inference.
        The default is `0.09`.
    precision : int
        The numerical precision of geographical coordinate calculation when dividing sub-regions.
        The default is `2`.
    batch_size : int
        Batch size for CNN' inference.
        The default is `16`.
    padding : float
        Padding size outside the target region (in degrees) when exporting satellite images.
        The default is `0.02`.
    patch_size_ratio : int
        The ratio between the patch size of Sentinel-1/2 and SRTM data.
        The default is `1`.
    s2_cloud_prob_threshold : int
        The minimum threshold of Sentinel-2 cloud probability for filtering out cloudy pixels.
        The default is `20`.
    s2_cloud_prob_max : int
        The maximum threshold of Sentinel-2 cloud probability for filtering out cloudy pixels.
        The default is `80`.
    MTL : bool
        Whether to use the MTL model.
        The default is `True`.
    removed : bool
        Whether to remove the temporary datasets exported from Google Earth Engine and stored in Google Cloud Storage.
        The default is `True`.
    num_task_queue : int
        The number of concurrent Google Earth Engine's tasks for exporting datasets in the task queue.
        The default is `10`.
    num_queue_min : int
        The minimum number of tasks in the task queue. When the number of tasks in the task queue is less than `num_queue_min`, new tasks will be added to the task queue.
        The default is `2`.
    file_prefix : str
        The prefix of the temporary dataset name.
        The default is `None`.
    output_folder : str
        The path to the folder which saves the building height and footprint files.
        The default is `.`.

    """
    BH_min = 2.0
    BH_max = 1000.0
    BF_min = 0.0
    BF_max = 1.0

    BASE_INTERVAL = 1.0
    MIN_INTERVAL = 1.5
    GAMMA = 0.9

    SERVICE_ACCOUNT = GCS_config["SERVICE_ACCOUNT"]
    GS_ACCOUNT_JSON = GCS_config["GS_ACCOUNT_JSON"]

    # ---Google Cloud Storage bucket into which prediction datset will be written
    BUCKET = GCS_config["BUCKET"]
    DATA_FOLDER = GCS_config["DATA_FOLDER"]
    
    # ------model configuration
    if MTL:
        # ------prepare the Tensorflow-based MTL models
        m = tf.keras.models.load_model(pretrained_model)
        print("Model loaded from: {0}".format(pretrained_model))
    else:
        # ------prepare the Tensorflow-based STL models
        m_footprint = tf.keras.models.load_model(pretrained_model["footprint"])
        print("Model loaded from: {0}".format(pretrained_model["footprint"]))

        m_height = tf.keras.models.load_model(pretrained_model["height"])
        print("Model loaded from: {0}".format(pretrained_model["height"]))

    # ------set a client for operations related to Google Cloud Storage
    # client_GCS = storage.Client(project=PROJECT_ID)
    client_GCS = storage.Client.from_service_account_json(GS_ACCOUNT_JSON)
    bucket_src = client_GCS.get_bucket(BUCKET)

    ee.Initialize(ee.ServiceAccountCredentials(SERVICE_ACCOUNT, GS_ACCOUNT_JSON))

    # ------load Sentinel-1 and Sentinel-2 dataset
    DEM_dta = ee.Image("NASA/NASADEM_HGT/001").select(["elevation"]).float()
    # ------load Sentinel-1 dataset
    s1_ds = ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT")
    s1_ds = s1_ds.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    s1_ds = s1_ds.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    s1_ds = s1_ds.filter(ee.Filter.eq('instrumentMode', 'IW'))
    s1_ds = s1_ds.select(["VV", "VH"])
    s1_year = min(2022, max(year, 2015))    # Sentinel-1 availability: from 2014-10 to 2022-12
    # ---------for northern hemisphere
    s1_time_start_N = datetime.datetime(year=s1_year, month=12, day=1).strftime("%Y-%m-%d")
    s1_time_end_N = datetime.datetime(year=s1_year+1, month=3, day=1).strftime("%Y-%m-%d")
    s1_ds_N = s1_ds.filter(ee.Filter.date(s1_time_start_N, s1_time_end_N))
    # ---------for southern hemisphere
    s1_time_start_S = datetime.datetime(year=s1_year, month=6, day=1).strftime("%Y-%m-%d")
    s1_time_end_S = datetime.datetime(year=s1_year, month=9, day=1).strftime("%Y-%m-%d")
    s1_ds_S = s1_ds.filter(ee.Filter.date(s1_time_start_S, s1_time_end_S))

    # ------load Sentinel-2 dataset
    s2_cld_threshold_base = 20
    s2_ds = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', s2_cld_threshold_base))
    s2_year = min(2022, max(year, 2018))    # Sentinel-2 availability: from 2017-03 to 2022-12
    # ---------for northern hemisphere, we prefer: autumn > spring > summer
    # ---------for southern hemisphere, we prefer: spring > autumn > summer
    s2_time_start_aut = datetime.datetime(year=s2_year, month=9, day=1).strftime("%Y-%m-%d")
    s2_time_end_aut = datetime.datetime(year=s2_year, month=12, day=1).strftime("%Y-%m-%d")
    s2_ds_aut = s2_ds.filter(ee.Filter.date(s2_time_start_aut, s2_time_end_aut))

    s2_time_start_spr = datetime.datetime(year=s2_year, month=3, day=1).strftime("%Y-%m-%d")
    s2_time_end_spr = datetime.datetime(year=s2_year, month=6, day=1).strftime("%Y-%m-%d")
    s2_ds_spr = s2_ds.filter(ee.Filter.date(s2_time_start_spr, s2_time_end_spr))
    '''
    s2_time_start_sum = datetime.datetime(year=s2_year, month=6, day=1).strftime("%Y-%m-%d")
    s2_time_end_sum = datetime.datetime(year=s2_year, month=9, day=1).strftime("%Y-%m-%d")
    s2_ds_sum = s2_ds.filter(ee.Filter.date(s2_time_start_sum, s2_time_end_sum))

    s2_time_start_win = datetime.datetime(year=s2_year, month=12, day=1).strftime("%Y-%m-%d")
    s2_time_end_win = datetime.datetime(year=s2_year+1, month=3, day=1).strftime("%Y-%m-%d")
    s2_ds_win = s2_ds.filter(ee.Filter.date(s2_time_start_win, s2_time_end_win))    
    '''
    s2_cloudProb_ds = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")

    # ------divide ROI into smaller patches
    num_lon = int(round((lon_max - lon_min) / dx, precision))
    if num_lon > 0:
        lon_min_sub = np.arange(lon_min, lon_min + num_lon * dx, dx)
    else:
        lon_min_sub = np.array([lon_min])
    xCell_num = len(lon_min_sub)

    num_lat = int(round((lat_max - lat_min) / dy, precision))
    if num_lat > 0:
        lat_max_sub = np.arange(lat_max, lat_max - num_lat * dy, -dy)
    else:
        lat_max_sub = np.array([lat_max])
    yCell_num = len(lat_max_sub)

    ee_taskList = []
    ee_coordsList = []
    # ee_suffixList = []
    for xId in range(0, xCell_num):
        for yId in range(0, yCell_num):
            lon_min_tmp = lon_min_sub[xId]
            lon_max_tmp = lon_min_tmp + dx
            lat_max_tmp = lat_max_sub[yId]
            lat_min_tmp = lat_max_tmp - dy

            # ------export patched datasets from Google Earth Engine (GEE) to Google Cloud Storage (GCS)
            if any([not isinstance(lon_min_tmp, int), not isinstance(lon_max_tmp, int), not isinstance(lat_min_tmp, int), not isinstance(lat_max_tmp, int)]):
                # on_min_f, lon_min_i = math.modf(lon_min_tmp)
                lon_min_str = str(round(lon_min_tmp, precision))
                # lat_min_f, lat_min_i = math.modf(lat_min_tmp)
                lat_min_str = str(round(lat_min_tmp, precision))
                # lon_max_f, lon_max_i = math.modf(lon_max_tmp)
                # lon_max_str = "%d.%d" % (lon_max_i, round(lon_max_f * 100))
                lon_max_str = str(round(lon_max_tmp, precision))
                # lat_max_f, lat_max_i = math.modf(lat_max_tmp)
                # lat_max_str = "%d.%d" % (lat_max_i, round(lat_max_f * 100))
                lat_max_str = str(round(lat_max_tmp, precision))
                file_suffix = "_".join([lon_min_str, lon_max_str, lat_min_str, lat_max_str])
            else:
                file_suffix = "_".join([str(lon_min_tmp), str(lon_max_tmp), str(lat_min_tmp), str(lat_max_tmp)])
            
            if 0.5 * (lat_max_tmp + lat_min_tmp) > 0:
                task_tmp = export_satData_GEE(DEM_dta, s1_ds_N, [s2_ds_aut, s2_ds_spr], s2_cloudProb_ds, GCS_config=GCS_config,
                                              lon_min=lon_min_tmp, lat_min=lat_min_tmp, lat_max=lat_max_tmp, lon_max=lon_max_tmp, year=year, target_resolution=target_resolution,
                                              dst_dir=DATA_FOLDER, precision=precision, destination="CloudStorage", file_prefix=file_prefix, padding=padding, patch_size_ratio=patch_size_ratio,
                                              s2_cloud_prob_threshold=s2_cloud_prob_threshold, s2_cloud_prob_max=s2_cloud_prob_max)
            else:
                task_tmp = export_satData_GEE(DEM_dta, s1_ds_S, [s2_ds_spr, s2_ds_aut], s2_cloudProb_ds, GCS_config=GCS_config,
                                              lon_min=lon_min_tmp, lat_min=lat_min_tmp, lat_max=lat_max_tmp, lon_max=lon_max_tmp, year=year, target_resolution=target_resolution,
                                              dst_dir=DATA_FOLDER, precision=precision, destination="CloudStorage", file_prefix=file_prefix, padding=padding, patch_size_ratio=patch_size_ratio,
                                              s2_cloud_prob_threshold=s2_cloud_prob_threshold, s2_cloud_prob_max=s2_cloud_prob_max)
            
            ee_taskList.append(task_tmp)
            ee_coordsList.append([lon_min_str, lon_max_str, lat_min_str, lat_max_str])

            if xId == xCell_num - 1 and yId == yCell_num - 1:
                final_flag = True
            else:
                final_flag = False

            if len(ee_taskList) >= num_task_queue or final_flag:
                task_infoList = [task.status() for task in ee_taskList]
                task_IDList = [tinfo["name"].split("/")[-1] for tinfo in task_infoList]
                task_stateList = [tinfo["state"] for tinfo in task_infoList]
                # task_msgList = ["" for task in task_infoList]
                task_runningID = [i for i in range(0, len(task_stateList)) if task_stateList[i] in ["UNSUBMITTED", "READY", "RUNNING", "CANCEL_REQUESTED"]]
                num_iter = 0
                while (len(task_runningID) > num_queue_min and not final_flag) or (len(task_runningID) > 0 and final_flag):
                    num_active = 0
                    task_finishedID = []
                    # ----only check the status of tasks which were previously active
                    print(task_runningID)
                    for loc in range(0, len(task_runningID)):
                        tid = task_runningID[loc]
                        tid_st = ee_taskList[loc].status()
                        # ------ref to: https://developers.google.com/earth-engine/guides/processing_environments
                        if tid_st["state"] not in ["UNSUBMITTED", "READY", "RUNNING", "CANCEL_REQUESTED"]:
                            task_finishedID.append(loc)
                            # task_stateList[tid] = tid_st["state"]
                            if tid_st["state"] != "COMPLETED":
                                # task_msgList[loc] = tid_st["error_message"]
                                print("*****[Error] TFRecords fails to be exported at [{0}] (TaskID = {1}) due to {2}*****".format(", ".join(ee_coordsList[loc]), task_IDList[tid], tid_st["error_message"]))
                            else:
                                print("*****[Success] TFRecords is exported at [{0}] (TaskID = {1})*****".format(", ".join(ee_coordsList[loc]), task_IDList[tid]))
                        else:
                            # print(tid_st["state"])
                            num_active += 1
                    
                    print(task_finishedID)
                    for loc in sorted(task_finishedID, reverse=True):
                        del task_runningID[loc]
                        del ee_taskList[loc]
                        del ee_coordsList[loc]
                    
                    if num_active <= 2:
                        t_sleep = MIN_INTERVAL
                    else:
                        t_sleep = max(BASE_INTERVAL * (GAMMA**num_iter) * num_active, MIN_INTERVAL)
                    print("[Iter. {0}] {1} tasks are still running. Sleep {2} s".format(num_iter, num_active, t_sleep))
                    num_iter += 1
                    time.sleep(t_sleep)
                '''
                for i in range(0, len(ee_taskList)):
                    if task_stateList[i] != "COMPLETED":
                        print("[Error] TFRecords fails to be exported at [{0}] (TaskID = {1}) due to {2}".format(", ".join(ee_coordsList[i]), task_IDList[i], task_msgList[i]))
                    else:
                        print("[Success] TFRecords is exported at [{0}] (TaskID = {1})".format(", ".join(ee_coordsList[i]), task_IDList[i]))
                '''

    for xId in range(0, xCell_num):
        for yId in range(0, yCell_num):
            lon_min_tmp = lon_min_sub[xId]
            lon_max_tmp = lon_min_tmp + dx
            lat_max_tmp = lat_max_sub[yId]
            lat_min_tmp = lat_max_tmp - dy

            # ------export patched datasets from Google Earth Engine (GEE) to Google Cloud Storage (GCS)
            if any([not isinstance(lon_min_tmp, int), not isinstance(lon_max_tmp, int), not isinstance(lat_min_tmp, int), not isinstance(lat_max_tmp, int)]):
                lon_min_str = str(round(lon_min_tmp, precision))
                lat_min_str = str(round(lat_min_tmp, precision))
                lon_max_str = str(round(lon_max_tmp, precision))
                lat_max_str = str(round(lat_max_tmp, precision))
                file_suffix = "_".join([lon_min_str, lon_max_str, lat_min_str, lat_max_str])
            else:
                file_suffix = "_".join([str(lon_min_tmp), str(lon_max_tmp), str(lat_min_tmp), str(lat_max_tmp)])

            # ------prepare the TFRecords dataset on GC
            fullPath_list = [f.name for f in bucket_src.list_blobs(prefix=DATA_FOLDER)]
            record_list = ["gs://" + BUCKET + "/" + f for f in fullPath_list if f.endswith(".tfrecord.gz") if file_suffix in f]

            if len(record_list) > 0:
                img_ds = createInferDataset(record_list, target_resolution, batch_size=batch_size)

                # ---------determine the information of dimensions of the dataset
                h, w = re.findall(pattern=r"H(\d+)W(\d+)", string=record_list[0])[0]
                h = int(h)
                w = int(w)
                
                # ------model configuration
                if MTL:
                    footprint_dta, height_dta = m.predict(x=img_ds, verbose=1)
                else:
                    height_dta = m_height.predict(x=img_ds, verbose=1)
                    footprint_dta = m_footprint.predict(x=img_ds, verbose=1)
                
                footprint_dta = np.reshape(footprint_dta, newshape=(h, w))
                footprint_dta = np.where(footprint_dta > BF_max, BF_max, footprint_dta)

                height_dta = np.reshape(height_dta, newshape=(h, w))
                height_dta = np.where(height_dta > BH_max, BH_max, height_dta)
                height_dta = np.where(height_dta < BH_min, BH_min, height_dta)

                # ------export predicted images to Google Cloud Storage
                output_geo_trans = rtransform.Affine(degree_ref[target_resolution], 0.0, lon_min_tmp, 0.0, -degree_ref[target_resolution], lat_max_tmp)

                # footprint_name = BF_OUTPUT_GCS_PREFIX + "/" + "BF" + "_" + file_suffix + ".tif"
                footprint_name = os.path.join(output_folder, "BF_" + file_suffix + ".tif")
                with rasterio.Env():
                    with rasterio.open(footprint_name, "w+", width=w, height=h, count=1, crs="EPSG:4326", transform=output_geo_trans, dtype="float32") as out:
                        out.write_band(1, footprint_dta)
                    print("*" * 10 + " Output BuildingFootprint File: {0} ".format(footprint_name) + "*" * 10)
                
                # height_name = BH_OUTPUT_GCS_PREFIX + "/" + "BH" + "_" + file_suffix + ".tif"
                height_name = os.path.join(output_folder, "BH_" + file_suffix + ".tif")
                # with rasterio.Env(GS_OAUTH2_PRIVATE_KEY=GS_OAUTH2_PRIVATE_KEY, GS_OAUTH2_CLIENT_EMAIL=GS_OAUTH2_CLIENT_EMAIL):
                with rasterio.Env():
                    with rasterio.open(height_name, "w+", width=w, height=h, count=1, crs="EPSG:4326", transform=output_geo_trans, dtype="float32") as out:
                        out.write_band(1, height_dta)
                    print("*" * 10 + " Output BuildingHeight File: {0} ".format(height_name) + "*" * 10)
                
                # ------delete temporary TFRcords files on the Google Cloud Storage
                if removed:
                    for blob in bucket_src.list_blobs(prefix=DATA_FOLDER):
                        if file_suffix in blob.name:
                            blob.delete()

    # ------merge the results of smaller patches
    if any([not isinstance(lon_min, int), not isinstance(lon_max, int), not isinstance(lat_min, int), not isinstance(lat_max, int)]):
        lon_min_str = str(round(lon_min, precision))
        lat_min_str = str(round(lat_min, precision))
        lon_max_str = str(round(lon_max, precision))
        lat_max_str = str(round(lat_max, precision))
        file_suffix = "_".join([lon_min_str, lon_max_str, lat_min_str, lat_max_str])
    else:
        file_suffix = "_".join([str(lon_min), str(lon_max), str(lat_min), str(lat_max)])
    
    height_target_file = os.path.join(output_folder, "BH_" + file_suffix + ".tif")
    height_subfile = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith(".tif")]
    gdal.Warp(destNameOrDestDS=height_target_file, srcDSOrSrcDSTab=height_subfile)
    for f in height_subfile:
        os.remove(f)

    footprint_target_file = os.path.join(output_folder, "BF_" + file_suffix + ".tif")
    footprint_subfile = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith(".tif")]
    gdal.Warp(destNameOrDestDS=footprint_target_file, srcDSOrSrcDSTab=footprint_subfile)
    for f in footprint_subfile:
        os.remove(f)
    

if __name__ == "__main__":
    # ---Pretrained SHAFTS's model specification
    pretrained_weight = './dl-models/height/check_pt_senet_100m_MTL_TF_gpu'

    # ---Google Cloud Service configuration
    GCS_config = {
        'SERVICE_ACCOUNT': '***** GCSAccountName *****',
        'GS_ACCOUNT_JSON': '***** GCSAccountCredential *****',
        'BUCKET': '***** name of the bucket set for dataset exporting in Google Cloud Storage *****',
        'DATA_FOLDER': '*****  name of the folder which stores the exported dataset under the `BUCKET` *****',
    }

    # ---Building height and footprint mapping

    GBuildingMap(
        lon_min=-0.50,
        lat_min=51.00,
        lon_max=0.4,
        lat_max=51.90,
        year=2020,
        dx=0.09,
        dy=0.09,
        precision=3,
        batch_size=512,
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
        output_folder='./results',
    )
