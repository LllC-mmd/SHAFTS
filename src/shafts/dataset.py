import os
import shutil
import re
import glob
import multiprocessing
from functools import reduce

import numpy as np
import h5py
import pandas as pd
import scipy.interpolate as interp

from osgeo import gdal

from .mathexpr import rgb_rescale_band


def fill_nan_nearest(dta):
    grid_y, grid_x = np.mgrid[0:dta.shape[0], 0:dta.shape[1]]
    pt_y, pt_x = np.where(~np.isnan(dta))
    pt_use = np.concatenate((np.reshape(pt_y, (-1, 1)), np.reshape(pt_x, (-1, 1))), axis=1)
    pt_val = dta[pt_y, pt_x]
    dta_filled = interp.griddata(pt_use, pt_val, (grid_y, grid_x), method="nearest")

    return dta_filled


def calc_nan_ratio(dta):
    nan_test = np.isnan(dta)
    num_nan = np.sum(nan_test.astype(int))
    nan_ratio = num_nan / (nan_test.shape[0] * nan_test.shape[1])

    return nan_ratio


def merge_dataset(dataset_list):
    if len(dataset_list) == 1:
        pass
    else:
        dst_db = h5py.File(dataset_list[0], "a")
        src_db_list = [h5py.File(ds, "r") for ds in dataset_list]
        for src_db in src_db_list:
            for k in src_db.keys():
                h5py.h5o.copy(src_db.id, bytes(k), dst_db, bytes(k))


def split_dataset(src_dataset_path, dataset_split_ratio=0.2, group_split_ratio=0.1, group_exclude=None):
    src_db = h5py.File(src_dataset_path, "r")
    src_dir = os.path.dirname(src_dataset_path)
    src_base = os.path.splitext(os.path.basename(src_dataset_path))[0]

    train_db_path = os.path.join(src_dir, src_base + "_train.h5")
    train_db = h5py.File(train_db_path, mode="w")
    valid_db_path = os.path.join(src_dir, src_base + "_valid.h5")
    valid_db = h5py.File(valid_db_path, mode="w")

    src_db_groups = list(src_db.keys())
    if group_exclude is not None:
        # ---------store the whole dataset of the excluded group into the validation set
        for g in group_exclude:
            src_db_groups.remove(g)
            valid_group_db = valid_db.create_group(g)
            for dataset_name in src_db[g].keys():
                dataset = src_db[g][dataset_name][()]
                valid_group_db.create_dataset(dataset_name, data=dataset)

    num_group_train = int(np.ceil(len(src_db_groups) * (1 - group_split_ratio)))
    group_train = np.random.choice(src_db_groups, size=num_group_train, replace=False)

    for group_name in src_db_groups:
        # ---------if the group is to be included in the training set,
        # ------------we divide its dataset into (1-dataset_split_ratio):dataset_split_ratio for training & validation
        if group_name in group_train:
            train_group_db = train_db.create_group(group_name)
            valid_group_db = valid_db.create_group(group_name)

            for _ in src_db[group_name].keys():
                num_sample = len(src_db[group_name][_])
                break

            num_sample_train = int(np.ceil(num_sample * (1 - dataset_split_ratio)))
            sample_train_id = np.random.choice(num_sample, size=num_sample_train, replace=False)
            sample_valid_id = np.setdiff1d(np.arange(num_sample), sample_train_id, assume_unique=True)

            for dataset_name in src_db[group_name].keys():
                dataset = src_db[group_name][dataset_name][()]
                train_group_db.create_dataset(dataset_name, data=dataset[sample_train_id])
                valid_group_db.create_dataset(dataset_name, data=dataset[sample_valid_id])
        # ---------if the group is to be included in the validation set,
        # ------------we store the whole dataset in it for validation
        else:
            valid_group_db = valid_db.create_group(group_name)
            for dataset_name in src_db[group_name].keys():
                dataset = src_db[group_name][dataset_name][()]
                valid_group_db.create_dataset(dataset_name, data=dataset)

    train_db.close()
    valid_db.close()
    src_db.close()


def sample_dataset(src_dataset_path, out_dataset_path, sample_ratio=0.05):
    src_db = h5py.File(src_dataset_path, "r")
    out_db = h5py.File(out_dataset_path, "w")

    for group_name in src_db.keys():
        out_group_db = out_db.create_group(group_name)

        for _ in src_db[group_name].keys():
            num_sample = len(src_db[group_name][_])
            break

        sample_id = np.random.choice(num_sample, size=int(num_sample*sample_ratio), replace=False)

        for dataset_name in src_db[group_name].keys():
            dataset = src_db[group_name][dataset_name][()]
            out_group_db.create_dataset(dataset_name, data=dataset[sample_id])

    src_db.close()
    out_db.close()


def get_dataset(src_dataset_path, target_group):
    src_db = h5py.File(src_dataset_path, "r")
    src_dir = os.path.dirname(src_dataset_path)
    src_base = os.path.splitext(os.path.basename(src_dataset_path))[0]

    out_db_path = os.path.join(src_dir, src_base + "_out.h5")
    out_db = h5py.File(out_db_path, mode="w")

    if target_group is not None:
        for g in target_group:
            out_group_db = out_db.create_group(g)
            for dataset_name in src_db[g].keys():
                dataset = src_db[g][dataset_name][()]
                out_group_db.create_dataset(dataset_name, data=dataset)


def dataset_num_summary(src_dataset_path, target_variable="BuildingHeight"):
    src_db = h5py.File(src_dataset_path, "r")
    src_base = os.path.splitext(os.path.basename(src_dataset_path))[0]

    tmp = {}
    city_name = sorted([k for k in src_db.keys()])
    for city in city_name:
        n_sample_city = src_db[city][target_variable].shape[0]
        city = re.findall(r"(\w+)@", city)[0]
        if city in tmp.keys():
            tmp[city] += n_sample_city
        else:
            tmp[city] = n_sample_city

    df_summary = pd.DataFrame(columns=["City", "NumOfSample"])
    for k in sorted(tmp.keys()):
        df_summary = df_summary.append({"City": k, "NumOfSample": tmp[k]}, ignore_index=True)

    df_summary.to_csv(src_base + "_summary.csv", index=False)


def dataset_statistic_summary(src_dataset_path, target_variable="BuildingHeight"):
    src_base = os.path.splitext(os.path.basename(src_dataset_path))[0]

    target_list = []
    with h5py.File(src_dataset_path, "r") as h5_file:
        for group_name in h5_file.keys():
            target_dataset = h5_file[group_name][target_variable][()]
            target_list.append(target_dataset)

    target_list = np.concatenate(target_list, axis=0)
    target_median = np.median(target_list)
    target_q1 = np.quantile(target_list, 0.25)
    target_q3 = np.quantile(target_list, 0.75)
    target_qDelta = target_q3 - target_q1

    print("\t".join(["FileName", "Median", "Q1", "Q3", "Q3-Q1"]))
    print("\t".join([src_base, str(target_median), str(target_q1), str(target_q3), str(target_qDelta)]))


# ************************* [1] Get Patch Dataset from Raw Satellite Data *************************
def GetArrayFromTiff(tiff_path):
    tiff_ds = gdal.Open(tiff_path)
    tiff_array = tiff_ds.ReadAsArray() * 1.0

    if tiff_ds.RasterCount == 1:
        tiff_array = np.expand_dims(tiff_array, axis=0)
    return tiff_array


def GetGeoTransformFromTiff(tiff_path):
    tiff_ds = gdal.Open(tiff_path)
    tiff_geoTransform = tiff_ds.GetGeoTransform()
    return tiff_geoTransform


def GetPatchFromTiffArray(tiff_array, tiff_geoTransform, x, y, patch_size, padding_val=np.nan):
    num_band = tiff_array.shape[0]

    x_min = tiff_geoTransform[0]
    dx = tiff_geoTransform[1]
    y_max = tiff_geoTransform[3]
    dy = tiff_geoTransform[5]
    x_id = np.floor((x - x_min) / dx)
    y_id = np.floor((y - y_max) / dy)

    patch = np.zeros(shape=(len(patch_size) * num_band, max(patch_size), max(patch_size)))
    # ---------extract data of different patch size
    for i in range(0, len(patch_size)):
        s = patch_size[i]
        x_start = int(x_id - s/2 + 1)
        x_end = int(x_id + s/2 + 1)
        y_start = int(y_id - s/2 + 1)
        y_end = int(y_id + s/2 + 1)
        w = int((max(patch_size)-s) / 2)
        ar_n = np.arange(num_band)

        patch[i * num_band + ar_n] = np.pad(tiff_array[ar_n, y_start:y_end, x_start:x_end], pad_width=((0, 0), (w, w), (w, w)), mode='constant', constant_values=padding_val)

    return patch


def GetPatchFromCSV_subproc(city, id_list, dta_height, height_name, dta_footprint, footprint_name, x_center, y_center,
                            root_dir, s1_prefix, s2_prefix, suffix_list, patch_size, save_prefix, marker="@",
                            chunk_size=100000, kept_ratio=0.01, aux_feat_info=None):
    num_s1_band = 2
    num_s2_band = 4

    res_list = []
    res = {height_name: [], footprint_name: []}
    feature = {}
    # ---------store the value of TiffArray into a dict in advance where the key is its file name
    tiffArr = {}
    tiffGeoTransform = {}

    for suffix in suffix_list:
        for n in range(1, num_s1_band+1):
            feature[s1_prefix + suffix + "_B" + str(n)] = []
        for n in range(1, num_s2_band+1):
            feature[s2_prefix + suffix + "_B" + str(n)] = []
        # ---------read the target Sentinel-1 data with certain suffix
        s1_subdir = s1_prefix + suffix
        s1_list = [f for f in os.listdir(os.path.join(root_dir, s1_subdir)) if f.startswith(city)]
        s1_target_base = re.findall(r"\w+_{0}".format(suffix), s1_list[0])[0] + ".tif"
        s1_target_file = os.path.join(root_dir, s1_subdir, s1_target_base)
        tiffArr[s1_target_file] = GetArrayFromTiff(s1_target_file)
        tiffGeoTransform[s1_target_file] = GetGeoTransformFromTiff(s1_target_file)
        # ---------read the target Sentinel-2 data with certain suffix
        s2_subdir = s2_prefix + suffix
        s2_list = [f for f in os.listdir(os.path.join(root_dir, s2_subdir)) if f.startswith(city)]
        s2_target_base = re.findall(r"\w+_{0}".format(suffix), s2_list[0])[0] + ".tif"
        s2_target_file = os.path.join(root_dir, s2_subdir, s2_target_base)
        tiffArr[s2_target_file] = GetArrayFromTiff(s2_target_file)
        for band_id in range(0, num_s2_band):
            tiffArr[s2_target_file][band_id] = rgb_rescale_band(tiffArr[s2_target_file][band_id])
        tiffGeoTransform[s2_target_file] = GetGeoTransformFromTiff(s2_target_file)

    # ---------read the auxiliary feature data (e.g. DEM) with certain suffix
    if aux_feat_info is not None:
        for feat in aux_feat_info.keys():
            feature[feat] = []
            aux_subdir = aux_feat_info[feat]["directory"]
            aux_suffix = aux_feat_info[feat]["suffix"]
            aux_list = [f for f in os.listdir(os.path.join(root_dir, aux_subdir)) if f.startswith(city)]
            aux_target_base = re.findall(r"\w+_{0}".format(aux_suffix), aux_list[0])[0] + ".tif"
            aux_target_file = os.path.join(root_dir, aux_subdir, aux_target_base)
            tiffArr[feat] = GetArrayFromTiff(aux_target_file)
            tiffGeoTransform[feat] = GetGeoTransformFromTiff(aux_target_file)

    tmp_feature = dict.fromkeys(feature.keys())

    n_sample = len(id_list)

    chunk_id = 0
    sample_id = 0
    while sample_id < n_sample:
        # ---------get the geolocation of the selected pixel
        y_id = id_list[sample_id][0]
        x_id = id_list[sample_id][1]
        x_loc = x_center[x_id]
        y_loc = y_center[y_id]

        # ---------gather the feature information centered on this selected pixel
        for suffix in suffix_list:
            # ------------get the target Sentinel-1 data from dict via the file name
            s1_subdir = s1_prefix + suffix
            s1_list = [f for f in os.listdir(os.path.join(root_dir, s1_subdir)) if f.startswith(city)]
            s1_target_base = re.findall(r"\w+_{0}".format(suffix), s1_list[0])[0] + ".tif"
            s1_target_file = os.path.join(root_dir, s1_subdir, s1_target_base)
            # ------------get the multi-scale patches center on the selected pixel from the target Sentinel-1 data
            s1_patch = GetPatchFromTiffArray(tiff_array=tiffArr[s1_target_file], tiff_geoTransform=tiffGeoTransform[s1_target_file],
                                             x=x_loc, y=y_loc, patch_size=patch_size)

            for n in range(0, num_s1_band):
                s1_feature_name = s1_prefix + suffix + "_B" + str(n+1)
                # --------------Note that s1_patch has #num_patch_size 2D arrays of #num_s1_bands
                # --------------i.e. the first dimension of s1_patch = [B1_size1, B2_size1, B1_size2, B2_size2, ...]
                tmp_feature[s1_feature_name] = s1_patch[n:len(s1_patch):num_s1_band]

            # ------------get the target Sentinel-2 data from dict via the file name
            s2_subdir = s2_prefix + suffix
            s2_list = [f for f in os.listdir(os.path.join(root_dir, s2_subdir)) if f.startswith(city)]
            s2_target_base = re.findall(r"\w+_{0}".format(suffix), s2_list[0])[0] + ".tif"
            s2_target_file = os.path.join(root_dir, s2_subdir, s2_target_base)
            # ------------get the multi-scale patches center on the selected pixel from the Sentinel-2 target data
            s2_patch = GetPatchFromTiffArray(tiff_array=tiffArr[s2_target_file], tiff_geoTransform=tiffGeoTransform[s2_target_file],
                                             x=x_loc, y=y_loc, patch_size=patch_size)

            for n in range(0, num_s2_band):
                s2_feature_name = s2_prefix + suffix + "_B" + str(n+1)
                tmp_feature[s2_feature_name] = s2_patch[n:len(s2_patch):num_s2_band]

        # ---------gather the feature information centered on this selected pixel
        if aux_feat_info is not None:
            for feat in aux_feat_info.keys():
                aux_psize_ratio = aux_feat_info[feat]["patch_size_ratio"]
                aux_patch = GetPatchFromTiffArray(tiff_array=tiffArr[feat], tiff_geoTransform=tiffGeoTransform[feat],
                                                  x=x_loc, y=y_loc, patch_size=[int(p * aux_psize_ratio) for p in patch_size])
                tmp_feature[feat] = aux_patch

        # --------------discard the extracted patch if it contains any NaN
        '''
        if not np.isnan([v for v in tmp_feature.values()]).any():
            # -----------------add the feature value into our dataset
            for k, v in tmp_feature.items():
                feature[k].append(v)
            # -----------------add the target variable value into our dataset
            res[dta_name].append(dta_array[y_id, x_id])

            tmp_feature.clear()
        '''
        # --------------keep the extracted patch only if it does not contain much NaNs
        nan_ratio_dict = dict((k, calc_nan_ratio(tmp_feature[k][0])) for k in tmp_feature.keys())
        if all(r <= kept_ratio for r in nan_ratio_dict.values()):
            # -----------------add the feature value into our dataset
            for k, v in tmp_feature.items():
                if nan_ratio_dict[k] > 0:
                    feature[k].append(np.expand_dims(fill_nan_nearest(v[0]), axis=0))
                else:
                    feature[k].append(v)
            # -----------------add the target variable value into our dataset
            res[height_name].append(dta_height[y_id, x_id])
            res[footprint_name].append(dta_footprint[y_id, x_id])

            tmp_feature.clear()

        # --------------save results into .npy files if the amount of accumulated data is equal to chunk_size
        # -----------------or the pointer has reached the end of dataset
        if sample_id % chunk_size == chunk_size - 1 or (sample_id == n_sample - 1 and len(res[height_name]) > 0):
            height_file = save_prefix + marker + height_name + "_chunk{0}".format(chunk_id) + ".npy"
            np.save(height_file, res[height_name])
            res_list.append(height_file)
            res[height_name].clear()

            footprint_file = save_prefix + marker + footprint_name + "_chunk{0}".format(chunk_id) + ".npy"
            np.save(footprint_file, res[footprint_name])
            res_list.append(footprint_file)
            res[footprint_name].clear()

            for k, v in feature.items():
                feature_file = save_prefix + marker + k + "_chunk{0}".format(chunk_id) + ".npy"
                np.save(feature_file, v)
                res_list.append(feature_file)
                feature[k].clear()

            chunk_id += 1

        sample_id += 1

    return res_list


def GetPatchFromCSV(csv_path, satellite_data_dir, save_path, target_height_col, target_footprint_col, patch_size,
                    height_name, footprint_name, height_prefix=None, footprint_prefix=None,
                    height_min=3.0, height_max=600.0, area_min=1e-2, chunk_size=100000, num_cpu=1, kept_ratio=0.01,
                    aux_feat_info=None):
    # ------By default, we organize our raw satellite data directory as follows:
    # ---------------------------------------------------------------------------
    #       +---SatelliteDataDirectory
    #       |   +---sentinel_1_'suffix'
    #       |       +---'city'_'year'_sentinel_1_'suffix'.tif
    #       |       +---'city'_'year'_sentinel_1_'suffix'([1-9]+)-([1-9]+).tif
    #       |   +---sentinel_2_'suffix'
    #       |       +---'city'_'year'_sentinel_2_'suffix'.tif
    # ---------------------------------------------------------------------------
    # ------where 'suffix' is chosen as follows:
    # aggregate_suffix = ["avg", "25pt", "50pt", "75pt"]
    aggregate_suffix = ["50pt"]
    # ------Also, if you want to rename the sub-directory, please modify the following prefix
    s1_prefix = "sentinel_1_"
    s2_prefix = "sentinel_2_"

    # ------By default, we store our extracted patches in HDF5 format where data are organized as follows
    # ---------------------------------------------------------------------------
    #       +---CityName_1
    #       |   +---target_variable (e.g. Height, Footprint)
    #       |   +---CityName_sentinel_1_'suffix'_B1
    #       |   +---CityName_sentinel_1_'suffix'_B2
    #       |   +---CityName_sentinel_2_'suffix'_B1
    #       |   +---CityName_sentinel_2_'suffix'_B2
    #       |   +---CityName_sentinel_2_'suffix'_B3
    #       |   +---CityName_sentinel_2_'suffix'_B4
    #       +---CityName_2
    #       |   +---...
    # ---------------------------------------------------------------------------
    hf_db = h5py.File(save_path, mode="w")
    hf_dir = os.path.dirname(save_path)

    raw_band = []
    num_s1_band = 2
    num_s2_band = 4
    for suffix in aggregate_suffix:
        for n in range(1, num_s1_band + 1):
            raw_band.append(s1_prefix + suffix + "_B" + str(n))
        for n in range(1, num_s2_band + 1):
            raw_band.append(s2_prefix + suffix + "_B" + str(n))

    num_cpu_available = multiprocessing.cpu_count()
    if num_cpu_available < num_cpu:
        print("Use %d CPUs which is available now" % num_cpu_available)
        num_cpu = num_cpu_available

    df = pd.read_csv(csv_path)
    for row_id in df.index:
        city = df.loc[row_id]["City"]
        print("*"*10, city, "Starts", "*"*10)
        # ------[1] read the target GeoTiff file as footprint/height value array
        height_path = os.path.join(height_prefix, df.loc[row_id][target_height_col])
        height_ds = gdal.Open(height_path)
        height_nodata_val = height_ds.GetRasterBand(1).GetNoDataValue()
        height_val = height_ds.ReadAsArray()
        footprint_path = os.path.join(footprint_prefix, df.loc[row_id][target_footprint_col])
        footprint_ds = gdal.Open(footprint_path)
        footprint_val = footprint_ds.ReadAsArray()
        # ---------get the basic spatial coordinate information of target GeoTiff file
        target_geoTransform = height_ds.GetGeoTransform()
        x_min = target_geoTransform[0]
        y_max = target_geoTransform[3]
        x_center = np.array([x_min + target_geoTransform[1]*(i+1/2) for i in range(0, height_ds.RasterXSize)])
        y_center = np.array([y_max + target_geoTransform[5]*(i+1/2) for i in range(0, height_ds.RasterYSize)])

        # ------[2] mask unreliable pixels
        # ---------keep the pixel whose height value is in the range [threshold_min, threshold_max]
        sample_height_test = np.logical_and(height_val != height_nodata_val, height_val > height_min)
        sample_height_test = np.logical_and(sample_height_test, height_val < height_max)

        # ---------discard the pixel whose footprint value is too small (<1e-6)
        # ---------discard the pixel whose height value is large (>50m) but footprint value is small (<1e-6)
        sample_footprint_test_1 = np.logical_and(footprint_val >= 0, footprint_val < area_min)
        sample_footprint_test_2 = np.logical_and(footprint_val < 4 * area_min, height_val > 20.0)
        sample_footprint_test = ~np.logical_or(sample_footprint_test_1, sample_footprint_test_2)

        sample_test = np.logical_and(sample_height_test, sample_footprint_test)

        # ------[3] sample from the reliable pixels
        sample_loc = np.where(sample_test)
        sample_loc = np.concatenate([sample_loc[0].reshape(-1, 1), sample_loc[1].reshape(-1, 1)], axis=1)

        # ---------check whether we have one GeoTiff file for the entire region or multiple files for sub-regions
        # ------------and we desire one GeoTiff file
        for suffix in aggregate_suffix:
            s1_subdir = s1_prefix + suffix
            s1_list = [f for f in os.listdir(os.path.join(satellite_data_dir, s1_subdir)) if f.startswith(city)]
            s1_target_base = re.findall(r"\w+_{0}".format(suffix), s1_list[0])[0] + ".tif"
            s1_target_file = os.path.join(satellite_data_dir, s1_subdir, s1_target_base)
            if not os.path.exists(s1_target_file):
                # os.system("gdalwarp {0} {1}".format(" ".join([os.path.join(satellite_data_dir, s1_subdir, f) for f in s1_list]), s1_target_file))
                gdal.Warp(destNameOrDestDS=s1_target_file, srcDSOrSrcDSTab=[os.path.join(satellite_data_dir, s1_subdir, f) for f in s1_list])

            s2_subdir = s2_prefix + suffix
            s2_list = [f for f in os.listdir(os.path.join(satellite_data_dir, s2_subdir)) if f.startswith(city)]
            s2_target_base = re.findall(r"\w+_{0}".format(suffix), s2_list[0])[0] + ".tif"
            s2_target_file = os.path.join(satellite_data_dir, s2_subdir, s2_target_base)
            if not os.path.exists(s2_target_file):
                # os.system("gdalwarp {0} {1}".format(" ".join([os.path.join(satellite_data_dir, s2_subdir, f) for f in s2_list]), s2_target_file))
                gdal.Warp(destNameOrDestDS=s2_target_file, srcDSOrSrcDSTab=[os.path.join(satellite_data_dir, s2_subdir, f) for f in s2_list])

        if aux_feat_info is not None:
            for feat in aux_feat_info.keys():
                if feat not in raw_band:
                    raw_band.append(feat)
                aux_subdir = aux_feat_info[feat]["directory"]
                aux_suffix = aux_feat_info[feat]["suffix"]
                aux_list = [f for f in os.listdir(os.path.join(satellite_data_dir, aux_subdir)) if f.startswith(city)]
                aux_target_base = re.findall(r"\w+_{0}".format(aux_suffix), aux_list[0])[0] + ".tif"
                aux_target_file = os.path.join(satellite_data_dir, aux_subdir, aux_target_base)
                if not os.path.exists(aux_target_file):
                    # os.system("gdalwarp {0} {1}".format(" ".join([os.path.join(satellite_data_dir, aux_subdir, f) for f in aux_list]), aux_target_file))
                    gdal.Warp(destNameOrDestDS=aux_target_file, srcDSOrSrcDSTab=[os.path.join(satellite_data_dir, aux_subdir, f) for f in aux_list])

        # ---------To accelerate data extraction, we call sub-processes
        pool = multiprocessing.Pool(processes=num_cpu)
        id_sub = np.array_split(sample_loc, num_cpu)
        arg_list = [(city, id_sub[j], height_val, height_name, footprint_val, footprint_name, x_center, y_center,
                     satellite_data_dir, s1_prefix, s2_prefix, aggregate_suffix, patch_size,
                     os.path.join(hf_dir, "_".join([city, "TEMP", str(j)])),
                     "@", int(chunk_size/num_cpu), kept_ratio, aux_feat_info) for j in range(0, num_cpu)]
        res_list = pool.starmap(GetPatchFromCSV_subproc, arg_list)
        pool.close()
        pool.join()

        # ---------determine how many chunks we need to divide
        res_list_flatten = reduce(lambda x, y: x+y, res_list)
        subproc_chunk = [int(re.findall(r"chunk(\d+).npy", res_file)[0]) for res_file in res_list_flatten]
        n_chunk = max(subproc_chunk)
        n_sample = 0
        for chunk_id in range(0, n_chunk + 1):
            # ------------create (sub-)group for the city
            city_db = hf_db.create_group(city + "@{0}".format(chunk_id))

            # ------------gather target variable
            target_height_file = []
            target_footprint_file = []
            for res_file in res_list_flatten:
                if len(re.findall(r"@{0}_chunk{1}".format(height_name, chunk_id), res_file)) > 0:
                    target_height_file.append(res_file)
                if len(re.findall(r"@{0}_chunk{1}".format(footprint_name, chunk_id), res_file)) > 0:
                    target_footprint_file.append(res_file)

            # ---------------sort the file according to the Process ID (key1) and chunk_id in each subprocess (key2)
            target_height_file_sorted = sorted(target_height_file, key=lambda x: (int(re.findall(r"TEMP_(\d+)@", x)[0]),
                                                                                  int(re.findall(r"_chunk(\d+).npy", x)[0])))
            target_height = np.concatenate([np.load(f) for f in target_height_file_sorted])
            n_sample += len(target_height)
            city_db.create_dataset(height_name, data=target_height)

            target_footprint_file_sorted = sorted(target_footprint_file, key=lambda x: (int(re.findall(r"TEMP_(\d+)@", x)[0]),
                                                                                        int(re.findall(r"_chunk(\d+).npy", x)[0])))
            target_footprint = np.concatenate([np.load(f) for f in target_footprint_file_sorted])
            city_db.create_dataset(footprint_name, data=target_footprint)

            # ------------gather raw band variable
            for band in raw_band:
                band_file = []
                for res_file in res_list_flatten:
                    if len(re.findall(r"@{0}_chunk{1}".format(band, chunk_id), res_file)) > 0:
                        band_file.append(res_file)

                band_file_sorted = sorted(band_file, key=lambda x: (int(re.findall(r"TEMP_(\d+)@", x)[0]),
                                                                    int(re.findall(r"_chunk(\d+).npy", x)[0])))
                city_db.create_dataset(band, data=np.concatenate([np.load(f) for f in band_file_sorted]))

        print(n_sample, "sample(s) collected")
        # os.system("rm -rf {0}".format(os.path.join(hf_dir, "*TEMP*")))
        tmp_list = glob.glob(os.path.join(hf_dir, "*TEMP*"))
        for f in tmp_list:
            os.remove(f)

        print("*" * 10, city, "Finished", "*" * 10)

    hf_db.close()


if __name__ == "__main__":
    
    path_prefix = "/data/lyy/BuildingProject"
    '''
    ref_data_dir_prefix = os.path.join(path_prefix, "ReferenceData/Summary")
    rs_data_dir_prefix = path_prefix
    rs_data_dir = os.path.join(rs_data_dir_prefix, "raw_data")

    feat_add = {"DEM": {"directory": "SRTM", "suffix": "srtm", "patch_size_ratio": 1}}

    res_scale_mapping = {100: 15, 250: 30, 500: 60, 1000: 120}
    num_cpu = 1

    for rs_res in [100, 250, 500, 1000]:
        scale = [res_scale_mapping[rs_res]]
        height_prefix = os.path.join(ref_data_dir_prefix, "BuildingHeight_option_%dm" % rs_res)
        footprint_prefix = os.path.join(ref_data_dir_prefix, "BuildingFootprint_%dm" % rs_res)

        h5_path = os.path.join(rs_data_dir, "patch_data_50pt_s%d_%dm.h5" % (scale[0], rs_res))

        csv_name = "PatchDatasetGen_back.csv"
        height_col = "BuildingHeight_Path"
        footprint_col = "BuildingFootprint_Path"

        GetPatchFromCSV(csv_path=csv_name, satellite_data_dir=rs_data_dir, save_path=h5_path,
                        target_height_col=height_col, target_footprint_col=footprint_col,
                        patch_size=scale,
                        height_name="BuildingHeight", footprint_name="BuildingFootprint",
                        height_prefix=height_prefix, footprint_prefix=footprint_prefix,
                        height_min=2.0, height_max=500.0, area_min=1.0 / (scale[0]*scale[0]),
                        chunk_size=100000, num_cpu=num_cpu, aux_feat_info=feat_add)

    city_exclude_1 = None
    city_exclude_2 = None

    #sample_dataset(src_dataset_path="/data/lyy/BuildingProject/dataset/patch_data_50pt_s60_500m_valid.h5", out_dataset_path="/data/lyy/BuildingProject/dataset/patch_data_50pt_s60_500m_sample.h5",
                   #sample_ratio=0.02)

    for rs_res in [100, 250, 500, 1000]:
        s = res_scale_mapping[rs_res]

        split_dataset(src_dataset_path=os.path.join(rs_data_dir, "patch_data_50pt_s{0}_{1}m.h5".format(s, rs_res)),
                        dataset_split_ratio=0.2, group_split_ratio=0.0,
                        group_exclude=city_exclude_1)
        
        split_dataset(src_dataset_path=os.path.join(rs_data_dir, "patch_data_50pt_s{0}_{1}m_valid.h5".format(s, rs_res)),
                        dataset_split_ratio=0.5, group_split_ratio=0.0,
                        group_exclude=city_exclude_1)

        shutil.move("patch_data_50pt_s{0}_{1}m_valid_train.h5".format(s, rs_res), "patch_data_50pt_s{0}_{1}m_valid.h5".format(s, rs_res))
        shutil.move("patch_data_50pt_s{0}_{1}m_valid_valid.h5".format(s, rs_res), "patch_data_50pt_s{0}_{1}m_test.h5".format(s, rs_res))
    '''
      
    # -----calculate the number of samples in each dataset
    ref_data_dir_prefix = os.path.join(path_prefix, "ReferenceData/Summary")
    rs_data_dir_prefix = path_prefix
    rs_data_dir = os.path.join(rs_data_dir_prefix, "dataset")

    res_scale_mapping = {100: 15, 250: 30, 500: 60, 1000: 120}

    for rs_res in [100, 250, 500, 1000]:
        scale = [res_scale_mapping[rs_res]]
        h5_path = os.path.join(rs_data_dir, "patch_data_50pt_s%d_%dm.h5" % (scale[0], rs_res))
        dataset_num_summary(src_dataset_path=h5_path)
    '''
    '''  

    '''
    rs_data_dir = os.path.join(rs_data_dir_prefix, "dataset")
    for rs_res in [100, 250, 500, 1000]:
        scale = [res_scale_mapping[rs_res]]
        h5_path = os.path.join(rs_data_dir, "patch_data_50pt_s%d_%dm.h5" % (scale[0], rs_res))
        dataset_statistic_summary(src_dataset_path=h5_path, target_variable="BuildingFootprint")
    '''
