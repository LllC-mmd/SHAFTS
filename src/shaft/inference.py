import glob

from torch import cuda

from train import *
from dataset import *
from DL_train import *

import albumentations as album


clahe_transform = album.CLAHE(p=1.0)


def check_extent(base_dir, tif_ref, x_min, y_min, x_max, y_max, padding):
    x_min_list = []
    y_min_list = []
    x_max_list = []
    y_max_list = []

    for aggregate_key in tif_ref.keys():
        sub_dict = tif_ref[aggregate_key]

        if not isinstance(sub_dict[s1_key], list):
            s1_input = [tif_ref[aggregate_key][s1_key]]
        else:
            s1_input = tif_ref[aggregate_key][s1_key]

        test_s1_tif = os.path.join(base_dir, "TEMP_S1.tif")
        # os.system("gdalwarp {0} {1}".format(" ".join(s1_input), test_s1_tif))
        gdal.Warp(destNameOrDestDS=test_s1_tif, srcDSOrSrcDSTab=s1_input)

        s1_ds = gdal.Open(test_s1_tif)
        s1_x_min, s1_dx, _, s1_y_max, _, s1_dy = s1_ds.GetGeoTransform()
        s1_x_max = s1_x_min + s1_ds.RasterXSize * s1_dx
        s1_y_min = s1_y_max + s1_ds.RasterYSize * s1_dy

        if not isinstance(sub_dict[s2_key], list):
            s2_input = [tif_ref[aggregate_key][s2_key]]
        else:
            s2_input = tif_ref[aggregate_key][s2_key]

        test_s2_tif = os.path.join(base_dir, "TEMP_S2.tif")
        # os.system("gdalwarp {0} {1}".format(" ".join(s2_input), test_s2_tif))
        gdal.Warp(destNameOrDestDS=test_s2_tif, srcDSOrSrcDSTab=s2_input)

        s2_ds = gdal.Open(test_s2_tif)
        s2_x_min, s2_dx, _, s2_y_max, _, s2_dy = s2_ds.GetGeoTransform()
        s2_x_max = s2_x_min + s2_ds.RasterXSize * s2_dx
        s2_y_min = s2_y_max + s2_ds.RasterYSize * s2_dy

        x_min_list.append(s1_x_min)
        x_min_list.append(s2_x_min)
        x_max_list.append(s1_x_max)
        x_max_list.append(s2_x_max)
        y_min_list.append(s1_y_min)
        y_min_list.append(s2_y_min)
        y_max_list.append(s1_y_max)
        y_max_list.append(s2_y_max)

        os.remove(test_s1_tif)
        os.remove(test_s2_tif)

    if x_min - padding < max(x_min_list) or y_min - padding < max(y_min_list) or x_max + padding > min(x_max_list) or y_max + padding > min(y_max_list):
        raise RuntimeError("ExtentError: Given extent is {0} but expected is {1}.".format((max(x_min_list), max(y_min_list),
                                                                                           min(x_max_list), min(y_max_list)),
                                                                                          (x_min, y_min, x_max, y_max)))


def get_patch_from_tiff(file_dict, loc_list, s1_key, s2_key, suffix_list, patch_size, save_prefix, marker="@", chunk_size=100000):
    # ------By default, we assume the parameter `file_dict` takes the form as follows:
    # ---------------------------------------------------------------------------
    #       {"50pt":
    #               {
    #                   `s1_key`:  `the location of Sentinel-1-50pt GeoTiff file`,
    #                   `s2_key`:  `the location of Sentinel-2-50pt GeoTiff file`,
    #                },
    #        "AnotherAggregation":
    #               {
    #                   `s1_key`:  `the location of Sentinel-1-aggregation GeoTiff file`,
    #                   `s2_key`:  `the location of Sentinel-2-aggregation GeoTiff file`,
    #               },
    #         ......
    #       }
    # ---------------------------------------------------------------------------
    num_s1_band = 2
    num_s2_band = 4

    res_list = []
    feature = {}
    # ---------store the value of TiffArray into a dict in advance where the key is its file name
    tiffArr = {}
    tiffGeoTransform = {}

    for aggregate_key in file_dict.keys():
        for n in range(1, num_s1_band+1):
            s1_feature_name = "_".join([s1_key, aggregate_key, "B" + str(n)])
            feature[s1_feature_name] = []
        for n in range(1, num_s2_band+1):
            s2_feature_name = "_".join([s2_key, aggregate_key, "B" + str(n)])
            feature[s2_feature_name] = []

        sub_dict = file_dict[aggregate_key]
        # ---------read the target Sentinel-1 data
        s1_name = "_".join([s1_key, aggregate_key])
        tiffArr[s1_name] = GetArrayFromTiff(sub_dict[s1_key])
        # ------------fill the NaN value in the band data using Nearest-neighbor interpolation
        # ------------note that for inference, we fill NaNs at the beginning (different from training patch collection)
        for band_id in range(0, num_s1_band):
            tiffArr[s1_name][band_id] = fill_nan_nearest(tiffArr[s1_name][band_id])
        tiffGeoTransform[s1_name] = GetGeoTransformFromTiff(sub_dict[s1_key])
        # ---------read the target Sentinel-2 data
        s2_name = "_".join([s2_key, aggregate_key])
        tiffArr[s2_name] = GetArrayFromTiff(sub_dict[s2_key])
        for band_id in range(0, num_s2_band):
            tiffArr[s2_name][band_id] = fill_nan_nearest(tiffArr[s2_name][band_id])
        tiffGeoTransform[s2_name] = GetGeoTransformFromTiff(sub_dict[s2_key])

    tmp_feature = dict.fromkeys(feature.keys())

    n_sample = len(loc_list)

    chunk_id = 0
    sample_id = 0
    while sample_id < n_sample:
        # ---------get the geolocation of the selected pixel
        y_loc = loc_list[sample_id][0]
        x_loc = loc_list[sample_id][1]

        # ---------gather the feature information centered on this selected pixel
        for aggregate_key in suffix_list:
            # ------------get the target Sentinel-1 data from dict via the file name
            s1_name = "_".join([s1_key, aggregate_key])
            # ------------get the multi-scale patches center on the selected pixel from the target Sentinel-1 data
            s1_patch = GetPatchFromTiffArray(tiff_array=tiffArr[s1_name], tiff_geoTransform=tiffGeoTransform[s1_name],
                                             x=x_loc, y=y_loc, patch_size=patch_size)

            for n in range(0, num_s1_band):
                s1_feature_name = "_".join([s1_key, aggregate_key, "B" + str(n+1)])
                # --------------Note that s1_patch has #num_patch_size 2D arrays of #num_s1_bands
                # --------------i.e. the first dimension of s1_patch = [B1_size1, B2_size1, B1_size2, B2_size2, ...]
                tmp_feature[s1_feature_name] = s1_patch[n:len(s1_patch):num_s1_band]

            # ------------get the multi-scale patches center on the selected pixel from the Sentinel-2 target data
            s2_name = "_".join([s2_key, aggregate_key])
            s2_patch = GetPatchFromTiffArray(tiff_array=tiffArr[s2_name], tiff_geoTransform=tiffGeoTransform[s2_name],
                                             x=x_loc, y=y_loc, patch_size=patch_size)

            for n in range(0, num_s2_band):
                s2_feature_name = "_".join([s2_key, aggregate_key, "B" + str(n+1)])
                tmp_feature[s2_feature_name] = s2_patch[n:len(s2_patch):num_s2_band]

        for k, v in tmp_feature.items():
            feature[k].append(v)

        # --------------save results into .npy files if the amount of accumulated data is equal to chunk_size
        # -----------------or the pointer has reached the end of dataset
        if sample_id % chunk_size == chunk_size - 1 or sample_id == n_sample - 1:
            for k, v in feature.items():
                feature_file = save_prefix + marker + k + "_chunk" + str(chunk_id) + ".npy"
                np.save(feature_file, v)
                res_list.append(feature_file)
                feature[k].clear()

            chunk_id += 1

        sample_id += 1

    return res_list


def get_s1_patch_from_tiff(file_dict, loc_list, s1_key, suffix_list, patch_size, save_prefix, marker="@", chunk_size=100000):
    num_s1_band = 2

    res_list = []
    feature = {}
    # ---------store the value of TiffArray into a dict in advance where the key is its file name
    tiffArr = {}
    tiffGeoTransform = {}

    for aggregate_key in file_dict.keys():
        for n in range(1, num_s1_band+1):
            s1_feature_name = "_".join([s1_key, aggregate_key, "B" + str(n)])
            feature[s1_feature_name] = []

        sub_dict = file_dict[aggregate_key]
        # ---------read the target Sentinel-1 data
        s1_name = "_".join([s1_key, aggregate_key])
        tiffArr[s1_name] = GetArrayFromTiff(sub_dict[s1_key])
        tiffGeoTransform[s1_name] = GetGeoTransformFromTiff(sub_dict[s1_key])

    n_sample = len(loc_list)

    chunk_id = 0
    sample_id = 0
    while sample_id < n_sample:
        # ---------get the geolocation of the selected pixel
        y_loc = loc_list[sample_id][0]
        x_loc = loc_list[sample_id][1]

        # ---------gather the feature information centered on this selected pixel
        for aggregate_key in suffix_list:
            # ------------get the target Sentinel-1 data from dict via the file name
            s1_name = "_".join([s1_key, aggregate_key])
            # ------------get the multi-scale patches center on the selected pixel from the target Sentinel-1 data
            s1_patch = GetPatchFromTiffArray(tiff_array=tiffArr[s1_name], tiff_geoTransform=tiffGeoTransform[s1_name],
                                             x=x_loc, y=y_loc, patch_size=patch_size)

            for n in range(0, num_s1_band):
                s1_feature_name = "_".join([s1_key, aggregate_key, "B" + str(n+1)])
                # --------------Note that s1_patch has #num_patch_size 2D arrays of #num_s1_bands
                # --------------i.e. the first dimension of s1_patch = [B1_size1, B2_size1, B1_size2, B2_size2, ...]
                feature[s1_feature_name].append(s1_patch[n:len(s1_patch):num_s1_band])

        # --------------save results into .npy files if the amount of accumulated data is equal to chunk_size
        # -----------------or the pointer has reached the end of dataset
        if sample_id % chunk_size == chunk_size - 1 or sample_id == n_sample - 1:
            for k, v in feature.items():
                feature_file = save_prefix + marker + k + "_chunk" + str(chunk_id) + ".npy"
                np.save(feature_file, v)
                res_list.append(feature_file)
                feature[k].clear()

            chunk_id += 1

        sample_id += 1

    return res_list


def get_backscatterCoef_from_dataset(dataset_path, aggregate, gamma=5.0, s1_prefix="sentinel_1", scale_level=0, saved=True, chunk_size=50000, save_file=None):
    band_ref = {"VV": "B1", "VH": "B2"}

    # ------Read HDF5 dataset
    f = h5py.File(dataset_path, mode="r")
    city_name = [k for k in f.keys()]

    s1_vv_list = []
    s1_vh_list = []

    for city in city_name:
        test_key = list(f[city].keys())[0]
        n_sample_city = len(f[city][test_key])

        n_chunk = int(np.floor(n_sample_city / chunk_size))
        for chunk_id in range(0, n_chunk + 1):
            id_start = chunk_id * chunk_size
            n_subsample = min(n_sample_city - id_start, chunk_size)
            id_end = id_start + n_subsample

            # ---------get the raw VV data
            s1_vv_ref = "_".join([s1_prefix, aggregate, band_ref["VV"]])
            vv_raw = f[city][s1_vv_ref][id_start:id_end]
            # ---------convert to the raw VV observation into the backscatter coefficients
            s1_vv = get_backscatterCoef(vv_raw[:, scale_level, :, :])
            # ---------calculate the mean backscatter coefficients over our patch
            s1_vv_avg = np.mean(s1_vv, axis=(-1, -2))
            s1_vv_list.append(s1_vv_avg)

            s1_vh_ref = "_".join([s1_prefix, aggregate, band_ref["VH"]])
            vh_raw = f[city][s1_vh_ref][id_start:id_end]
            s1_vh = get_backscatterCoef(vh_raw[:, scale_level, :, :])
            s1_vh_avg = np.mean(s1_vh, axis=(-1, -2))
            s1_vh_list.append(s1_vh_avg)

    f.close()

    # ---------calculate VVH = VV * VH^gamma
    s1_feature = np.concatenate((np.expand_dims(np.concatenate(s1_vv_list), 1), np.expand_dims(np.concatenate(s1_vh_list), 1)), axis=1)

    if saved:
        np.save(save_file, s1_feature)

    return s1_feature


def get_feature_from_dataset(dataset_path, aggregate_suffix, s1_prefix="sentinel_1", s2_prefix="sentinel_2", scale_level=0, saved=True, num_cpu=1, chunk_size=50000, save_file=None):
    # ------Experiment Settings
    scale = 0.0001
    s2_band_use = ["Red", "Green", "Blue", "NIR", "Brightness"]
    band_statistics = ["Mean", "Std", "Max", "Min", "50pt", "25pt", "75pt"]
    s2_morph_ops = ["Opening", "Closing", "OpeningByReconstruction", "ClosingByReconstruction"]
    band_ref = {"VV": "B1", "VH": "B2", "R": "B1", "G": "B2", "B": "B3", "NIR": "B4"}
    DMP_size = [3, 5, 7, 9, 11, 13]
    GLCM_angle = [0.0, 0.25*np.pi, 0.5*np.pi, 0.75*np.pi]
    texture_list = ["Mean", "Variance", "Homogeneity", "Contrast", "Dissimilarity", "Entropy", "2nd_moment", "Correlation"]

    # ------Read HDF5 dataset
    f = h5py.File(dataset_path, mode="r")
    city_name = [k for k in f.keys()]

    feature_list = []

    city_feature = {}
    city_band_ref = {}

    for city in city_name:
        test_key = list(f[city].keys())[0]
        n_sample_city = len(f[city][test_key])

        n_chunk = int(np.floor(n_sample_city / chunk_size))
        for chunk_id in range(0, n_chunk + 1):
            id_start = chunk_id * chunk_size
            n_subsample = min(n_sample_city - id_start, chunk_size)

            if n_subsample == 0:
                continue

            id_end = id_start + n_subsample

            print("sampling interval: [{0}, {1})".format(id_start, id_end))

            for suffix in aggregate_suffix:
                city_band_ref[suffix] = {}
                # ---------------[1] Sentinel-1-VV
                s1_vv_ref = "_".join([s1_prefix, suffix, band_ref["VV"]])
                vv_raw = f[city][s1_vv_ref][id_start:id_end, scale_level, :, :]
                s1_vv = get_backscatterCoef(vv_raw)
                city_feature[s1_vv_ref] = s1_vv
                city_band_ref[suffix]["VV"] = s1_vv_ref
                # ---------------[2] Sentinel-1-VH
                s1_vh_ref = "_".join([s1_prefix, suffix, band_ref["VH"]])
                vh_raw = f[city][s1_vh_ref][id_start:id_end, scale_level, :, :]
                s1_vh = get_backscatterCoef(vh_raw)
                city_feature[s1_vh_ref] = s1_vh
                city_band_ref[suffix]["VH"] = s1_vh_ref
                # ---------------[3] Sentinel-2-R
                s2_r_ref = "_".join([s2_prefix, suffix, band_ref["R"]])
                s2_r = f[city][s2_r_ref][id_start:id_end, scale_level, :, :] * scale
                city_feature[s2_r_ref] = s2_r
                city_band_ref[suffix]["R"] = s2_r_ref
                # ---------------[4] Sentinel-2-G
                s2_g_ref = "_".join([s2_prefix, suffix, band_ref["G"]])
                s2_g = f[city][s2_g_ref][id_start:id_end, scale_level, :, :] * scale
                city_feature[s2_g_ref] = s2_g
                city_band_ref[suffix]["G"] = s2_g_ref
                # ---------------[5] Sentinel-2-B
                s2_b_ref = "_".join([s2_prefix, suffix, band_ref["B"]])
                s2_b = f[city][s2_b_ref][id_start:id_end, scale_level, :, :] * scale
                city_feature[s2_b_ref] = s2_b
                city_band_ref[suffix]["B"] = s2_b_ref
                # ---------------[6] Sentinel-2-NIR
                s2_nir_ref = "_".join([s2_prefix, suffix, band_ref["NIR"]])
                s2_nir = f[city][s2_nir_ref][id_start:id_end, scale_level, :, :] * scale
                city_feature[s2_nir_ref] = s2_nir
                city_band_ref[suffix]["NIR"] = s2_nir_ref

            res = get_feature(city_feature, city_band_ref, s2_band_use, band_statistics, s2_morph_ops, DMP_size,
                              GLCM_angle, texture_list, True, True, num_cpu)

            city_feature.clear()
            city_band_ref.clear()

            # ---------get the features, i.e., some statistics of patch data from each city
            # ------------whose shape is: (n_sample_city, n_aggregate*feature)
            city_feature_dta = np.concatenate([feat for feat in res], axis=1)
            feature_list.append(city_feature_dta)

        print("*" * 10, city, "Finished", "*" * 10)

    feature = np.concatenate([c for c in feature_list], axis=0)
    # ---------inf might exist in feature matrix due to numerical problems, which requires post-processing
    feature = np.where(np.isinf(np.abs(feature)), 0.0, feature)

    if saved:
        np.save(save_file, feature)

    return feature


def pred_height_from_tiff_VVH(x_min, y_min, x_max, y_max, out_file, tif_ref, patch_size, predictor, resolution,
                              s1_key="sentinel_1", base_dir=".", chunk_size=100000, num_cpu=1, log_scale=True):
    # ------By default, we assume the parameter `tif_dict` takes the form as follows:
    # ---------------------------------------------------------------------------
    #       {"50pt":
    #               {
    #                   `s1_key`:  `the location of Sentinel-1-50pt GeoTiff file`,
    #                },
    #        "AnotherAggregation":
    #               {
    #                   `s1_key`:  `the location of Sentinel-1-aggregation GeoTiff file`,
    #               },
    #         ......
    #       }
    # ---------------------------------------------------------------------------
    hf_db_path = os.path.join(base_dir, "TEMP.h5")
    hf_db = h5py.File(hf_db_path, mode="w")

    num_s1_band = 2
    marker = "@"
    padding = 0.02

    gamma = 5.0

    raw_band = []
    for aggregate_key in tif_ref.keys():
        for n in range(1, num_s1_band+1):
            raw_band.append("_".join([s1_key, aggregate_key, "B" + str(n)]))

    num_cpu_available = multiprocessing.cpu_count()
    if num_cpu_available < num_cpu:
        print("Use %d CPUs which is available now" % num_cpu_available)
        num_cpu = num_cpu_available

    # ------get features from input GeoTif files
    # ---------[0] check whether the available GeoTiff file satisfies our needs for prediction
    #check_extent(tif_ref, x_min, y_min, x_max, y_max, padding)

    # ---------[0] clip the given GeoTiff file to the target extent for memory saving
    tif_input = dict.fromkeys(tif_ref)
    target_extent = [x_min - padding, y_min - padding, x_max + padding, y_max + padding]
    for aggregate_key in tif_ref.keys():
        tif_input[aggregate_key] = {}
        if not isinstance(tif_ref[aggregate_key][s1_key], list):
            s1_input = [tif_ref[aggregate_key][s1_key]]
        else:
            s1_input = tif_ref[aggregate_key][s1_key]

        s1_input_tif = os.path.join(base_dir, "@".join([aggregate_key, s1_key, "TEMP"]) + ".tif")
        # os.system("gdalwarp -te {0} {1} {2}".format(" ".join([str(cc) for cc in target_extent]), " ".join(s1_input), s1_input_tif))
        gdal.Warp(destNameOrDestDS=s1_input_tif, srcDSOrSrcDSTab=s1_input, options=gdal.WarpOptions(outputBounds=target_extent))
        tif_input[aggregate_key][s1_key] = s1_input_tif

    # ---------[1] calculate the coordinates of center points of patches where the building height is to be predicted
    nx = int(np.ceil((x_max - x_min) / resolution))
    ny = int(np.ceil((y_max - y_min) / resolution))
    x_center = np.array([x_min + resolution * (i + 1/2) for i in range(0, nx)])
    y_center = np.array([y_max - resolution * (i + 1/2) for i in range(0, ny)])
    xx_center, yy_center = np.meshgrid(x_center, y_center)
    sample_loc = np.concatenate([yy_center.reshape(-1, 1), xx_center.reshape(-1, 1)], axis=1)

    # ---------[2] divide the whole GeoTiff files into patches whose size is: resolution * resolution
    # ------------use multiprocessing for acceleration
    pool = multiprocessing.Pool(processes=num_cpu)

    sub_chunk_size = int(chunk_size/num_cpu)
    n_sample = len(sample_loc)
    n_sub_chunk = int(np.ceil(n_sample / sub_chunk_size))
    id_sub = {}
    for j in range(0, num_cpu):
        id_sub[j] = []

    for sub_chunk_id in range(0, n_sub_chunk):
        subproc_id = sub_chunk_id % num_cpu
        id_sub[subproc_id].append(sample_loc[sub_chunk_size*sub_chunk_id:sub_chunk_size*(sub_chunk_id+1)])

    for k in id_sub.keys():
        id_sub[k] = np.concatenate(id_sub[k], axis=0)

    arg_list = [(tif_input, id_sub[j], s1_key, list(tif_input.keys()), patch_size,
                 os.path.join(base_dir, "_".join(["TEMP", str(j)])), marker, sub_chunk_size) for j in range(0, num_cpu)]
    res_list = pool.starmap(get_s1_patch_from_tiff, arg_list)
    pool.close()
    pool.join()

    # ---------determine how many chunks we need to divide using the maximum of chunk_id in each subprocess
    res_list_flatten = reduce(lambda x, y: x + y, res_list)
    subproc_chunk = [int(re.findall(r"chunk(\d+).npy", res_file)[0]) for res_file in res_list_flatten]
    n_chunk = max(subproc_chunk)

    for chunk_id in range(0, n_chunk + 1):
        # ------------create (sub-)group for the city
        city_db = hf_db.create_group("TEMP@{0}".format(chunk_id))

        # ------------gather raw band variable
        for band in raw_band:
            band_file = []
            for res_file in res_list_flatten:
                if len(re.findall(r"@{0}_chunk{1}".format(band, chunk_id), res_file)) > 0:
                    band_file.append(res_file)

            band_file_sorted = sorted(band_file, key=lambda x: (int(re.findall(r"TEMP_(\d+)@", x)[0]), int(re.findall(r"_chunk(\d+).npy", x)[0])))
            city_db.create_dataset(band, data=np.concatenate([np.load(f) for f in band_file_sorted]))

    # os.system("rm -rf {0}".format(os.path.join(base_dir, "*TEMP*.npy")))
    tmp_list = glob.glob(os.path.join(base_dir, "*TEMP*.npy"))
    for f in tmp_list:
        os.remove(f)

    # ---------[3] calculate the feature of each patch and then do some preprocessing with it
    feature = get_backscatterCoef_from_dataset(hf_db_path, list(tif_input.keys())[0], gamma=gamma, s1_prefix=s1_key,
                                               scale_level=0, saved=True, chunk_size=int(chunk_size/2),
                                               save_file=os.path.join(base_dir, "TEMP.npy"))

    # ------predict the height
    est_model = VVH_model(gamma=gamma, a=0.0, b=0.0, c=0.0)
    est_model.load_model(predictor)
    height_pred = np.reshape(est_model.predict(feature), (ny, nx))
    # ---------if the log_scale == True, the results of prediction is ln(H)
    if log_scale:
        height_pred = np.exp(height_pred)

    # ------save the height into the GeoTiff file
    driver = gdal.GetDriverByName("GTiff")
    height_ds = driver.Create(out_file, nx, ny, 1, gdal.GDT_Float64)
    height_ds.GetRasterBand(1).WriteArray(height_pred)

    aggregate_example = list(tif_input.keys())[0]
    tif_example = tif_input[aggregate_example][s1_key]

    ds = gdal.Open(tif_example)
    geo_trans = ds.GetGeoTransform()
    height_geo_trans = (x_min, resolution, geo_trans[2], y_max, geo_trans[4], -resolution)
    proj = ds.GetProjection()
    height_ds.SetGeoTransform(height_geo_trans)
    height_ds.SetProjection(proj)
    height_ds.FlushCache()
    height_ds = None

    # os.system("rm -rf {0}".format(os.path.join(base_dir, "*TEMP*")))
    tmp_list = glob.glob(os.path.join(base_dir, "*TEMP*"))
    for f in tmp_list:
        os.remove(f)


def pred_height_from_tiff_ML(x_min, y_min, x_max, y_max, out_file, tif_ref, patch_size, predictor, resolution,
                             s1_key="sentinel_1", s2_key="sentinel_2", base_dir=".", chunk_size=100000, num_cpu=1,
                             log_scale=True, **kwargs):
    # ------By default, we assume the parameter `tif_dict` takes the form as follows:
    # ---------------------------------------------------------------------------
    #       {"50pt":
    #               {
    #                   `s1_key`:  `the location of Sentinel-1-50pt GeoTiff file`,
    #                   `s2_key`:  `the location of Sentinel-2-50pt GeoTiff file`,
    #                },
    #        "AnotherAggregation":
    #               {
    #                   `s1_key`:  `the location of Sentinel-1-aggregation GeoTiff file`,
    #                   `s2_key`:  `the location of Sentinel-2-aggregation GeoTiff file`,
    #               },
    #         ......
    #       }
    # ---------------------------------------------------------------------------
    hf_db_path = os.path.join(base_dir, "TEMP.h5")
    hf_db = h5py.File(hf_db_path, mode="w")

    num_s1_band = 2
    num_s2_band = 4
    marker = "@"
    padding = 0.02

    raw_band = []
    for aggregate_key in tif_ref.keys():
        for n in range(1, num_s1_band+1):
            raw_band.append("_".join([s1_key, aggregate_key, "B" + str(n)]))
        for n in range(1, num_s2_band+1):
            raw_band.append("_".join([s2_key, aggregate_key, "B" + str(n)]))

    # ------get provided preprocessors
    if "filter" in kwargs.keys():
        filter = joblib.load(kwargs["filter"])
    else:
        filter = None

    if "scaler" in kwargs.keys():
        scaler = joblib.load(kwargs["scaler"])
    else:
        scaler = None

    if "transformer" in kwargs.keys():
        transformer = joblib.load(kwargs["transformer"])
    else:
        transformer = None

    num_cpu_available = multiprocessing.cpu_count()
    if num_cpu_available < num_cpu:
        print("Use %d CPUs which is available now" % num_cpu_available)
        num_cpu = num_cpu_available

    # ------get features from input GeoTif files
    # ---------[0] check whether the available GeoTiff file satisfies our needs for prediction
    check_extent(tif_ref, x_min, y_min, x_max, y_max, padding)

    # ---------[0] clip the given GeoTiff file to the target extent for memory saving
    tif_input = dict.fromkeys(tif_ref)
    target_extent = [x_min - padding, y_min - padding, x_max + padding, y_max + padding]
    for aggregate_key in tif_ref.keys():
        tif_input[aggregate_key] = {}
        if not isinstance(tif_ref[aggregate_key][s1_key], list):
            s1_input = [tif_ref[aggregate_key][s1_key]]
        else:
            s1_input = tif_ref[aggregate_key][s1_key]

        s1_input_tif = os.path.join(base_dir, "@".join([aggregate_key, s1_key, "TEMP"]) + ".tif")
        # os.system("gdalwarp -te {0} {1} {2}".format(" ".join([str(cc) for cc in target_extent]), " ".join(s1_input), s1_input_tif))
        gdal.Warp(destNameOrDestDS=s1_input_tif, srcDSOrSrcDSTab=s1_input, options=gdal.WarpOptions(outputBounds=target_extent))
        tif_input[aggregate_key][s1_key] = s1_input_tif

        if not isinstance(tif_ref[aggregate_key][s2_key], list):
            s2_input = [tif_ref[aggregate_key][s2_key]]
        else:
            s2_input = tif_ref[aggregate_key][s2_key]

        s2_input_tif = os.path.join(base_dir, "@".join([aggregate_key, s2_key, "TEMP"]) + ".tif")
        # os.system("gdalwarp -te {0} {1} {2}".format(" ".join([str(cc) for cc in target_extent]), " ".join(s2_input), s2_input_tif))
        gdal.Warp(destNameOrDestDS=s2_input_tif, srcDSOrSrcDSTab=s2_input, options=gdal.WarpOptions(outputBounds=target_extent))
        tif_input[aggregate_key][s2_key] = s2_input_tif

    # ---------[1] calculate the coordinates of center points of patches where the building height is to be predicted
    nx = int(np.ceil((x_max - x_min) / resolution))
    ny = int(np.ceil((y_max - y_min) / resolution))
    x_center = np.array([x_min + resolution * (i + 1/2) for i in range(0, nx)])
    y_center = np.array([y_max - resolution * (i + 1/2) for i in range(0, ny)])
    xx_center, yy_center = np.meshgrid(x_center, y_center)
    sample_loc = np.concatenate([yy_center.reshape(-1, 1), xx_center.reshape(-1, 1)], axis=1)

    # ---------[2] divide the whole GeoTiff files into patches whose size is: resolution * resolution
    # ------------use multiprocessing for acceleration
    pool = multiprocessing.Pool(processes=num_cpu)

    sub_chunk_size = int(chunk_size / num_cpu)
    n_sample = len(sample_loc)
    n_sub_chunk = int(np.ceil(n_sample / sub_chunk_size))
    id_sub = {}
    for j in range(0, num_cpu):
        id_sub[j] = []

    for sub_chunk_id in range(0, n_sub_chunk):
        subproc_id = sub_chunk_id % num_cpu
        id_sub[subproc_id].append(sample_loc[sub_chunk_size * sub_chunk_id:sub_chunk_size * (sub_chunk_id + 1)])

    for k in id_sub.keys():
        id_sub[k] = np.concatenate(id_sub[k], axis=0)

    arg_list = [(tif_input, id_sub[j], s1_key, s2_key, list(tif_input.keys()), patch_size,
                 os.path.join(base_dir, "_".join(["TEMP", str(j)])), marker, int(chunk_size/num_cpu)) for j in range(0, num_cpu)]
    res_list = pool.starmap(get_patch_from_tiff, arg_list)
    pool.close()
    pool.join()

    # ---------determine how many chunks we need to divide using the maximum of chunk_id in each subprocess
    res_list_flatten = reduce(lambda x, y: x + y, res_list)
    subproc_chunk = [int(re.findall(r"chunk(\d+).npy", res_file)[0]) for res_file in res_list_flatten]
    n_chunk = max(subproc_chunk)

    for chunk_id in range(0, n_chunk + 1):
        # ------------create (sub-)group for the city
        city_db = hf_db.create_group("TEMP@{0}".format(chunk_id))

        # ------------gather raw band variable
        for band in raw_band:
            band_file = []
            for res_file in res_list_flatten:
                if len(re.findall(r"@{0}_chunk{1}".format(band, chunk_id), res_file)) > 0:
                    band_file.append(res_file)

            band_file_sorted = sorted(band_file, key=lambda x: (int(re.findall(r"TEMP_(\d+)@", x)[0]), int(re.findall(r"_chunk(\d+).npy", x)[0])))
            city_db.create_dataset(band, data=np.concatenate([np.load(f) for f in band_file_sorted]))

    # os.system("rm -rf {0}".format(os.path.join(base_dir, "*TEMP*.npy")))
    tmp_list = glob.glob(os.path.join(base_dir, "*TEMP*.npy"))
    for f in tmp_list:
        os.remove(f)

    # ---------[3] calculate the feature of each patch and then do some preprocessing with it
    feature = get_feature_from_dataset(hf_db_path, list(tif_input.keys()), s1_prefix=s1_key, s2_prefix=s2_key,
                                       scale_level=0, saved=True, num_cpu=num_cpu, chunk_size=int(chunk_size/2),
                                       save_file=os.path.join(base_dir, "TEMP.npy"))

    if filter is not None:
        feature = filter.transform(feature)

    if scaler is not None:
        feature = scaler.transform(feature)

    if transformer is not None:
        feature = transformer.transform(feature)

    # ------predict the height
    est_model = joblib.load(predictor)
    height_pred = np.reshape(est_model.predict(feature), (ny, nx))
    # ---------if the log_scale == True, the results of prediction is ln(H)
    if log_scale:
        height_pred = np.exp(height_pred)

    # ------save the height into the GeoTiff file
    driver = gdal.GetDriverByName("GTiff")
    height_ds = driver.Create(out_file, nx, ny, 1, gdal.GDT_Float64)
    height_ds.GetRasterBand(1).WriteArray(height_pred)

    aggregate_example = list(tif_input.keys())[0]
    tif_example = tif_input[aggregate_example][s1_key]

    ds = gdal.Open(tif_example)
    geo_trans = ds.GetGeoTransform()
    height_geo_trans = (x_min, resolution, geo_trans[2], y_max, geo_trans[4], -resolution)
    proj = ds.GetProjection()
    height_ds.SetGeoTransform(height_geo_trans)
    height_ds.SetProjection(proj)
    height_ds.FlushCache()
    height_ds = None

    # os.system("rm -rf {0}".format(os.path.join(base_dir, "*TEMP*")))
    tmp_list = glob.glob(os.path.join(base_dir, "*TEMP*"))
    for f in tmp_list:
        os.remove(f)


def pred_height_from_tiff_DL_patch(extent: list, out_file: str, tif_ref: dict, patch_size: list, predictor: str, trained_record: str,
                                   resolution: float, s1_key="sentinel_1", s2_key="sentinel_2", aux_feat_info=None,
                                   base_dir=".", padding=0.005, batch_size=4, tmp_suffix=None,
                                   activation=None, log_scale=False, cuda_used=False, v_min=None, v_max=None):
    """Predict building height or footprint using Deep-Learing-based (DL) models trained by Single Task Learning (STL).

    Parameters
    ----------

    extent : list
        A list in the format `[x_min, y_min, x_max, y_max]` which specifies the extent of the target area.
    out_file : str
        Output Path for the predicted result.
    output_grid : str
        Output Path for the building footprint density raster layer.
    tif_ref : dict
        A dictionary which specifies the path of input Sentinel-1/2's files under different temporal aggregation.
        An example can be given as follows:
            {"50pt":
                   {
                       `s1_key`:  `path of Sentinel-1-50pt GeoTiff file`,
                       `s2_key`:  `path of Sentinel-2-50pt GeoTiff file`,
                    },
            "AggOps":
                   {
                       `s1_key`:  `path of Sentinel-1-aggregation GeoTiff file`,
                       `s2_key`:  `path of Sentinel-2-aggregation GeoTiff file`,
                   },
             ......
           }
    patch_size : list
        A list which specifies the size of input patch.
        It can be chosen from: `[15]`, `[30]`, `[60]`, `[120]`.
    predictor : str
        Name of DL models used for prediction.
        It can be chosen from: `ResNet18`, `SEResNet18`, `CBAMResNet18`.
    trained_record : str
        Path to the pretrained model file.
    resolution : float
        Target resolution for building information mapping.
        It can be chosen from: `0.0009`, `0.00225`, `0.0045`, `0.009`.
    s1_key : str
        Key in `tif_ref[AggOps]` which indicates the path of Sentinel-1's files.
        The default is `sentinel_1`.
    s2_key : str
        Key in `tif_ref[AggOps]` which indicates the path of Sentinel-2's files.
        The default is `sentinel_2`.
    aux_feat_info: dict
        A dictionary which contains the auxiliary feature information including `patch_size_ratio` and `path`.
        The default is `None`.
    base_dir : str
        Path of a directory for temporary results saving during prediction.
        The default is `.`.
    padding : float
        Padding size outside the target region (in degrees).
        The default is `0.005`.
    batch_size : int
        Batch size for inference。
        The default is `4`.
    tmp_suffix : str
        Naming suffix for temporary files.
        The default is `None`.
    activation : str
        Activation function for model output.
        It can be chosen from: `relu`, `sigmoid`. The default is `None`.
    log_scale: boolean
        A flag which controls whether log-transformation is used for target variable.
        The default is `False`.
    cuda_used: boolean
        A flag which controls whether CUDA is used for inference.
        The default is `False`.
    v_min: float
        Lower bound for prediction.
    v_max: float
        Upper bound for prediction.

    """

    x_min, y_min, x_max, y_max = extent

    num_s1_band = 2
    num_s2_band = 4
    # scale = 0.0001
    scale = 1.0
    band_ref = {"VV": "B1", "VH": "B2", "R": "B1", "G": "B2", "B": "B3", "NIR": "B4"}

    # ------[1] load the PyTorch model
    if patch_size[0] == 15:
        in_plane = 64
        num_block = 2
    elif patch_size[0] == 30:
        in_plane = 64
        num_block = 1
    elif patch_size[0] == 60:
        in_plane = 64
        num_block = 1
    else:
        in_plane = 64
        num_block = 1

    cuda_used = (cuda_used and torch.cuda.is_available())

    aux_namelist = None
    aux_size = None
    if aux_feat_info is not None:
        aux_namelist = sorted(aux_feat_info.keys())
        aux_size = int(aux_feat_info[aux_namelist[0]]["patch_size_ratio"] * patch_size[0])
    
    if predictor == "ResNet18":
        if aux_feat_info is None:
            est_model = model_ResNet(in_plane=in_plane, input_channels=6, input_size=patch_size[0],
                                        num_block=num_block, log_scale=log_scale, activation=activation,
                                        cuda_used=cuda_used, trained_record=trained_record)
        else:
            est_model = model_ResNet_aux(in_plane=in_plane, input_channels=6, input_size=patch_size[0],
                                            aux_input_size=aux_size, num_aux=len(aux_namelist),
                                            num_block=num_block, log_scale=log_scale, activation=activation,
                                            cuda_used=cuda_used, trained_record=trained_record)
    elif predictor == "SEResNet18":
        if aux_namelist is None:
            est_model = model_SEResNet(in_plane=in_plane, input_channels=6, input_size=patch_size[0],
                                        num_block=num_block, log_scale=log_scale, activation=activation,
                                        cuda_used=cuda_used, trained_record=trained_record)
        else:
            est_model = model_SEResNet_aux(in_plane=in_plane, input_channels=6, input_size=patch_size[0],
                                            aux_input_size=aux_size, num_aux=len(aux_namelist),
                                            num_block=num_block, log_scale=log_scale, activation=activation,
                                            cuda_used=cuda_used, trained_record=trained_record)
    elif predictor == "CBAMResNet18":
        if aux_namelist is None:
            est_model = model_CBAMResNet(in_plane=in_plane, input_channels=6, input_size=patch_size[0],
                                            num_block=num_block, log_scale=log_scale, activation=activation,
                                            cuda_used=cuda_used, trained_record=trained_record)
        else:
            est_model = model_CBAMResNet_aux(in_plane=in_plane, input_channels=6, input_size=patch_size[0],
                                                aux_input_size=aux_size, num_aux=len(aux_namelist),
                                                num_block=num_block, log_scale=log_scale, activation=activation,
                                                cuda_used=cuda_used, trained_record=trained_record)
    else:
        raise NotImplementedError

    cuda_used = (cuda_used and torch.cuda.is_available())

    if cuda_used:
        est_model = est_model.cuda()

    est_model.eval()
    '''
    for m in model.modules():
        for child in m.children():
            if type(child) == nn.BatchNorm2d:
                child.track_running_stats = False
    '''

    # ------[2] prepare the input tiff
    # ---------check whether the available GeoTiff file satisfies our needs for prediction
    check_extent(base_dir, tif_ref, x_min, y_min, x_max, y_max, padding)

    # ---------clip the given GeoTiff file to the target extent for memory saving
    tif_input = dict.fromkeys(tif_ref)
    target_extent = [x_min - padding, y_min - padding, x_max + padding, y_max + padding]

    for aggregate_key in tif_ref.keys():
        tif_input[aggregate_key] = {}
        if not isinstance(tif_ref[aggregate_key][s1_key], list):
            s1_input = [tif_ref[aggregate_key][s1_key]]
        else:
            s1_input = tif_ref[aggregate_key][s1_key]

        s1_input_tif = os.path.join(base_dir, "@".join([aggregate_key, s1_key, "TEMP"]) + str(tmp_suffix or '') + ".tif")
        # os.system("gdalwarp -te {0} {1} {2}".format(" ".join([str(cc) for cc in target_extent]), " ".join(s1_input), s1_input_tif))
        gdal.Warp(destNameOrDestDS=s1_input_tif, srcDSOrSrcDSTab=s1_input, options=gdal.WarpOptions(outputBounds=target_extent))
        tif_input[aggregate_key][s1_key] = s1_input_tif

        if not isinstance(tif_ref[aggregate_key][s2_key], list):
            s2_input = [tif_ref[aggregate_key][s2_key]]
        else:
            s2_input = tif_ref[aggregate_key][s2_key]

        s2_input_tif = os.path.join(base_dir, "@".join([aggregate_key, s2_key, "TEMP"]) + str(tmp_suffix or '') + ".tif")
        # os.system("gdalwarp -te {0} {1} {2}".format(" ".join([str(cc) for cc in target_extent]), " ".join(s2_input), s2_input_tif))
        gdal.Warp(destNameOrDestDS=s2_input_tif, srcDSOrSrcDSTab=s2_input, options=gdal.WarpOptions(outputBounds=target_extent))
        tif_input[aggregate_key][s2_key] = s2_input_tif

    # ----------store the value of TiffArray into a dict in advance where the key is its file name
    channel_ref = {}
    input_band = {}
    tiffArr = {}
    tiffGeoTransform = {}
    for aggregate_key in tif_ref.keys():
        channel_ref[aggregate_key] = {"S1_VV": "_".join([s1_key, aggregate_key, band_ref["VV"]]),
                                      "S1_VH": "_".join([s1_key, aggregate_key, band_ref["VH"]]),
                                      "S2_RGB": ["_".join([s2_key, aggregate_key, band_ref["R"]]),
                                                 "_".join([s2_key, aggregate_key, band_ref["G"]]),
                                                 "_".join([s2_key, aggregate_key, band_ref["B"]])],
                                      "S2_NIR": "_".join([s2_key, aggregate_key, band_ref["NIR"]])}
        for n in range(1, num_s1_band + 1):
            s1_band_name = "_".join([s1_key, aggregate_key, "B" + str(n)])
            input_band[s1_band_name] = []
        for n in range(1, num_s2_band + 1):
            s2_band_name = "_".join([s2_key, aggregate_key, "B" + str(n)])
            input_band[s2_band_name] = []

        sub_dict = tif_input[aggregate_key]
        # ---------read the target Sentinel-1 data
        s1_name = "_".join([s1_key, aggregate_key])
        tiffArr[s1_name] = GetArrayFromTiff(sub_dict[s1_key])
        # ------------fill the NaN value in the band data using Nearest-neighbor interpolation
        # ------------note that for inference, we fill NaNs at the beginning (different from training patch collection)
        for band_id in range(0, num_s1_band):
            tiffArr[s1_name][band_id] = fill_nan_nearest(tiffArr[s1_name][band_id])
        tiffGeoTransform[s1_name] = GetGeoTransformFromTiff(sub_dict[s1_key])
        # ---------read the target Sentinel-2 data
        s2_name = "_".join([s2_key, aggregate_key])
        tiffArr[s2_name] = GetArrayFromTiff(sub_dict[s2_key])
        for band_id in range(0, num_s2_band):
            tiffArr[s2_name][band_id] = fill_nan_nearest(tiffArr[s2_name][band_id])
            tiffArr[s2_name][band_id] = rgb_rescale_band(tiffArr[s2_name][band_id])
        tiffGeoTransform[s2_name] = GetGeoTransformFromTiff(sub_dict[s2_key])
    
    if aux_namelist is not None:
        for feat in aux_namelist:
            if not isinstance(tif_ref[aggregate_key][s2_key], list):
                aux_feat_input = [aux_feat_info[feat]["path"]]
            else:
                aux_feat_input = aux_feat_info[feat]["path"]
            
            aux_feat_input_tif = os.path.join(base_dir, "@".join([aggregate_key, feat, "TEMP"]) + str(tmp_suffix or '') + ".tif")
            # os.system("gdalwarp -te {0} {1} {2}".format(" ".join([str(cc) for cc in target_extent]), " ".join(aux_feat_input), aux_feat_input_tif))
            gdal.Warp(destNameOrDestDS=aux_feat_input_tif, srcDSOrSrcDSTab=aux_feat_input, options=gdal.WarpOptions(outputBounds=target_extent))
            tiffArr[feat] = GetArrayFromTiff(aux_feat_input_tif)
            tiffGeoTransform[feat] = GetGeoTransformFromTiff(aux_feat_input_tif)

    # ------[3] height mapping on the given tiff array
    # ---------calculate the coordinates of center points of patches where the building height is to be predicted
    # ---------note that for floating number ops, we may have: 0.1 + 0.2 = 0.30000004
    nx = int(np.ceil(round(round(x_max - x_min, 6) / resolution, 6)))
    ny = int(np.ceil(round(round(y_max - y_min, 6) / resolution, 6)))
    
    x_center = np.array([x_min + resolution * (i + 1/2) for i in range(0, nx)])
    y_center = np.array([y_max - resolution * (i + 1/2) for i in range(0, ny)])

    tensor_transform = transforms.ToTensor()

    height_pred = []
    num_pixel = nx * ny
    num_batch = int(np.ceil(num_pixel / batch_size))

    for batch_id in range(0, num_batch):
        id_shift = batch_id * batch_size
        batch_input = []

        if aux_namelist is not None:
            aux_batch_input = []
        else:
            aux_batch_input = None

        for i in range(0, batch_size):
            pid = id_shift + i
            if pid < num_pixel:
                yid = int(np.floor(pid / nx))
                xid = int(pid - yid * nx)

                y_loc = y_center[yid]
                x_loc = x_center[xid]
                patch_multi_agg = []
                # ------------gather the feature patch centered on this selected pixel
                for aggregate_key in tif_input.keys():
                    s1_name = "_".join([s1_key, aggregate_key])
                    # s1_patch.shape: (len(patch_size) * num_band, max(patch_size), max(patch_size))
                    s1_patch = GetPatchFromTiffArray(tiff_array=tiffArr[s1_name],
                                                     tiff_geoTransform=tiffGeoTransform[s1_name],
                                                     x=x_loc, y=y_loc, patch_size=patch_size)

                    for n in range(0, num_s1_band):
                        s1_feature_name = "_".join([s1_key, aggregate_key, "B" + str(n + 1)])
                        # ---------------Note that s1_patch has #num_patch_size 2D arrays of #num_s1_bands
                        # --------------i.e. the first dimension of s1_patch = [B1_size1, B2_size1, B1_size2, B2_size2, ...]
                        input_band[s1_feature_name] = s1_patch[n:len(s1_patch):num_s1_band]

                    s2_name = "_".join([s2_key, aggregate_key])
                    s2_patch = GetPatchFromTiffArray(tiff_array=tiffArr[s2_name],
                                                     tiff_geoTransform=tiffGeoTransform[s2_name],
                                                     x=x_loc, y=y_loc, patch_size=patch_size)

                    for n in range(0, num_s2_band):
                        s2_feature_name = "_".join([s2_key, aggregate_key, "B" + str(n + 1)])
                        input_band[s2_feature_name] = s2_patch[n:len(s2_patch):num_s2_band]
                
                # ---------gather the feature information centered on this selected pixel
                if aux_namelist is not None:
                    for feat in aux_namelist:
                        aux_patch = GetPatchFromTiffArray(tiff_array=tiffArr[feat], tiff_geoTransform=tiffGeoTransform[feat],
                                                        x=x_loc, y=y_loc, patch_size=[aux_size])
                        input_band[feat] = aux_patch

                '''
                # ------------fill NaN values
                nan_ratio_dict = dict((k, calc_nan_ratio(input_band[k][0])) for k in input_band.keys())
                for k, v in input_band.items():
                    if nan_ratio_dict[k] > 0:
                        input_band[k] = np.expand_dims(fill_nan_nearest(v[0]), axis=0)
                '''

                # ------------patch preprocessing
                for aggregation_ops in channel_ref.keys():
                    s1_vv_name = channel_ref[aggregation_ops]["S1_VV"]
                    s1_vv_patch = get_backscatterCoef(input_band[s1_vv_name]) * 255
                    s1_vv_patch = np.transpose(s1_vv_patch, (1, 2, 0))

                    s1_vh_name = channel_ref[aggregation_ops]["S1_VH"]
                    s1_vh_patch = get_backscatterCoef(input_band[s1_vh_name]) * 255
                    s1_vh_patch = np.transpose(s1_vh_patch, (1, 2, 0))

                    s2_rgb_patch_tmp = []
                    s2_band_namelist = channel_ref[aggregation_ops]["S2_RGB"]
                    # ---------shape of s2_patch_tmp: (num_s1_band, num_scale, size_y, size_x)
                    for s2_band in s2_band_namelist:
                        s2_rgb_patch_tmp.append(input_band[s2_band])
                    s2_rgb_patch = np.concatenate(s2_rgb_patch_tmp, axis=0) * scale * 255
                    s2_rgb_patch = np.transpose(s2_rgb_patch, (1, 2, 0))
                    # s2_rgb_patch = rgb_rescale(s2_rgb_patch, vmin=0, vmax=255, axis=(0, 1))
                    #s2_rgb_patch = clahe_transform(image=s2_rgb_patch.astype(np.uint8))["image"]

                    s2_nir_name = channel_ref[aggregation_ops]["S2_NIR"]
                    s2_nir_patch = input_band[s2_nir_name] * scale * 255
                    s2_nir_patch = np.transpose(s2_nir_patch, (1, 2, 0))
                    # s2_nir_patch = rgb_rescale(s2_nir_patch, vmin=0, vmax=255, axis=(0, 1))
                    #s2_nir_patch = clahe_transform(image=s2_nir_patch.astype(np.uint8))["image"]

                    patch = np.concatenate([s1_vv_patch.astype(np.uint8), s1_vh_patch.astype(np.uint8),
                                            s2_rgb_patch.astype(np.uint8), s2_nir_patch.astype(np.uint8)], axis=-1)

                    # ---------convert numpy.ndarray of shape [H, W, C] to torch.tensor [C, H, W]
                    patch = tensor_transform(patch).type(torch.FloatTensor)
                    patch_multi_agg.append(patch)

                feat = torch.cat(patch_multi_agg, dim=0)
                feat = torch.unsqueeze(feat, dim=0)
                batch_input.append(feat)

                if aux_namelist is not None:
                    aux_feat = []
                    for k in aux_namelist:
                        aux_patch = input_band[feat]
                        aux_patch = np.transpose(aux_patch, (1, 2, 0))
                        aux_feat.append(tensor_transform(aux_patch).type(torch.FloatTensor))
                    aux_feat = torch.cat(aux_feat, dim=0)
                    aux_feat = torch.unsqueeze(aux_feat, dim=0)

                    aux_batch_input.append(aux_feat)

        batch_input = torch.cat(batch_input, dim=0)
        if aux_batch_input is not None:
            aux_batch_input = torch.cat(aux_batch_input, dim=0)

        if cuda_used:
            batch_input = batch_input.cuda()
            if aux_batch_input is not None:
                aux_batch_input = aux_batch_input.cuda()

        if aux_batch_input is not None:
            with torch.no_grad():
                output = est_model(batch_input)
        else:
            with torch.no_grad():
                output = est_model(batch_input, aux_batch_input)

        output = torch.squeeze(output)
        output = output.cpu().numpy()
        height_pred.append(output)

    height_pred = np.concatenate(height_pred, 0).reshape((ny, nx))

    if log_scale:
        height_pred = np.exp(height_pred)

    if v_max is not None:
        height_pred = np.where(height_pred > v_max, v_max, height_pred)
    
    if v_min is not None:
        height_pred = np.where(height_pred < v_min, 0.0, height_pred)

    # ------save the height into the GeoTiff file
    driver = gdal.GetDriverByName("GTiff")
    height_ds = driver.Create(out_file, nx, ny, 1, gdal.GDT_Float64)
    height_ds.GetRasterBand(1).WriteArray(height_pred)

    aggregate_example = list(tif_input.keys())[0]
    tif_example = tif_input[aggregate_example][s1_key]

    ds = gdal.Open(tif_example)
    geo_trans = ds.GetGeoTransform()
    height_geo_trans = (x_min, resolution, geo_trans[2], y_max, geo_trans[4], -resolution)
    proj = ds.GetProjection()
    height_ds.SetGeoTransform(height_geo_trans)
    height_ds.SetProjection(proj)
    height_ds.FlushCache()
    height_ds = None


def pred_height_from_tiff_DL_patch_MTL(extent: list, out_footprint_file: str, out_height_file: str, tif_ref: dict,
                                       patch_size: list, predictor: str, trained_record: str, resolution: float,
                                       s1_key="sentinel_1", s2_key="sentinel_2", aux_feat_info=None,
                                       crossed=False, base_dir=".", padding=0.005, batch_size=1, tmp_suffix=None,
                                       log_scale=True, cuda_used=False,
                                       h_min=0.0, h_max=None, f_min=0.0, f_max=None):
    """Predict building height and footprint using Deep-Learing-based (DL) models trained by Multi-Task Learning (STL).

    Parameters
    ----------

    extent : list
        A list in the format `[x_min, y_min, x_max, y_max]` which specifies the extent of the target area.
    out_footprint_file : str
        Output Path for the predicted building footprint result.
    out_height_file: str
        Output Path for the predicted building height result.
    tif_ref : dict
        A dictionary which specifies the path of input Sentinel-1/2's files under different temporal aggregation.
        An example can be given as follows:
            {"50pt":
                   {
                       `s1_key`:  `path of Sentinel-1-50pt GeoTiff file`,
                       `s2_key`:  `path of Sentinel-2-50pt GeoTiff file`,
                    },
            "AggOps":
                   {
                       `s1_key`:  `path of Sentinel-1-aggregation GeoTiff file`,
                       `s2_key`:  `path of Sentinel-2-aggregation GeoTiff file`,
                   },
             ......
           }
    patch_size : list
        A list which specifies the size of input patch.
        It can be chosen from: `[15]`, `[30]`, `[60]`, `[120]`.
    predictor : str
        Name of DL models used for prediction.
        It can be chosen from: `ResNet18`, `SEResNet18`, `CBAMResNet18`.
    trained_record : str
        Path to the pretrained model file.
    resolution : float
        Target resolution for building information mapping.
        It can be chosen from: `0.0009`, `0.00225`, `0.0045`, `0.009`.
    s1_key : str
        Key in `tif_ref[AggOps]` which indicates the path of Sentinel-1's files.
        The default is `sentinel_1`.
    s2_key : str
        Key in `tif_ref[AggOps]` which indicates the path of Sentinel-2's files.
        The default is `sentinel_2`.
    aux_feat_info: dict
        A dictionary which contains the auxiliary feature information including `patch_size_ratio` and `path`.
        The default is `None`.
    crossed: boolean
        A flag which controls whether a link is to be added between last fully-connected layers of building footprint and height prediction.
        The default is `False`.
    base_dir : str
        Path of a directory for temporary results saving during prediction.
        The default is `.`.
    padding : float
        Padding size outside the target region (in degrees).
        The default is `0.005`.
    batch_size : int
        Batch size for inference。
        The default is `4`.
    tmp_suffix : str
        Naming suffix for temporary files.
        The default is `None`.
    activation : str
        Activation function for model output.
        It can be chosen from: `relu`, `sigmoid`. The default is `None`.
    log_scale: boolean
        A flag which controls whether log-transformation is used for target variable.
        The default is `False`.
    cuda_used: boolean
        A flag which controls whether CUDA is used for inference.
        The default is `False`.
    h_min: float
        Lower bound for building height prediction.
        The default is `0.0`.
    h_max: float
        Upper bound for building height prediction.
        The default is `None`.
    f_min: float
        Lower bound for building footprint prediction.
        The default is `0.0`.
    f_max: float
        Upper bound for building footprint prediction.
        The default is `None`.

    """

    x_min, y_min, x_max, y_max = extent
    
    num_s1_band = 2
    num_s2_band = 4
    # scale = 0.0001
    scale = 1.0
    band_ref = {"VV": "B1", "VH": "B2", "R": "B1", "G": "B2", "B": "B3", "NIR": "B4"}

    # ------[1] load the PyTorch model
    if patch_size[0] == 15:
        in_plane = 64
        num_block = 2
    elif patch_size[0] == 30:
        in_plane = 64
        num_block = 1
    elif patch_size[0] == 60:
        in_plane = 64
        num_block = 1
    else:
        in_plane = 64
        num_block = 1

    cuda_used = (cuda_used and torch.cuda.is_available())

    aux_namelist = None
    aux_size = None
    if aux_feat_info is not None:
        aux_namelist = sorted(aux_feat_info.keys())
        aux_size = int(aux_feat_info[aux_namelist[0]]["patch_size_ratio"] * patch_size[0])

    if predictor == "ResNet18":
        if aux_feat_info is None:
            est_model = model_ResNetMTL(in_plane=in_plane, input_channels=6, input_size=patch_size[0],
                                            num_block=num_block, log_scale=log_scale, crossed=crossed,
                                            cuda_used=cuda_used, trained_record=trained_record)
        else:
            est_model = model_ResNetMTL_aux(in_plane=in_plane, input_channels=6, input_size=patch_size[0],
                                                aux_input_size=aux_size, num_aux=len(aux_namelist),
                                                num_block=num_block, log_scale=log_scale, crossed=crossed,
                                                cuda_used=cuda_used, trained_record=trained_record)
    elif predictor == "SEResNet18":
        if aux_namelist is None:
            est_model = model_SEResNetMTL(in_plane=in_plane, input_channels=6, input_size=patch_size[0],
                                                num_block=num_block, log_scale=log_scale, crossed=crossed,
                                                cuda_used=cuda_used, trained_record=trained_record)
        else:
            est_model = model_SEResNetMTL_aux(in_plane=in_plane, input_channels=6, input_size=patch_size[0],
                                                aux_input_size=aux_size, num_aux=len(aux_namelist),
                                                num_block=num_block, log_scale=log_scale, crossed=crossed,
                                                cuda_used=cuda_used, trained_record=trained_record)
    elif predictor == "CBAMResNet18":
        if aux_namelist is None:
            est_model = model_CBAMResNetMTL(in_plane=in_plane, input_channels=6, input_size=patch_size[0],
                                                num_block=num_block, log_scale=log_scale, crossed=crossed,
                                                cuda_used=cuda_used, trained_record=trained_record)
        else:
            est_model = model_CBAMResNetMTL_aux(in_plane=in_plane, input_channels=6, input_size=patch_size[0],
                                                    aux_input_size=aux_size, num_aux=len(aux_namelist),
                                                    num_block=num_block, log_scale=log_scale, crossed=crossed,
                                                    cuda_used=cuda_used, trained_record=trained_record)
    else:
        raise NotImplementedError

    if cuda_used:
        est_model = est_model.cuda()

    est_model.eval()

    # ------[2] prepare the input tiff
    # ---------check whether the available GeoTiff file satisfies our needs for prediction
    check_extent(base_dir, tif_ref, x_min, y_min, x_max, y_max, padding)

    # ---------clip the given GeoTiff file to the target extent for memory saving
    tif_input = dict.fromkeys(tif_ref)
    target_extent = [x_min - padding, y_min - padding, x_max + padding, y_max + padding]
    for aggregate_key in tif_ref.keys():
        tif_input[aggregate_key] = {}
        if not isinstance(tif_ref[aggregate_key][s1_key], list):
            s1_input = [tif_ref[aggregate_key][s1_key]]
        else:
            s1_input = tif_ref[aggregate_key][s1_key]

        s1_input_tif = os.path.join(base_dir, "@".join([aggregate_key, s1_key, "TEMP"]) + str(tmp_suffix or '') + ".tif")
        # os.system("gdalwarp -te {0} {1} {2}".format(" ".join([str(cc) for cc in target_extent]), " ".join(s1_input), s1_input_tif))
        gdal.Warp(destNameOrDestDS=s1_input_tif, srcDSOrSrcDSTab=s1_input, options=gdal.WarpOptions(outputBounds=target_extent))
        tif_input[aggregate_key][s1_key] = s1_input_tif

        if not isinstance(tif_ref[aggregate_key][s2_key], list):
            s2_input = [tif_ref[aggregate_key][s2_key]]
        else:
            s2_input = tif_ref[aggregate_key][s2_key]

        s2_input_tif = os.path.join(base_dir, "@".join([aggregate_key, s2_key, "TEMP"]) + str(tmp_suffix or '') + ".tif")
        # os.system("gdalwarp -te {0} {1} {2}".format(" ".join([str(cc) for cc in target_extent]), " ".join(s2_input), s2_input_tif))
        gdal.Warp(destNameOrDestDS=s2_input_tif, srcDSOrSrcDSTab=s2_input, options=gdal.WarpOptions(outputBounds=target_extent))
        tif_input[aggregate_key][s2_key] = s2_input_tif

    # ----------store the value of TiffArray into a dict in advance where the key is its file name
    channel_ref = {}
    input_band = {}
    tiffArr = {}
    tiffGeoTransform = {}
    for aggregate_key in tif_ref.keys():
        channel_ref[aggregate_key] = {"S1_VV": "_".join([s1_key, aggregate_key, band_ref["VV"]]),
                                      "S1_VH": "_".join([s1_key, aggregate_key, band_ref["VH"]]),
                                      "S2_RGB": ["_".join([s2_key, aggregate_key, band_ref["R"]]),
                                                 "_".join([s2_key, aggregate_key, band_ref["G"]]),
                                                 "_".join([s2_key, aggregate_key, band_ref["B"]])],
                                      "S2_NIR": "_".join([s2_key, aggregate_key, band_ref["NIR"]])}
        for n in range(1, num_s1_band + 1):
            s1_band_name = "_".join([s1_key, aggregate_key, "B" + str(n)])
            input_band[s1_band_name] = []
        for n in range(1, num_s2_band + 1):
            s2_band_name = "_".join([s2_key, aggregate_key, "B" + str(n)])
            input_band[s2_band_name] = []

        sub_dict = tif_input[aggregate_key]
        # ---------read the target Sentinel-1 data
        s1_name = "_".join([s1_key, aggregate_key])
        tiffArr[s1_name] = GetArrayFromTiff(sub_dict[s1_key])
        # ------------fill the NaN value in the band data using Nearest-neighbor interpolation
        # ------------note that for inference, we fill NaNs at the beginning (different from training patch collection)
        for band_id in range(0, num_s1_band):
            tiffArr[s1_name][band_id] = fill_nan_nearest(tiffArr[s1_name][band_id])
        tiffGeoTransform[s1_name] = GetGeoTransformFromTiff(sub_dict[s1_key])
        # ---------read the target Sentinel-2 data
        s2_name = "_".join([s2_key, aggregate_key])
        tiffArr[s2_name] = GetArrayFromTiff(sub_dict[s2_key])
        for band_id in range(0, num_s2_band):
            tiffArr[s2_name][band_id] = fill_nan_nearest(tiffArr[s2_name][band_id])
            tiffArr[s2_name][band_id] = rgb_rescale_band(tiffArr[s2_name][band_id])
        tiffGeoTransform[s2_name] = GetGeoTransformFromTiff(sub_dict[s2_key])
    
    if aux_namelist is not None:
        for feat in aux_namelist:
            if not isinstance(tif_ref[aggregate_key][s2_key], list):
                aux_feat_input = [aux_feat_info[feat]["path"]]
            else:
                aux_feat_input = aux_feat_info[feat]["path"]
            
            aux_feat_input_tif = os.path.join(base_dir, "@".join([aggregate_key, feat, "TEMP"]) + str(tmp_suffix or '') + ".tif")
            # os.system("gdalwarp -te {0} {1} {2}".format(" ".join([str(cc) for cc in target_extent]), " ".join(aux_feat_input), aux_feat_input_tif))
            gdal.Warp(destNameOrDestDS=aux_feat_input_tif, srcDSOrSrcDSTab=aux_feat_input, options=gdal.WarpOptions(outputBounds=target_extent))
            tiffArr[feat] = GetArrayFromTiff(aux_feat_input_tif)
            tiffGeoTransform[feat] = GetGeoTransformFromTiff(aux_feat_input_tif)

    # ------[3] height mapping on the given tiff array
    # ---------calculate the coordinates of center points of patches where the building height is to be predicted
    nx = int(np.ceil(round(round(x_max - x_min, 6) / resolution, 6)))
    ny = int(np.ceil(round(round(y_max - y_min, 6) / resolution, 6)))
    x_center = np.array([x_min + resolution * (i + 1/2) for i in range(0, nx)])
    y_center = np.array([y_max - resolution * (i + 1/2) for i in range(0, ny)])

    tensor_transform = transforms.ToTensor()

    footprint_pred = []
    height_pred = []

    height_pred = []
    num_pixel = nx * ny
    num_batch = int(np.ceil(num_pixel / batch_size))

    for batch_id in range(0, num_batch):
        id_shift = batch_id * batch_size
        batch_input = []

        if aux_namelist is not None:
            aux_batch_input = []
        else:
            aux_batch_input = None

        for i in range(0, batch_size):
            pid = id_shift + i
            if pid < num_pixel:
                yid = int(np.floor(pid / nx))
                xid = int(pid - yid * nx)

                y_loc = y_center[yid]
                x_loc = x_center[xid]

                patch_multi_agg = []
                # ------------gather the feature patch centered on this selected pixel
                for aggregate_key in tif_input.keys():
                    s1_name = "_".join([s1_key, aggregate_key])
                    # s1_patch.shape: (len(patch_size) * num_band, max(patch_size), max(patch_size))
                    s1_patch = GetPatchFromTiffArray(tiff_array=tiffArr[s1_name],
                                                     tiff_geoTransform=tiffGeoTransform[s1_name],
                                                     x=x_loc, y=y_loc, patch_size=patch_size)

                    for n in range(0, num_s1_band):
                        s1_feature_name = "_".join([s1_key, aggregate_key, "B" + str(n + 1)])
                        # ---------------Note that s1_patch has #num_patch_size 2D arrays of #num_s1_bands
                        # --------------i.e. the first dimension of s1_patch = [B1_size1, B2_size1, B1_size2, B2_size2, ...]
                        input_band[s1_feature_name] = s1_patch[n:len(s1_patch):num_s1_band]

                    s2_name = "_".join([s2_key, aggregate_key])
                    s2_patch = GetPatchFromTiffArray(tiff_array=tiffArr[s2_name],
                                                     tiff_geoTransform=tiffGeoTransform[s2_name],
                                                     x=x_loc, y=y_loc, patch_size=patch_size)

                    for n in range(0, num_s2_band):
                        s2_feature_name = "_".join([s2_key, aggregate_key, "B" + str(n + 1)])
                        input_band[s2_feature_name] = s2_patch[n:len(s2_patch):num_s2_band]

                '''
                # ------------fill NaN values
                nan_ratio_dict = dict((k, calc_nan_ratio(input_band[k][0])) for k in input_band.keys())
                for k, v in input_band.items():
                    if nan_ratio_dict[k] > 0:
                        input_band[k] = np.expand_dims(fill_nan_nearest(v[0]), axis=0)
                '''

                # ------------patch preprocessing
                for aggregation_ops in channel_ref.keys():
                    s1_vv_name = channel_ref[aggregation_ops]["S1_VV"]
                    s1_vv_patch = get_backscatterCoef(input_band[s1_vv_name]) * 255
                    s1_vv_patch = np.transpose(s1_vv_patch, (1, 2, 0))

                    s1_vh_name = channel_ref[aggregation_ops]["S1_VH"]
                    s1_vh_patch = get_backscatterCoef(input_band[s1_vh_name]) * 255
                    s1_vh_patch = np.transpose(s1_vh_patch, (1, 2, 0))

                    s2_rgb_patch_tmp = []
                    s2_band_namelist = channel_ref[aggregation_ops]["S2_RGB"]
                    # ---------shape of s2_patch_tmp: (num_s1_band, num_scale, size_y, size_x)
                    for s2_band in s2_band_namelist:
                        s2_rgb_patch_tmp.append(input_band[s2_band])
                    s2_rgb_patch = np.concatenate(s2_rgb_patch_tmp, axis=0) * scale * 255
                    s2_rgb_patch = np.transpose(s2_rgb_patch, (1, 2, 0))
                    # s2_rgb_patch = rgb_rescale(s2_rgb_patch, vmin=0, vmax=255, axis=(0, 1))
                    #s2_rgb_patch = clahe_transform(image=s2_rgb_patch.astype(np.uint8))["image"]

                    s2_nir_name = channel_ref[aggregation_ops]["S2_NIR"]
                    s2_nir_patch = input_band[s2_nir_name] * scale * 255
                    s2_nir_patch = np.transpose(s2_nir_patch, (1, 2, 0))
                    # s2_nir_patch = rgb_rescale(s2_nir_patch, vmin=0, vmax=255, axis=(0, 1))
                    #s2_nir_patch = clahe_transform(image=s2_nir_patch.astype(np.uint8))["image"]

                    patch = np.concatenate([s1_vv_patch.astype(np.uint8), s1_vh_patch.astype(np.uint8),
                                            s2_rgb_patch.astype(np.uint8), s2_nir_patch.astype(np.uint8)], axis=-1)

                    # ---------convert numpy.ndarray of shape [H, W, C] to torch.tensor [C, H, W]
                    patch = tensor_transform(patch).type(torch.FloatTensor)
                    patch_multi_agg.append(patch)

                feat = torch.cat(patch_multi_agg, dim=0)
                feat = torch.unsqueeze(feat, dim=0)
                batch_input.append(feat)

                if aux_namelist is not None:
                    aux_feat = []
                    for k in aux_namelist:
                        aux_patch = input_band[feat]
                        aux_patch = np.transpose(aux_patch, (1, 2, 0))
                        aux_feat.append(tensor_transform(aux_patch).type(torch.FloatTensor))
                    aux_feat = torch.cat(aux_feat, dim=0)
                    aux_feat = torch.unsqueeze(aux_feat, dim=0)

                    aux_batch_input.append(aux_feat)

        batch_input = torch.cat(batch_input, dim=0)
        if aux_batch_input is not None:
            aux_batch_input = torch.cat(aux_batch_input, dim=0)

        if cuda_used:
            batch_input = batch_input.cuda()
            if aux_batch_input is not None:
                aux_batch_input = aux_batch_input.cuda()
        
        if aux_batch_input is not None:
            with torch.no_grad():
                output_footprint, output_height = est_model(batch_input)
        else:
            with torch.no_grad():
                output_footprint, output_height = est_model(batch_input, aux_batch_input)

        output_footprint = torch.squeeze(output_footprint)
        output_height = torch.squeeze(output_height)

        footprint_pred.append(output_footprint.cpu().numpy())
        height_pred.append(output_height.cpu().numpy())

    height_pred = np.concatenate(height_pred, 0).reshape((ny, nx))
    footprint_pred = np.concatenate(footprint_pred, 0).reshape((ny, nx))

    if log_scale:
        height_pred = np.exp(height_pred)

    if h_min is not None:
        height_pred = np.where(height_pred < h_min, 0.0, height_pred)
    
    if h_max is not None:
        height_pred = np.where(height_pred > h_max, h_max, height_pred)

    if f_min is not None:
        footprint_pred = np.where(footprint_pred < f_min, 0.0, footprint_pred)
    
    if f_max is not None:
        footprint_pred = np.where(footprint_pred > f_max, f_max, footprint_pred)

    # ------save the height into the GeoTiff file
    driver = gdal.GetDriverByName("GTiff")
    footprint_ds = driver.Create(out_footprint_file, nx, ny, 1, gdal.GDT_Float64)
    footprint_ds.GetRasterBand(1).WriteArray(footprint_pred)
    height_ds = driver.Create(out_height_file, nx, ny, 1, gdal.GDT_Float64)
    height_ds.GetRasterBand(1).WriteArray(height_pred)

    aggregate_example = list(tif_input.keys())[0]
    tif_example = tif_input[aggregate_example][s1_key]

    ds = gdal.Open(tif_example)
    geo_trans = ds.GetGeoTransform()
    height_geo_trans = (x_min, resolution, geo_trans[2], y_max, geo_trans[4], -resolution)
    proj = ds.GetProjection()
    footprint_ds.SetGeoTransform(height_geo_trans)
    footprint_ds.SetProjection(proj)
    footprint_ds.FlushCache()
    footprint_ds = None
    height_ds.SetGeoTransform(height_geo_trans)
    height_ds.SetProjection(proj)
    height_ds.FlushCache()
    height_ds = None


if __name__ == "__main__":
    h_min = 2.0
    h_max = 1000.0
    f_min = 0.0
    f_max = 1.0

    # ---mapping settings
    s1_key = "sentinel_1"
    s2_key = "sentinel_2"
    base_dir = "tmp"

    # ---height
    var = "height"
    backbone = "senet"
    model_name = "SEResNet18"

    # ------pretrained models
    if backbone == "cbam":
        model_dir_path = {
            "100m": {
                backbone: os.path.join("DL_run", var, "check_pt_{0}_100m".format(backbone), "experiment_1", "checkpoint.pth.tar"),
                backbone + "_MTL": os.path.join("DL_run", var, "check_pt_{0}_100m_MTL".format(backbone), "experiment_5", "checkpoint.pth.tar")
            },
            "250m": {
                backbone: os.path.join("DL_run", var, "check_pt_{0}_250m".format(backbone), "experiment_2", "checkpoint.pth.tar"),
                backbone + "_MTL": os.path.join("DL_run", var, "check_pt_{0}_250m_MTL".format(backbone), "experiment_5", "checkpoint.pth.tar")
            },
            "500m": {
                backbone: os.path.join("DL_run", var, "check_pt_{0}_500m".format(backbone), "experiment_1", "checkpoint.pth.tar"),
                backbone + "_MTL": os.path.join("DL_run", var, "check_pt_{0}_500m_MTL".format(backbone), "experiment_5", "checkpoint.pth.tar")
            },
            "1000m": {
                backbone: os.path.join("DL_run", var, "check_pt_{0}_1000m".format(backbone), "experiment_1", "checkpoint.pth.tar"),
                backbone + "_MTL": os.path.join("DL_run", var, "check_pt_{0}_1000m_MTL".format(backbone), "experiment_4", "checkpoint.pth.tar")
            }
        }
    elif backbone == "senet":
        model_dir_path = {
            "100m": {
                backbone: os.path.join("DL_run", var, "check_pt_{0}_100m".format(backbone), "experiment_6", "checkpoint.pth.tar"),
                backbone + "_MTL": os.path.join("DL_run", var, "check_pt_{0}_100m_MTL".format(backbone), "experiment_10", "checkpoint.pth.tar")
            },
            "250m": {
                backbone: os.path.join("DL_run", var, "check_pt_{0}_250m".format(backbone), "experiment_6", "checkpoint.pth.tar"),
                backbone + "_MTL": os.path.join("DL_run", var, "check_pt_{0}_250m_MTL".format(backbone), "experiment_10", "checkpoint.pth.tar")
            },
            "500m": {
                backbone: os.path.join("DL_run", var, "check_pt_{0}_500m".format(backbone), "experiment_6", "checkpoint.pth.tar"),
                backbone + "_MTL": os.path.join("DL_run", var, "check_pt_{0}_500m_MTL".format(backbone), "experiment_10", "checkpoint.pth.tar")
            },
            "1000m": {
                backbone: os.path.join("DL_run", var, "check_pt_{0}_1000m".format(backbone), "experiment_7", "checkpoint.pth.tar"),
                backbone + "_MTL": os.path.join("DL_run", var, "check_pt_{0}_1000m_MTL".format(backbone), "experiment_11", "checkpoint.pth.tar")
            }
        }
    else:
        raise NotImplementedError("Unknown models")

    # ------mapping settings for Glasgow
    res_map = {"1000m": ([120], 0.009), "500m": ([60], 0.0045), "250m": ([30], 0.00225), "100m": ([15], 0.0009)}
    
    extent_ref = {
        "100m": [-4.4786000, 55.6759000, -3.8828000, 56.0197000],
        "250m": [-4.4786000, 55.6754500, -3.8823500, 56.0197000],
        "500m": [-4.4786000, 55.6777000, -3.8846000, 56.0197000],
        "1000m": [-4.4786000, 55.6777000, -3.8846000, 56.0197000]
    }

    path_prefix = os.path.join("testCase", "infer_test_Glasgow", "raw_data")
    tif_ref = {"50pt":
        {
            s1_key: os.path.join(path_prefix, "Glasgow_2020_sentinel_1.tif"),
            s2_key: os.path.join(path_prefix, "Glasgow_2020_sentinel_2.tif"),
        }
    }

    # ------execute building height mapping
    for resolution in ["100m", "250m", "500m", "1000m"]:
        scale, resolution_deg = res_map[resolution]
        x_min, y_min, x_max, y_max = extent_ref[resolution]
        for model in [backbone, backbone+"_MTL"]:
            ckpt_file = model_dir_path[resolution][model]
            if "MTL" in ckpt_file:
                out_footprint_file = os.path.join("testCase", "infer_test_Glasgow", resolution,
                                                  "_".join(["Glasgow_footprint", model]) + ".tif")
                out_height_file = os.path.join("testCase", "infer_test_Glasgow", resolution,
                                               "_".join(["Glasgow_height", model]) + ".tif")
                pred_height_from_tiff_DL_patch_MTL(x_min, y_min, x_max, y_max, out_footprint_file, out_height_file, tif_ref,
                                                   patch_size=scale,
                                                   predictor=model_name,
                                                   trained_record=ckpt_file,
                                                   padding=0.01,
                                                   batch_size=256,
                                                   resolution=resolution_deg,
                                                   s1_key=s1_key, s2_key=s2_key, base_dir=base_dir,
                                                   log_scale=False, cuda_used=True,
                                                   h_min=h_min, h_max=h_max,
                                                   f_min=f_min, f_max=f_max)
            else:
                out_height_file = os.path.join("testCase", "infer_test_Glasgow", resolution,
                                               "_".join(["Glasgow", var, model]) + ".tif")
                pred_height_from_tiff_DL_patch(x_min, y_min, x_max, y_max, out_height_file, tif_ref,
                                               patch_size=scale,
                                               predictor=model_name,
                                               trained_record=ckpt_file,
                                               padding=0.01,
                                               batch_size=64,
                                               resolution=resolution_deg,
                                               s1_key=s1_key, s2_key=s2_key, base_dir=base_dir,
                                               log_scale=False, cuda_used=True,
                                               activation="relu",
                                               v_min=h_min, v_max=h_max)
            os.system("rm -rf {0}/*".format(base_dir))
    
    '''
    x_min = -118.4050
    y_min = 33.8790
    x_max = -118.0000
    y_max = 34.1750

    path_prefix = os.path.join("testCase", "infer_test_LosAngeles", "raw_data")

    tif_ref = {"50pt":
        {
            s1_key: os.path.join(path_prefix, "LosAngeles_2018_sentinel_1.tif"),
            s2_key: os.path.join(path_prefix, "LosAngeles_2018_sentinel_2.tif"),
        }
    }

    for resolution in ["100m", "250m", "500m", "1000m"]:
        scale, resolution_deg = res_map[resolution]
        for model in [backbone, backbone+"_MTL"]:
            ckpt_file = model_dir_path[resolution][model]
            if "MTL" in ckpt_file:
                out_footprint_file = os.path.join("testCase", "infer_test_LosAngeles", resolution,
                                                  "_".join(["LosAngeles_footprint", model]) + ".tif")
                out_height_file = os.path.join("testCase", "infer_test_LosAngeles", resolution,
                                               "_".join(["LosAngeles_height", model]) + ".tif")
                pred_height_from_tiff_DL_patch_MTL(x_min, y_min, x_max, y_max, out_footprint_file, out_height_file,
                                                   tif_ref,
                                                   patch_size=scale,
                                                   predictor=model_name,
                                                   trained_record=ckpt_file,
                                                   padding=0.01,
                                                   resolution=resolution_deg,
                                                   s1_key=s1_key, s2_key=s2_key, base_dir=base_dir,
                                                   log_scale=False, cuda_used=True,
                                                   h_min=h_min, h_max=h_max,
                                                   f_min=f_min, f_max=f_max)
            else:
                out_height_file = os.path.join("testCase", "infer_test_LosAngeles", resolution,
                                               "_".join(["LosAngeles", var, model]) + ".tif")
                pred_height_from_tiff_DL_patch(x_min, y_min, x_max, y_max, out_height_file, tif_ref,
                                               patch_size=scale,
                                               predictor=model_name,
                                               trained_record=ckpt_file,
                                               padding=0.01,
                                               resolution=resolution_deg,
                                               s1_key=s1_key, s2_key=s2_key, base_dir=base_dir,
                                               log_scale=False, cuda_used=True,
                                               activation="relu",
                                               v_min=h_min, v_max=h_max)
            os.system("rm -rf {0}/*".format(base_dir))
    '''
    
    # ------mapping settings for Beijing and Chicago
    scale = [120]
    resolution_deg = 0.009
    for city_name in ["Beijing", "Chicago"]:
        if city_name == "Beijing":
            x_min = 116.204
            y_min = 39.823
            x_max = 116.575
            y_max = 40.038
            path_prefix = os.path.join("testCase", "infer_test_Beijing", "raw_data")
            tif_ref = {"50pt":
                {
                    s1_key: os.path.join(path_prefix, "BeijingC6_2020_sentinel_1_50pt.tif"),
                    s2_key: os.path.join(path_prefix, "BeijingC6_2020_sentinel_2_50pt.tif"),
                }
            }
        elif city_name == "Chicago":
            x_min = -87.740
            y_min = 41.733
            x_max = -87.545
            y_max = 41.996
            path_prefix = os.path.join("testCase", "infer_test_Chicago", "raw_data")
            tif_ref = {"50pt":
                {
                    s1_key: os.path.join(path_prefix, "Chicago_2018_sentinel_1_50pt.tif"),
                    s2_key: os.path.join(path_prefix, "Chicago_2018_sentinel_2_50pt.tif"),
                }
            }
        else:
            raise NotImplementedError("Unknown city")

        # ------execute building height mapping    
        for model in [backbone, backbone+"_MTL"]:
            ckpt_file = model_dir_path["1000m"][model]
            if "MTL" in ckpt_file:
                out_footprint_file = os.path.join("testCase", "infer_test_"+city_name, "1000m",
                                                  "_".join([city_name, "footprint", model]) + ".tif")
                out_height_file = os.path.join("testCase", "infer_test_"+city_name, "1000m",
                                               "_".join([city_name, "height", model]) + ".tif")
                pred_height_from_tiff_DL_patch_MTL(x_min, y_min, x_max, y_max, out_footprint_file, out_height_file, tif_ref,
                                                   patch_size=scale,
                                                   predictor=model_name,
                                                   trained_record=ckpt_file,
                                                   padding=0.01,
                                                   batch_size=256,
                                                   resolution=resolution_deg,
                                                   s1_key=s1_key, s2_key=s2_key, base_dir=base_dir,
                                                   log_scale=False, cuda_used=True,
                                                   h_min=h_min, h_max=h_max,
                                                   f_min=f_min, f_max=f_max)
            else:
                out_height_file = os.path.join("testCase", "infer_test_"+city_name, "1000m",
                                               "_".join([city_name, "height", model]) + ".tif")
                pred_height_from_tiff_DL_patch(x_min, y_min, x_max, y_max, out_height_file, tif_ref,
                                               patch_size=scale,
                                               predictor=model_name,
                                               trained_record=ckpt_file,
                                               padding=0.01,
                                               batch_size=64,
                                               resolution=resolution_deg,
                                               s1_key=s1_key, s2_key=s2_key, base_dir=base_dir,
                                               log_scale=False, cuda_used=True,
                                               activation="relu",
                                               v_min=h_min, v_max=h_max)
            os.system("rm -rf {0}/*".format(base_dir))

    # ---footprint
    var = "footprint"
    backbone = "senet"
    model_name = "SEResNet18"

    # ------pretrained models
    if backbone == "cbam":
        model_dir_path = {
            "100m": {
                backbone: os.path.join("DL_run", var, "check_pt_{0}_100m".format(backbone), "experiment_1", "checkpoint.pth.tar"),
                backbone + "_MTL": os.path.join("DL_run", var, "check_pt_{0}_100m_MTL".format(backbone), "experiment_5", "checkpoint.pth.tar")
            },
            "250m": {
                backbone: os.path.join("DL_run", var, "check_pt_{0}_250m".format(backbone), "experiment_0", "checkpoint.pth.tar"),
                backbone + "_MTL": os.path.join("DL_run", "height", "check_pt_{0}_250m_MTL".format(backbone), "experiment_5", "checkpoint.pth.tar")
            },
            "500m": {
                backbone: os.path.join("DL_run", var, "check_pt_{0}_500m".format(backbone), "experiment_1", "checkpoint.pth.tar"),
                backbone + "_MTL": os.path.join("DL_run", var, "check_pt_{0}_500m_MTL".format(backbone), "experiment_5", "checkpoint.pth.tar")
            },
            "1000m": {
                backbone: os.path.join("DL_run", var, "check_pt_{0}_1000m".format(backbone), "experiment_2", "checkpoint.pth.tar"),
                backbone + "_MTL": os.path.join("DL_run", var, "check_pt_{0}_1000m_MTL".format(backbone), "experiment_4", "checkpoint.pth.tar")
            }
        }
    elif backbone == "senet":
        model_dir_path = {
            "100m": {
                backbone: os.path.join("DL_run", var, "check_pt_{0}_100m".format(backbone), "experiment_6", "checkpoint.pth.tar"),
                backbone + "_MTL": os.path.join("DL_run", var, "check_pt_{0}_100m_MTL".format(backbone), "experiment_10", "checkpoint.pth.tar")
            },
            "250m": {
                backbone: os.path.join("DL_run", var, "check_pt_{0}_250m".format(backbone), "experiment_6", "checkpoint.pth.tar"),
                backbone + "_MTL": os.path.join("DL_run", "height", "check_pt_{0}_250m_MTL".format(backbone), "experiment_10", "checkpoint.pth.tar")
            },
            "500m": {
                backbone: os.path.join("DL_run", var, "check_pt_{0}_500m".format(backbone), "experiment_6", "checkpoint.pth.tar"),
                backbone + "_MTL": os.path.join("DL_run", var, "check_pt_{0}_500m_MTL".format(backbone), "experiment_10", "checkpoint.pth.tar")
            },
            "1000m": {
                backbone: os.path.join("DL_run", var, "check_pt_{0}_1000m".format(backbone), "experiment_6", "checkpoint.pth.tar"),
                backbone + "_MTL": os.path.join("DL_run", var, "check_pt_{0}_1000m_MTL".format(backbone), "experiment_10", "checkpoint.pth.tar")
            }
        }
    else:
        raise NotImplementedError("Unknown models")
    
    # ------mapping settings for Glasgow
    extent_ref = {
        "100m": [-4.4786000, 55.6759000, -3.8828000, 56.0197000],
        "250m": [-4.4786000, 55.6754500, -3.8823500, 56.0197000],
        "500m": [-4.4786000, 55.6777000, -3.8846000, 56.0197000],
        "1000m": [-4.4786000, 55.6777000, -3.8846000, 56.0197000]
    }

    s1_key = "sentinel_1"
    s2_key = "sentinel_2"
    base_dir = "tmp"
    path_prefix = os.path.join("testCase", "infer_test_Glasgow", "raw_data")

    tif_ref = {"50pt":
        {
            s1_key: os.path.join(path_prefix, "Glasgow_2020_sentinel_1.tif"),
            s2_key: os.path.join(path_prefix, "Glasgow_2020_sentinel_2.tif"),
        }
    }

    # ------execute building footprint mapping   
    res_map = {"1000m": ([120], 0.009), "500m": ([60], 0.0045), "250m": ([30], 0.00225), "100m": ([15], 0.0009)}
    
    for resolution in ["100m", "250m", "500m", "1000m"]:
        scale, resolution_deg = res_map[resolution]
        x_min, y_min, x_max, y_max = extent_ref[resolution]
        for model in [backbone]:
            ckpt_file = model_dir_path[resolution][model]
            if "MTL" in ckpt_file:
                out_footprint_file = os.path.join("testCase", "infer_test_Glasgow", resolution,
                                                  "_".join(["Glasgow_footprint", model]) + ".tif")
                out_height_file = os.path.join("testCase", "infer_test_Glasgow", resolution,
                                               "_".join(["Glasgow_height", model]) + ".tif")
                pred_height_from_tiff_DL_patch_MTL(x_min, y_min, x_max, y_max, out_footprint_file, out_height_file,
                                                   tif_ref,
                                                   patch_size=scale,
                                                   predictor=model_name,
                                                   trained_record=ckpt_file,
                                                   padding=0.01,
                                                   batch_size=256,
                                                   resolution=resolution_deg,
                                                   s1_key=s1_key, s2_key=s2_key, base_dir=base_dir,
                                                   log_scale=False, cuda_used=True,
                                                   h_min=h_min, h_max=h_max,
                                                   f_min=f_min, f_max=f_max)
            else:
                out_height_file = os.path.join("testCase", "infer_test_Glasgow", resolution,
                                               "_".join(["Glasgow", var, model]) + ".tif")
                pred_height_from_tiff_DL_patch(x_min, y_min, x_max, y_max, out_height_file, tif_ref,
                                               patch_size=scale,
                                               predictor=model_name,
                                               trained_record=ckpt_file,
                                               padding=0.01,
                                               batch_size=64,
                                               resolution=resolution_deg,
                                               s1_key=s1_key, s2_key=s2_key, base_dir=base_dir,
                                               log_scale=False, cuda_used=True,
                                               activation="sigmoid",
                                               v_min=f_min, v_max=f_max)
            os.system("rm -rf {0}/*".format(base_dir))
    
    '''
    x_min = -118.4050
    y_min = 33.8790
    x_max = -118.0000
    y_max = 34.1750

    s1_key = "sentinel_1"
    s2_key = "sentinel_2"
    base_dir = "tmp"
    path_prefix = os.path.join("testCase", "infer_test_LosAngeles", "raw_data")

    tif_ref = {"50pt":
        {
            s1_key: os.path.join(path_prefix, "LosAngeles_2018_sentinel_1.tif"),
            s2_key: os.path.join(path_prefix, "LosAngeles_2018_sentinel_2.tif"),
        }
    }

    res_map = {"1000m": ([120], 0.009), "500m": ([60], 0.0045), "250m": ([30], 0.00225), "100m": ([15], 0.0009)}

    for resolution in ["100m", "250m", "500m", "1000m"]:
        scale, resolution_deg = res_map[resolution]
        for model in [backbone]:
            ckpt_file = model_dir_path[resolution][model]
            if "MTL" in ckpt_file:
                out_footprint_file = os.path.join("testCase", "infer_test_LosAngeles", resolution,
                                                  "_".join(["LosAngeles_footprint", model]) + ".tif")
                out_height_file = os.path.join("testCase", "infer_test_LosAngeles", resolution,
                                               "_".join(["LosAngeles_height", model]) + ".tif")
                pred_height_from_tiff_DL_patch_MTL(x_min, y_min, x_max, y_max, out_footprint_file, out_height_file,
                                                   tif_ref,
                                                   patch_size=scale,
                                                   predictor=model_name,
                                                   trained_record=ckpt_file,
                                                   padding=0.01,
                                                   batch_size=256,
                                                   resolution=resolution_deg,
                                                   s1_key=s1_key, s2_key=s2_key, base_dir=base_dir,
                                                   log_scale=False, cuda_used=True,
                                                   h_min=h_min, h_max=h_max,
                                                   f_min=f_min, f_max=f_max)
            else:
                out_height_file = os.path.join("testCase", "infer_test_LosAngeles", resolution,
                                               "_".join(["LosAngeles", var, model]) + ".tif")
                pred_height_from_tiff_DL_patch(x_min, y_min, x_max, y_max, out_height_file, tif_ref,
                                               patch_size=scale,
                                               predictor=model_name,
                                               trained_record=ckpt_file,
                                               padding=0.01,
                                               batch_size=64,
                                               resolution=resolution_deg,
                                               s1_key=s1_key, s2_key=s2_key, base_dir=base_dir,
                                               log_scale=False, cuda_used=True,
                                               activation="sigmoid",
                                               v_min=f_min, v_max=f_max)
            os.system("rm -rf {0}/*".format(base_dir))
    '''
    
    # ------mapping settings for Beijing and Chicago
    for city_name in ["Beijing", "Chicago"]:
        if city_name == "Beijing":
            x_min = 116.204
            y_min = 39.823
            x_max = 116.575
            y_max = 40.038
            path_prefix = os.path.join("testCase", "infer_test_Beijing", "raw_data")
            tif_ref = {"50pt":
                {
                    s1_key: os.path.join(path_prefix, "BeijingC6_2020_sentinel_1_50pt.tif"),
                    s2_key: os.path.join(path_prefix, "BeijingC6_2020_sentinel_2_50pt.tif"),
                }
            }
        elif city_name == "Chicago":
            x_min = -87.740
            y_min = 41.733
            x_max = -87.545
            y_max = 41.996
            path_prefix = os.path.join("testCase", "infer_test_Chicago", "raw_data")
            tif_ref = {"50pt":
                {
                    s1_key: os.path.join(path_prefix, "Chicago_2018_sentinel_1_50pt.tif"),
                    s2_key: os.path.join(path_prefix, "Chicago_2018_sentinel_2_50pt.tif"),
                }
            }
        else:
            raise NotImplementedError("Unknown city")

        # ------execute building footprint mapping   
        for model in [backbone]:
            ckpt_file = model_dir_path["1000m"][model]
            if "MTL" in ckpt_file:
                out_footprint_file = os.path.join("testCase", "infer_test_"+city_name, "1000m",
                                                  "_".join([city_name, "footprint", model]) + ".tif")
                out_height_file = os.path.join("testCase", "infer_test_"+city_name, "1000m",
                                               "_".join([city_name, "height", model]) + ".tif")
                pred_height_from_tiff_DL_patch_MTL(x_min, y_min, x_max, y_max, out_footprint_file, out_height_file, tif_ref,
                                                   patch_size=scale,
                                                   predictor=model_name,
                                                   trained_record=ckpt_file,
                                                   padding=0.01,
                                                   batch_size=256,
                                                   resolution=resolution_deg,
                                                   s1_key=s1_key, s2_key=s2_key, base_dir=base_dir,
                                                   log_scale=False, cuda_used=True,
                                                   h_min=h_min, h_max=h_max,
                                                   f_min=f_min, f_max=f_max)
            else:
                out_height_file = os.path.join("testCase", "infer_test_"+city_name, "1000m",
                                               "_".join([city_name, var, model]) + ".tif")
                pred_height_from_tiff_DL_patch(x_min, y_min, x_max, y_max, out_height_file, tif_ref,
                                               patch_size=scale,
                                               predictor=model_name,
                                               trained_record=ckpt_file,
                                               padding=0.01,
                                               batch_size=64,
                                               resolution=resolution_deg,
                                               s1_key=s1_key, s2_key=s2_key, base_dir=base_dir,
                                               log_scale=False, cuda_used=True,
                                               activation="sigmoid",
                                               v_min=f_min, v_max=f_max)
            os.system("rm -rf {0}/*".format(base_dir))
