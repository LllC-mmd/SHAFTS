import os
import json
import multiprocessing
import numpy as np
import cv2
from osgeo import gdal
from kneed import KneeLocator


# ************************* GeoTiff File *************************
def setNodata(input_file, output_file, noData, lower=None, higher=None):
    input_ds = gdal.Open(input_file)
    input_band = input_ds.GetRasterBand(1)
    input_array = np.array(input_band.ReadAsArray())

    if input_band.GetNoDataValue() is not None:
        print("use the original NoData value instead")
        noData = input_band.GetNoDataValue()

    if lower is not None:
        input_array = np.where(input_array < lower, noData, input_array)

    if higher is not None:
        input_array = np.where(input_array > higher, noData, input_array)

    driver = gdal.GetDriverByName("GTiff")
    output_ds = driver.Create(output_file, input_array.shape[1], input_array.shape[0], 1, gdal.GDT_Float64)
    output_ds.GetRasterBand(1).WriteArray(input_array)
    output_ds.SetGeoTransform(input_ds.GetGeoTransform())
    output_ds.SetProjection(input_ds.GetProjection())
    output_band = output_ds.GetRasterBand(1)
    output_band.SetNoDataValue(noData)

    output_ds.FlushCache()
    output_band = None
    output_ds = None


def calc_masked_area(pid, x_max_list, x_min_list, y_max_list, y_min_list, xm_max, xm_min, ym_max, ym_min, mask_dta):
    res = []
    for i in range(0, len(x_max_list)):
        pt_x_max_i = x_max_list[i]
        pt_x_min_i = x_min_list[i]
        pt_y_max_i = y_max_list[i]
        pt_y_min_i = y_min_list[i]
        # ------calculate the intersection area of the current input grid with all mask grids
        x_range = np.maximum(0.0, np.minimum(pt_x_max_i, xm_max) - np.maximum(pt_x_min_i, xm_min))
        y_range = np.maximum(0.0, np.minimum(pt_y_max_i, ym_max) - np.maximum(pt_y_min_i, ym_min))
        area_intersect = x_range * y_range
        # ------calculate the area in the current input grid which is masked by mask grids
        masked_area = np.sum(area_intersect * mask_dta)
        res.append(masked_area)
    return {"PID": pid, "result": res}


def setNodataByGUFMask(input_file, mask_file, output_file, masked_value, masked_threshold=0.2, dtype=gdal.GDT_Float64, noData=None, num_cpu=1):
    # ------read input file
    input_ds = gdal.Open(input_file)
    input_geoTransform = input_ds.GetGeoTransform()
    x_min = input_geoTransform[0]
    dx = input_geoTransform[1]
    y_max = input_geoTransform[3]
    dy = input_geoTransform[5]

    input_band = input_ds.GetRasterBand(1)
    input_dta = np.array(input_band.ReadAsArray())
    ny, nx = input_dta.shape

    if noData is None:
        noData = input_band.GetNoDataValue()

    # ------create arrays used for storing x-, y-coordinates of 4 corner point of input grids
    pt_x_min = np.array([x_min + dx * i for i in range(0, nx)])
    pt_x_max = pt_x_min + dx
    pt_y_max = np.array([y_max + dy * i for i in range(0, ny)])
    pt_y_min = pt_y_max + dy

    # ------read mask file
    mask_ds = gdal.Open(mask_file)
    mask_geoTransform = mask_ds.GetGeoTransform()
    xm_min = mask_geoTransform[0]
    dx_m = mask_geoTransform[1]
    ym_max = mask_geoTransform[3]
    dy_m = mask_geoTransform[5]

    mask_band = mask_ds.GetRasterBand(1)
    mask_dta = np.array(mask_band.ReadAsArray())
    ny_m, nx_m = mask_dta.shape
    mask_dta_trans = np.where(mask_dta == masked_value, 1.0, 0.0)

    # ------create arrays used for storing x-, y-coordinates of 4 corner point of mask grids
    ptm_x_min = np.array([xm_min + dx_m * i for i in range(0, nx_m)])
    ptm_x_min = np.concatenate([ptm_x_min.reshape((1, -1)) for i in range(0, ny_m)], axis=0)
    ptm_x_max = ptm_x_min + dx_m
    ptm_y_max = np.array([ym_max + dy_m * i for i in range(0, ny_m)])
    ptm_y_max = np.concatenate([ptm_y_max.reshape((-1, 1)) for i in range(0, nx_m)], axis=1)
    ptm_y_min = ptm_y_max + dy_m

    '''
    for i in range(0, ny):
        pt_y_max_i = pt_y_max[i]
        pt_y_min_i = pt_y_min[i]
        for j in range(0, nx):
            pt_x_max_i = pt_x_max[j]
            pt_x_min_i = pt_x_min[j]
            # ---------calculate the intersection area of the current input grid with all mask grids
            x_range = np.maximum(0.0, np.minimum(pt_x_max_i, ptm_x_max) - np.maximum(pt_x_min_i, ptm_x_min))
            y_range = np.maximum(0.0, np.minimum(pt_y_max_i, ptm_y_max) - np.maximum(pt_y_min_i, ptm_y_min))
            area_intersect = x_range * y_range
            # ---------calculate the percentage of area in the current input grid which is masked by mask grids
            masked_ratio[i][j] = np.sum(area_intersect * mask_dta_trans) / np.abs(dx * dy)
    '''

    pt_x_max_summary = []
    pt_x_min_summary = []
    pt_y_max_summary = []
    pt_y_min_summary = []

    for i in range(0, ny):
        for j in range(0, nx):
            pt_y_max_summary.append(pt_y_max[i])
            pt_y_min_summary.append(pt_y_min[i])
            pt_x_max_summary.append(pt_x_max[j])
            pt_x_min_summary.append(pt_x_min[j])

    pt_x_max_summary = np.array(pt_x_max_summary)
    pt_x_min_summary = np.array(pt_x_min_summary)
    pt_y_max_summary = np.array(pt_y_max_summary)
    pt_y_min_summary = np.array(pt_y_min_summary)

    id_summary = np.arange(ny * nx)
    id_sub = np.array_split(id_summary, num_cpu)
    arg_list = [(i, pt_x_max_summary[id_sub[i]], pt_x_min_summary[id_sub[i]],
                 pt_y_max_summary[id_sub[i]], pt_y_min_summary[id_sub[i]],
                 ptm_x_max, ptm_x_min, ptm_y_max, ptm_y_min, mask_dta_trans) for i in range(0, num_cpu)]

    pool = multiprocessing.Pool(processes=num_cpu)
    res_list = pool.starmap(calc_masked_area, arg_list)
    pool.close()
    pool.join()

    masked_ratio = []
    res_list = sorted(res_list, key=lambda x: x["PID"], reverse=False)
    for res in res_list:
        masked_ratio = masked_ratio + res["result"]

    masked_ratio = np.reshape(masked_ratio, (ny, nx)) / np.abs(dx * dy)

    input_dta = np.where(masked_ratio >= masked_threshold, noData, input_dta)

    driver = gdal.GetDriverByName("GTiff")
    output_ds = driver.Create(output_file, input_dta.shape[1], input_dta.shape[0], 1, dtype)
    output_ds.GetRasterBand(1).WriteArray(input_dta)
    output_ds.SetGeoTransform(input_ds.GetGeoTransform())
    output_ds.SetProjection(input_ds.GetProjection())
    output_band = output_ds.GetRasterBand(1)
    output_band.SetNoDataValue(noData)

    output_ds.FlushCache()
    output_band = None
    output_ds = None


def calc_maxConnectedComponent(height_dta, footprint_dta, h_threshold, d_threshold):
    img_gray = np.full_like(height_dta, 255)
    # -------if height >= h_threshold and footprint >= d_threshold, it would be a urban pixel
    non_urban_mask = ~np.logical_and(height_dta >= h_threshold, footprint_dta >= d_threshold)
    img_gray[non_urban_mask] = 0
    # -------see: https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connected-components-with-stats-in-python
    # -------for the meaning of each result
    num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(img_gray.astype('uint8'),
                                                                          connectivity=8,
                                                                          ltype=cv2.CV_32S)
    # -------exclude background pixels
    if len(stats) > 1:
        cc_maxsize = np.max(stats[1:, 4]) / np.sum(stats[1:, 4])
    else:
        cc_maxsize = 1.0
    return cc_maxsize


def calc_maxConnectedComponent_single(height_dta, footprint_dta, d_threshold, h_threshold=3.0):
    # -------white: 255, black: 0
    img_gray = np.full_like(footprint_dta, 0)
    # -------if footprint >= d_threshold, it would be a urban pixel
    urban_mask = np.logical_and(footprint_dta >= d_threshold, height_dta >= h_threshold)
    img_gray[urban_mask] = 255
    # -------see: https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connected-components-with-stats-in-python
    # -------for the meaning of each result
    num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(img_gray.astype('uint8'),
                                                                          connectivity=8,
                                                                          ltype=cv2.CV_32S)
    # -------exclude background pixels
    if len(stats) > 1:
        cc_maxsize = np.max(stats[1:, 4]) / np.sum(stats[1:, 4])
    else:
        cc_maxsize = 1.0
    return cc_maxsize


def calc_ConnectedComponentEntropy_single(height_dta, footprint_dta, d_threshold, h_threshold=2.0):
    # -------white: 255, black: 0
    img_gray = np.full_like(footprint_dta, 0)
    # -------if footprint >= d_threshold, it would be a urban pixel
    urban_mask = np.logical_and(footprint_dta >= d_threshold, height_dta >= h_threshold)
    img_gray[urban_mask] = 255
    # -------see: https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connected-components-with-stats-in-python
    # -------for the meaning of each result
    num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(img_gray.astype('uint8'),
                                                                          connectivity=8,
                                                                          ltype=cv2.CV_32S)
    # -------exclude background pixels
    if len(stats) > 1:
        cluster_area = stats[1:, 4]
        area_sum = np.sum(cluster_area)
        cluster_prob = cluster_area / area_sum
        entropy = np.sum(cluster_prob * np.log(cluster_prob))
    else:
        entropy = 0.0
    return entropy


def calc_percolation_thresold_subproc(pid, id_test, height_dta, footprint_dta, h_test, d_test):
    maxsize_list = []
    for ids in id_test:
        h_id, d_id = ids
        h_threshold = h_test[h_id]
        d_threshold = d_test[d_id]
        cc_maxsize = calc_maxConnectedComponent(height_dta, footprint_dta, h_threshold, d_threshold)
        maxsize_list.append(cc_maxsize)

    return {"PID": pid, "result": maxsize_list}


def calc_percolation_thresold_subproc_single(pid, id_test, height_dta, footprint_dta, d_test, criteria="entropy", h_min=3.0):
    res_list = []
    if criteria == "entropy":
        for d_id in id_test:
            d_threshold = d_test[d_id]
            cc_entropy = calc_ConnectedComponentEntropy_single(height_dta, footprint_dta, d_threshold, h_min)
            res_list.append(cc_entropy)
    elif criteria == "maxSize":
        for d_id in id_test:
            d_threshold = d_test[d_id]
            cc_maxsize = calc_maxConnectedComponent_single(height_dta, footprint_dta, d_threshold, h_min)
            res_list.append(cc_maxsize)

    return {"PID": pid, "result": res_list}


def calc_percolation_thresold(height_file, footprint_file, num_cpu=1, save_path=None):
    h_test = np.linspace(2.0, 20.0, 901)
    num_h_test = len(h_test)
    d_test = np.linspace(0.01, 0.9, 891)
    num_d_test = len(d_test)

    height_ds = gdal.Open(height_file)
    height_band = height_ds.GetRasterBand(1)
    h_noData = height_band.GetNoDataValue()
    height_dta = np.array(height_band.ReadAsArray())
    height_dta = np.where(height_dta == h_noData, 0.0, height_dta)

    footprint_ds = gdal.Open(footprint_file)
    footprint_band = footprint_ds.GetRasterBand(1)
    d_noData = footprint_band.GetNoDataValue()
    footprint_dta = np.array(footprint_band.ReadAsArray())
    footprint_dta = np.where(footprint_dta == d_noData, 0.0, footprint_dta)

    # cc_maxsize = calc_maxConnectedComponent(height_dta, footprint_dta, h_threshold=10.0, d_threshold=0.1)

    loc_id = np.array([(i, j) for i in range(0, num_h_test) for j in range(0, num_d_test)])
    id_summary = np.arange(num_h_test * num_d_test)
    id_sub = np.array_split(id_summary, num_cpu)
    arg_list = [(i, loc_id[id_sub[i]], height_dta, footprint_dta, h_test, d_test) for i in range(0, num_cpu)]

    pool = multiprocessing.Pool(processes=num_cpu)
    res_list = pool.starmap(calc_percolation_thresold_subproc, arg_list)
    pool.close()
    pool.join()

    cc_size = []
    res_list = sorted(res_list, key=lambda x: x["PID"], reverse=False)
    for res in res_list:
        cc_size = cc_size + res["result"]

    cc_size = np.reshape(cc_size, (num_h_test, num_d_test))
    # print(cc_size)

    if save_path is not None:
        np.save(save_path, cc_size)


def calc_percolation_thresold_single(height_file: str, footprint_file: str, density_test: np.array, h_min=2.0, criteria="entropy", num_cpu=1, save_path=None):
    """Calculate an optimal building footprint value differentiating between urban and non-urban area based on percolation theory.

    Parameters
    ----------

    height_file : str
        Path to the input building height file.
    footprint_file : str
        Path to the input building footprint file.
    density_test : np.array
        A pre-defined set of potential building footprint threshold.
    h_min : float
        A pre-defined minimum building height value differentiating between urban and non-urban area.
        The default is `2.0`.
    criteria : str
        A criterion which describes the property of urban cluster system.
        It can be chosen from: 
            `entropy` which stands for Shannon's entropy of urban size distribution.
            `maxSize` which stands for the area ratio between the largest urban and the summation of all urban area.
        The default is `entropy`.
    num_cpu : int
        Number of expected processes to be run in parallel for acceleration.
        The default is `1`.
    save_path : str
        Output Path for saving the matrix of urban cluster properties under different threshold.
    """

    num_d_test = len(density_test)

    height_ds = gdal.Open(height_file)
    height_band = height_ds.GetRasterBand(1)
    h_noData = height_band.GetNoDataValue()
    height_dta = np.array(height_band.ReadAsArray())
    height_dta = np.where(height_dta == h_noData, 0.0, height_dta)

    footprint_ds = gdal.Open(footprint_file)
    footprint_band = footprint_ds.GetRasterBand(1)
    d_noData = footprint_band.GetNoDataValue()
    footprint_dta = np.array(footprint_band.ReadAsArray())
    footprint_dta = np.where(footprint_dta == d_noData, 0.0, footprint_dta)

    loc_id = np.array([i for i in range(0, num_d_test)])
    id_summary = np.arange(num_d_test)
    id_sub = np.array_split(id_summary, num_cpu)
    arg_list = [(i, loc_id[id_sub[i]], height_dta, footprint_dta, density_test, criteria, h_min) for i in range(0, num_cpu)]

    pool = multiprocessing.Pool(processes=num_cpu)
    res_list = pool.starmap(calc_percolation_thresold_subproc_single, arg_list)
    pool.close()
    pool.join()

    cc_size = []
    res_list = sorted(res_list, key=lambda x: x["PID"], reverse=False)
    for res in res_list:
        cc_size = cc_size + res["result"]

    cc_size = np.array(cc_size)
    # print(cc_size)

    if save_path is not None:
        np.save(save_path, cc_size)


def get_binary_CCMat(cc_dta):
    dta_trans = (cc_dta - np.min(cc_dta)) / (np.max(cc_dta) - np.min(cc_dta)) * 255
    dta_trans = dta_trans.astype(np.uint8)

    ret, cc_binary = cv2.threshold(dta_trans, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return cc_binary


def get_binary_CCMat_single(cc_dta, d_test, d_min=0.05):
    '''
    cc_dta_delta = cc_dta[1:] - cc_dta[:-1]
    change_id = int(np.argmin(cc_dta_delta) + 1)

    cc_binary = np.ones_like(cc_dta)
    cc_binary[:change_id] = 0
    '''
    loc_convex_decreasing = KneeLocator(d_test, cc_dta, curve='convex', direction='decreasing', online=True, S=1.0)
    change_d = sorted([round(d, 3) for d in loc_convex_decreasing.all_knees if d >= d_min])
    cc_binary = np.ones_like(d_test)
    cc_binary[d_test < change_d[0]] = 0

    return cc_binary


def calc_curve_fromCCMatFusion(cc_file, height_test, density_test, save_path=None):
    num_h_test = len(height_test)
    num_d_test = len(density_test)

    cc_final = np.ones((num_h_test, num_d_test))

    for cc_f in cc_file:
        cc_dta = np.load(cc_f)
        cc_binary = get_binary_CCMat(cc_dta)
        cc_final = np.logical_and(cc_binary, cc_final)

    if save_path is not None:
        np.save(save_path, cc_final)

    return cc_final


# ---CCMat after fusion is called "binarized"
def calc_thresold_fromCCMat(cc_file, height_test, density_test, save_path, binarized=False):
    thresold_dict = {}

    cc_dta = np.load(cc_file)

    if binarized:
        cc_binary = cc_dta.astype(int)
    else:
        cc_dta = (cc_dta - np.min(cc_dta)) / (np.max(cc_dta) - np.min(cc_dta)) * 255
        cc_dta = cc_dta.astype(np.uint8)
        ret, cc_binary = cv2.threshold(cc_dta, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num_h_test = len(height_test)
    for i in range(0, num_h_test):
        min_loc = np.argmin(cc_binary[i])
        if min_loc != 0:
            thresold_dict[i] = [height_test[i], density_test[min_loc]]

    with open(save_path, "w") as fp:
        json.dump(thresold_dict, fp)


def calc_mask_fromCCMat(source_ref, ny, nx, height_test, density_test, save_path, ratio=0.8):
    num_h_test = len(height_test)

    urban_acc = None

    num_source = 0
    for s in source_ref.keys():
        cc_file = source_ref[s]["CCMat"]
        cc_dta = np.load(cc_file)
        cc_binary = get_binary_CCMat(cc_dta)

        height_file = source_ref[s]["height"]
        height_ds = gdal.Open(height_file)
        height_band = height_ds.GetRasterBand(1)
        h_noData = height_band.GetNoDataValue()
        height_dta = np.array(height_band.ReadAsArray())[0:ny, 0:nx]
        height_dta = np.where(height_dta == h_noData, 0.0, height_dta)

        if urban_acc is None:
            urban_acc = np.zeros((ny, nx))

        footprint_file = source_ref[s]["footprint"]
        footprint_ds = gdal.Open(footprint_file)
        footprint_band = footprint_ds.GetRasterBand(1)
        d_noData = footprint_band.GetNoDataValue()
        footprint_dta = np.array(footprint_band.ReadAsArray())[0:ny, 0:nx]
        footprint_dta = np.where(footprint_dta == d_noData, 0.0, footprint_dta)

        threshold_dict = {}
        for i in range(0, num_h_test):
            min_loc = np.argmin(cc_binary[i])
            if min_loc != 0:
                threshold_dict[i] = [height_test[i], density_test[min_loc]]

        threshold_sorted = sorted([v for v in threshold_dict.values()], key=lambda x: x[0])

        h_threshold = [0.0] + sorted([v[0] for v in threshold_sorted])
        threshold_sorted = [[0.0, 1.0]] + threshold_sorted
        h_dta_max = np.max(height_dta)
        if h_dta_max > h_threshold[-1]:
            h_threshold = h_threshold + [h_dta_max]
            threshold_sorted = threshold_sorted + [[h_dta_max, min([v[1] for v in threshold_dict.values()])]]

        nonurban_area_tmp = np.zeros((ny, nx))
        height_dta_1d = height_dta.flatten()
        height_dta_cat = np.reshape(np.digitize(height_dta_1d, bins=h_threshold), (ny, nx))
        for h_id in range(1, len(h_threshold) + 1):
            test_mask = np.logical_and(height_dta_cat == h_id, footprint_dta < threshold_sorted[h_id - 1][1])
            nonurban_area_tmp = np.logical_or(test_mask, nonurban_area_tmp)

        num_source += 1
        urban_acc += (~nonurban_area_tmp).astype(int)

    urban_acc = urban_acc / num_source
    mask_final = (urban_acc >= ratio)

    driver = gdal.GetDriverByName("GTiff")
    output_ds = driver.Create(save_path, nx, ny, 1, gdal.GDT_Int32)
    output_ds.GetRasterBand(1).WriteArray(mask_final.astype(int))
    output_ds.SetGeoTransform(footprint_ds.GetGeoTransform())
    output_ds.SetProjection(footprint_ds.GetProjection())
    output_band = output_ds.GetRasterBand(1)
    output_band.SetNoDataValue(0)

    output_ds.FlushCache()
    output_band = None
    output_ds = None

    return mask_final


def calc_mask_fromCCMat_single(source_ref: dict, density_test: np.array, save_path: str, ny=None, nx=None, h_min=2.0, ratio=0.8):
    """Calculate non-urban mask based on building footprint prediction resulted from different methods.

    Parameters
    ----------

    source_ref : dict
        A dictionary which specifies the path of input building height, building footprint and urban cluster system properties.
        An example can be given as follows:
            {"method-1":
                   {
                       `CCMat`:  `path of .npy file which contains urban cluster properties under different threshold`,
                       `height`:  `path of .tif file which is the building height prediction result`,
                       `footprint`: `path of .tif file which is the building footprint prediction result`,
                    },
             ......
           }
    density_test : np.array
        A pre-defined set of potential building footprint threshold.
    save_path : str
        Output Path for saving the non-urban mask.
    ny : int
        Number of output grids to be kept along the latitude direction.
        The default is `None`.
    nx : int
        Number of output grids to be kept along the longitude direction.
        The default is `None`.
    h_min : float
        A pre-defined minimum building height value differentiating between urban and non-urban area.
        The default is `2.0`.
    ratio : float
        A minimum required percentage to identify a pixel as urban pixel by the voting of different methods.
        The default is `0.8`.

    """
    urban_acc = None

    num_source = 0
    for s in source_ref.keys():
        cc_file = source_ref[s]["CCMat"]
        cc_dta = np.load(cc_file)
        cc_binary = get_binary_CCMat_single(cc_dta, d_test=density_test)
        change_id = np.argmax(cc_binary)
        d_threshold = density_test[change_id]

        footprint_file = source_ref[s]["footprint"]
        footprint_ds = gdal.Open(footprint_file)
        footprint_band = footprint_ds.GetRasterBand(1)
        d_noData = footprint_band.GetNoDataValue()

        footprint_dta = np.array(footprint_band.ReadAsArray())
        if ny is None:
            ny = footprint_dta.shape[0]
        if nx is None:
            nx = footprint_dta.shape[1]
        footprint_dta = footprint_dta[0:ny, 0:nx]
        footprint_dta = np.where(footprint_dta == d_noData, 0.0, footprint_dta)

        height_file = source_ref[s]["height"]
        height_ds = gdal.Open(height_file)
        height_band = height_ds.GetRasterBand(1)
        h_noData = height_band.GetNoDataValue()
        height_dta = np.array(height_band.ReadAsArray())[0:ny, 0:nx]
        height_dta = np.where(height_dta == h_noData, 0.0, height_dta)

        if urban_acc is None:
            urban_acc = np.zeros((ny, nx))

        urban_mask_tmp = np.logical_and(footprint_dta >= d_threshold, height_dta >= h_min)
        urban_mask_tmp = urban_mask_tmp.astype(int)

        num_source += 1
        urban_acc += urban_mask_tmp

    urban_acc = urban_acc / num_source
    mask_final = (urban_acc >= ratio)

    driver = gdal.GetDriverByName("GTiff")
    output_ds = driver.Create(save_path, nx, ny, 1, gdal.GDT_Int32)
    output_ds.GetRasterBand(1).WriteArray(mask_final.astype(int))
    output_ds.SetGeoTransform(footprint_ds.GetGeoTransform())
    output_ds.SetProjection(footprint_ds.GetProjection())
    output_band = output_ds.GetRasterBand(1)
    output_band.SetNoDataValue(0)

    output_ds.FlushCache()
    output_band = None
    output_ds = None

    return mask_final


def setNodataByMask(input_file, mask_file, output_file, ny=None, nx=None, dtype=gdal.GDT_Float64, noData=None):
    """Mask 3D building information prediction results.

    Parameters
    ----------

    input_file : str
        Path to the input file to be masked.
    mask_file : str
        Path to the masking file.
    output_file : str
        Output Path for the masked result.
    ny : int
        Number of output grids to be kept along the latitude direction.
        The default is `None`.
    nx : int
        Number of output grids to be kept along the longitude direction.
        The default is `None`.
    dtype : float
        Data type for the output file.
        The default is `gdal.GDT_Float64`.
    noData : float
        NoData value for the output file.
        The default is `None`.

    """
    mask_ds = gdal.Open(mask_file)
    mask_band = mask_ds.GetRasterBand(1)
    mask_noData = mask_band.GetNoDataValue()
    mask_dta = np.array(mask_band.ReadAsArray())
    if ny is None:
        ny = mask_dta.shape[0]
    if nx is None:
        nx = mask_dta.shape[1]
    mask_dta = mask_dta[0:ny, 0:nx]

    input_ds = gdal.Open(input_file)
    input_band = input_ds.GetRasterBand(1)
    input_noData = input_band.GetNoDataValue()
    input_dta = np.array(input_band.ReadAsArray())[0:ny, 0:nx]

    if noData is None:
        noData = input_noData

    mask_dta = np.where(mask_dta == mask_noData, 0.0, mask_band)
    mask_dta = np.where(mask_dta != 0.0, 1.0, 0.0)
    input_dta = input_dta * mask_dta
    input_dta = np.where(input_dta == 0.0, noData, input_dta)

    driver = gdal.GetDriverByName("GTiff")
    output_ds = driver.Create(output_file, nx, ny, 1, dtype)
    output_ds.GetRasterBand(1).WriteArray(input_dta)
    output_ds.SetGeoTransform(input_ds.GetGeoTransform())
    output_ds.SetProjection(input_ds.GetProjection())
    output_band = output_ds.GetRasterBand(1)
    output_band.SetNoDataValue(noData)

    output_ds.FlushCache()
    output_band = None
    output_ds = None


def setNodataByThreshold_fusion(res_dict, output_suffix, ny, nx, h_min=3.0, d_min=0.1, dtype=gdal.GDT_Float64):
    tmp_dict = {}
    for model in ["ref", "senet", "senet_MTL"]:
        height_file = res_dict[model]["height"]
        height_ds = gdal.Open(height_file)
        height_band = height_ds.GetRasterBand(1)
        h_noData = height_band.GetNoDataValue()
        height_dta = np.array(height_band.ReadAsArray())[0:ny, 0:nx]

        footprint_file = res_dict[model]["footprint"]
        footprint_ds = gdal.Open(footprint_file)
        footprint_band = footprint_ds.GetRasterBand(1)
        d_noData = footprint_band.GetNoDataValue()
        footprint_dta = np.array(footprint_band.ReadAsArray())[0:ny, 0:nx]

        if h_noData is None:
            h_noData = -100000

        if d_noData is None:
            d_noData = -255

        footprint_dta = np.where(footprint_dta == d_noData, 0.0, footprint_dta)
        height_dta = np.where(height_dta == h_noData, 0.0, height_dta)

        if model != "ref":
            threshold_json = None
            with open(res_dict[model]["threshold"], "r") as fp:
                threshold_json = json.load(fp)
            threshold_sorted = sorted([v for v in threshold_json.values()], key=lambda x: x[0])

            h_threshold = [0.0] + sorted([v[0] for v in threshold_sorted])
            threshold_sorted = [[0.0, 1.0]] + threshold_sorted
            h_dta_max = np.max(height_dta)
            if h_dta_max > h_threshold[-1]:
                h_threshold = h_threshold + [h_dta_max]
                threshold_sorted = threshold_sorted + [[h_dta_max, min([v[1] for v in threshold_json.values()])]]

            height_dta_1d = height_dta.flatten()
            height_dta_cat = np.reshape(np.digitize(height_dta_1d, bins=h_threshold), (ny, nx))
            for h_id in range(1, len(h_threshold) + 1):
                test_mask = np.logical_and(height_dta_cat == h_id, footprint_dta < threshold_sorted[h_id-1][1])
                height_dta[test_mask] = h_noData
                footprint_dta[test_mask] = d_noData
        else:
            test_mask = np.logical_and(height_dta > h_min, footprint_dta > d_min)
            height_dta[~test_mask] = h_noData
            footprint_dta[~test_mask] = d_noData

        tmp_dict[model] = {"height": height_dta, "footprint": footprint_dta}

    mask_final = np.zeros_like(tmp_dict["ref"]["height"])
    for k in ["ref", "senet", "senet_MTL"]:
        non_urban_mask = (tmp_dict[k]["height"] == h_noData)
        mask_final = np.logical_or(non_urban_mask, mask_final)

    for k in tmp_dict.keys():
        tmp_dict[k]["height"][mask_final] = h_noData
        tmp_dict[k]["footprint"][mask_final] = d_noData

    for k in tmp_dict.keys():
        height_file = res_dict[k]["height"]
        height_ds = gdal.Open(height_file)

        f_dir = os.path.dirname(height_file)
        f_base = os.path.splitext(os.path.basename(height_file))[0]
        output_height_file = os.path.join(f_dir, f_base + output_suffix + ".tif")

        height_dta = tmp_dict[k]["height"]
        driver = gdal.GetDriverByName("GTiff")
        output_ds = driver.Create(output_height_file, nx, ny, 1, dtype)
        output_ds.GetRasterBand(1).WriteArray(height_dta)
        output_ds.SetGeoTransform(height_ds.GetGeoTransform())
        output_ds.SetProjection(height_ds.GetProjection())
        output_band = output_ds.GetRasterBand(1)
        output_band.SetNoDataValue(int(h_noData))

        output_ds.FlushCache()
        output_band = None
        output_ds = None

        footprint_file = res_dict[k]["footprint"]
        f_dir = os.path.dirname(footprint_file)
        f_base = os.path.splitext(os.path.basename(footprint_file))[0]
        output_footprint_file = os.path.join(f_dir, f_base + output_suffix + ".tif")
        footprint_ds = gdal.Open(footprint_file)

        footprint_dta = tmp_dict[k]["footprint"]
        output_ds = driver.Create(output_footprint_file, nx, ny, 1, dtype)
        output_ds.GetRasterBand(1).WriteArray(footprint_dta)
        output_ds.SetGeoTransform(footprint_ds.GetGeoTransform())
        output_ds.SetProjection(footprint_ds.GetProjection())
        output_band = output_ds.GetRasterBand(1)
        output_band.SetNoDataValue(int(d_noData))

        output_ds.FlushCache()
        output_band = None
        output_ds = None

    
if __name__ == "__main__":
    '''
    # ------set data mask using Global Urban Footprint (GUF)
    infer_prefix = "../testCase"

    resolution_mapping = {"Glasgow": ["1000m", "100m", "250m", "500m"],
                          "Beijing": ["1000m"],
                          "Chicago": ["1000m"]}
    nodata_mapping = {"height": -100000, "footprint": -255}

    for city in ["Glasgow", "Beijing", "Chicago"]:
        guf_file = os.path.join(infer_prefix, "infer_test_{0}".format(city), "{0}_GUF.tif".format(city))
        for resolution in resolution_mapping[city]:
            for var in ["height", "footprint"]:
                for model in ["senet", "senet_MTL"]:
                    res_prefix = os.path.join(infer_prefix, "infer_test_{0}".format(city), resolution)
                    res_file = os.path.join(res_prefix, "{0}_{1}_{2}.tif".format(city, var, model))
                    output_file = os.path.join(res_prefix, "{0}_{1}_{2}_masked.tif".format(city, var, model))

                    setNodataByGUFMask(input_file=res_file, mask_file=guf_file, output_file=output_file,
                                    noData=nodata_mapping[var], masked_value=0, masked_threshold=0.8, num_cpu=25)
    '''

    # ------Glasgow close-ups
    # ---------calculate the minimum foortprint for urban area identification based on percolation theory
    h_test = np.linspace(2.0, 20.0, 901)
    d_test = np.linspace(0.01, 0.9, 891)
    
    infer_prefix = "../testCase"
    for resolution in ["1000m", "100m", "250m", "500m"]:
        res_prefix = os.path.join(infer_prefix, "infer_test_Glasgow", resolution)
        height_ref_file = os.path.join(res_prefix, "Glasgow_2020_building_height_clip.tif")
        footprint_ref_file = os.path.join(res_prefix, "Glasgow_2020_building_footprint_clip.tif")
        calc_percolation_thresold_single(height_file=height_ref_file, footprint_file=footprint_ref_file, density_test=d_test, h_min=5.0,
                                         num_cpu=10, save_path="cc_ref_{0}.npy".format(resolution))
        for model in ["senet", "senet_MTL"]:
        #for model in ["cbam", "cbam_MTL"]:
            height_file = os.path.join(res_prefix, "Glasgow_height_{0}.tif".format(model))
            footprint_file = os.path.join(res_prefix, "Glasgow_footprint_{0}.tif".format(model))
            calc_percolation_thresold_single(height_file=height_file, footprint_file=footprint_file, density_test=d_test, h_min=5.0,
                                             num_cpu=10, save_path="cc_{0}_{1}.npy".format(model, resolution))

    infer_prefix = "../testCase"

    num_mapping = {"1000m": [38, 66], "100m": [382, 662], "250m": [153, 265], "500m": [76, 132]}

    for resolution in ["1000m", "100m", "250m", "500m"]:
        ny, nx = num_mapping[resolution]
        res_prefix = os.path.join(infer_prefix, "infer_test_Glasgow", resolution)

        source_ref = {}
        res_type = "ref"
        source_ref["ref"] = {}
        source_ref["ref"]["CCMat"] = "cc_ref_{0}.npy".format(resolution)
        source_ref["ref"]["height"] = os.path.join(res_prefix, "Glasgow_2020_building_height_clip.tif")
        source_ref["ref"]["footprint"] = os.path.join(res_prefix, "Glasgow_2020_building_footprint_clip.tif")
        calc_mask_fromCCMat_single(source_ref=source_ref, density_test=d_test, ny=ny, nx=nx,
                                   save_path=os.path.join(res_prefix, "Glasgow_mask_{0}.tif".format(res_type)),
                                   h_min=5.0, ratio=1.0)
        print(os.path.join(res_prefix, "Glasgow_mask_{0}.tif".format(res_type)))

        source_ref = {}
        for res_type in ["senet", "senet_MTL"]:
            source_ref[res_type] = {}
            source_ref[res_type]["CCMat"] = "cc_{0}_{1}.npy".format(res_type, resolution)
            source_ref[res_type]["height"] = os.path.join(res_prefix, "Glasgow_height_{0}.tif".format(res_type))
            source_ref[res_type]["footprint"] = os.path.join(res_prefix, "Glasgow_footprint_{0}.tif".format(res_type))

        for res_type in ["senet", "senet_MTL"]:
            calc_mask_fromCCMat_single(source_ref=source_ref, density_test=d_test, ny=ny, nx=nx,
                                       save_path=os.path.join(res_prefix, "Glasgow_mask_{0}.tif".format(res_type)),
                                       h_min=5.0, ratio=0.5)
            print(os.path.join(res_prefix, "Glasgow_mask_{0}.tif".format(res_type)))

    nodata_mapping = {"height": -100000, "footprint": -255}

    for resolution in ["1000m", "100m", "250m", "500m"]:
        ny, nx = num_mapping[resolution]
        res_prefix = os.path.join(infer_prefix, "infer_test_Glasgow", resolution)

        for res_type in ["ref", "senet", "senet_MTL"]:
        #for res_type in ["ref", "cbam", "cbam_MTL"]:
            th_mask = os.path.join(res_prefix, "Glasgow_mask_{0}.tif".format(res_type))

            for var in ["height", "footprint"]:
                if res_type == "ref":
                    res_file = os.path.join(res_prefix, "Glasgow_2020_building_{0}_clip.tif".format(var))
                    output_file = os.path.join(res_prefix, "Glasgow_2020_building_{0}_clip_th.tif".format(var))
                    setNodataByMask(input_file=res_file, mask_file=th_mask, output_file=output_file, ny=ny, nx=nx,
                                    noData=nodata_mapping[var])
                else:
                    res_file = os.path.join(res_prefix, "Glasgow_{0}_{1}.tif".format(var, res_type))
                    output_file = os.path.join(res_prefix, "Glasgow_{0}_{1}_th.tif".format(var, res_type))

                    setNodataByMask(input_file=res_file, mask_file=th_mask, output_file=output_file, ny=ny, nx=nx,
                                    noData=nodata_mapping[var])
                    print(output_file)

    '''
    # ------Li comparison
    # ------calculate the minimum foortprint for urban area identification based on percolation theory
    infer_prefix = "../testCase"
    res_prefix = os.path.join(infer_prefix, "infer_test_Glasgow", "1000m", "Li_comparison")
    height_ref_file = os.path.join(res_prefix, "Glasgow_2020_building_height_LiClip.tif")
    footprint_ref_file = os.path.join(res_prefix, "Glasgow_2020_building_footprint_LiClip.tif")
    calc_percolation_thresold_single(height_file=height_ref_file, footprint_file=footprint_ref_file, density_test=d_test, h_min=5.0, num_cpu=10,
                                     save_path="cc_ref_1000m_Li.npy")
    height_li_file = os.path.join(res_prefix, "Glasgow_height_li2020_clip.tif")
    footprint_li_file = os.path.join(res_prefix, "Glasgow_footprint_li2020_clip.tif")
    calc_percolation_thresold_single(height_file=height_li_file, footprint_file=footprint_li_file, density_test=d_test, h_min=5.0, num_cpu=10,
                                     save_path="cc_li_1000m_Li.npy")
    for res_type in ["senet", "senet_MTL"]:
        height_file = os.path.join(res_prefix, "Glasgow_height_{0}_LiClip.tif".format(res_type))
        footprint_file = os.path.join(res_prefix, "Glasgow_footprint_{0}_LiClip.tif".format(res_type))
        calc_percolation_thresold_single(height_file=height_file, footprint_file=footprint_file,density_test=d_test, h_min=5.0, num_cpu=10,
                                         save_path="cc_{0}_1000m_Li.npy".format(res_type))

    ny = 38
    nx = 66
    res_prefix = os.path.join(infer_prefix, "infer_test_Glasgow", "1000m", "Li_comparison")

    source_ref = {}
    res_type = "ref"
    source_ref["ref"] = {}
    source_ref["ref"]["CCMat"] = "cc_ref_1000m_Li.npy"
    source_ref["ref"]["height"] = os.path.join(res_prefix, "Glasgow_2020_building_height_LiClip.tif")
    source_ref["ref"]["footprint"] = os.path.join(res_prefix, "Glasgow_2020_building_footprint_LiClip.tif")
    calc_mask_fromCCMat_single(source_ref=source_ref, density_test=d_test, ny=ny, nx=nx,
                               save_path=os.path.join(res_prefix, "Glasgow_mask_{0}.tif".format(res_type)),
                               h_min=5.0, ratio=1.0)
    print(os.path.join(res_prefix, "Glasgow_mask_{0}.tif".format(res_type)))

    source_ref = {}
    res_type = "li"
    source_ref["li"] = {}
    source_ref["li"]["CCMat"] = "cc_li_1000m_Li.npy"
    source_ref["li"]["height"] = os.path.join(res_prefix, "Glasgow_height_li2020_clip.tif")
    source_ref["li"]["footprint"] = os.path.join(res_prefix, "Glasgow_footprint_li2020_clip.tif")
    calc_mask_fromCCMat_single(source_ref=source_ref, density_test=d_test, ny=ny, nx=nx,
                               save_path=os.path.join(res_prefix, "Glasgow_mask_{0}.tif".format(res_type)),
                               h_min=5.0, ratio=1.0)
    print(os.path.join(res_prefix, "Glasgow_mask_{0}.tif".format(res_type)))

    source_ref = {}
    for res_type in ["senet", "senet_MTL"]:
        source_ref[res_type] = {}
        source_ref[res_type]["CCMat"] = "cc_{0}_1000m_Li.npy".format(res_type)
        source_ref[res_type]["height"] = os.path.join(res_prefix, "Glasgow_height_{0}_LiClip.tif".format(res_type))
        source_ref[res_type]["footprint"] = os.path.join(res_prefix, "Glasgow_footprint_{0}_LiClip.tif".format(res_type))

    for res_type in ["senet", "senet_MTL"]:
        calc_mask_fromCCMat_single(source_ref=source_ref, density_test=d_test, ny=ny, nx=nx,
                                   save_path=os.path.join(res_prefix, "Glasgow_mask_{0}.tif".format(res_type)),
                                   h_min=5.0, ratio=0.5)
        print(os.path.join(res_prefix, "Glasgow_mask_{0}.tif".format(res_type)))

    nodata_mapping = {"height": -100000, "footprint": -255}

    ny = 38
    nx = 66
    res_prefix = os.path.join(infer_prefix, "infer_test_Glasgow", "1000m", "Li_comparison")

    for res_type in ["ref", "li", "senet", "senet_MTL"]:
    #for res_type in ["ref", "li", "cbam", "cbam_MTL"]:
        th_mask = os.path.join(res_prefix, "Glasgow_mask_{0}.tif".format(res_type))

        for var in ["height", "footprint"]:
            if res_type == "ref":
                res_file = os.path.join(res_prefix, "Glasgow_2020_building_{0}_LiClip.tif".format(var))
                output_file = os.path.join(res_prefix, "Glasgow_2020_building_{0}_LiClip_th.tif".format(var))
                setNodataByMask(input_file=res_file, mask_file=th_mask, output_file=output_file, ny=ny, nx=nx,
                                noData=nodata_mapping[var])
            elif res_type == "li":
                res_file = os.path.join(res_prefix, "Glasgow_{0}_li2020_clip.tif".format(var))
                output_file = os.path.join(res_prefix, "Glasgow_{0}_li2020_clip_th.tif".format(var))
                setNodataByMask(input_file=res_file, mask_file=th_mask, output_file=output_file, ny=ny, nx=nx,
                                noData=nodata_mapping[var])

            else:
                res_file = os.path.join(res_prefix, "Glasgow_{0}_{1}_LiClip.tif".format(var, res_type))
                output_file = os.path.join(res_prefix, "Glasgow_{0}_{1}_LiClip_th.tif".format(var, res_type))

                setNodataByMask(input_file=res_file, mask_file=th_mask, output_file=output_file, ny=ny, nx=nx,
                                noData=nodata_mapping[var])
            print(output_file)
    '''
