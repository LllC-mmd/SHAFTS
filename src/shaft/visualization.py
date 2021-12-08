import os
import h5py
import rasterio
import rasterio.plot as rplt
import geopandas as gpd

from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
import matplotlib.colors as pcolor

from model import *


# ************************* [1] Dataset Visualization *************************
def patch_vis(h5_file, city, target_variable, patch_id, scale_level=0, rescaled=True):
    scale = 0.0001
    s1_prefix = "sentinel_1_"
    s2_prefix = "sentinel_2_"
    band_ref = {"VV": "B1", "VH": "B2", "R": "B1", "G": "B2", "B": "B3", "NIR": "B4"}

    f = h5py.File(h5_file, mode="r")
    h5_keys = [k for k in f[city].keys() if k != target_variable]
    aggregate_suffix = np.unique([s.split("_")[2] for s in h5_keys])

    s1_vv = []
    s1_vh = []
    s2_rgb = []
    s2_nir = []
    val = f[city][target_variable][patch_id]
    print(val)

    for s in aggregate_suffix:
        s1_vv_coef = np.power(10.0, f[city][s1_prefix + s + "_" + band_ref["VV"]][patch_id][scale_level] / 10.0)
        s1_vv.append(s1_vv_coef)
        s1_vh_coef = np.power(10.0, f[city][s1_prefix + s + "_" + band_ref["VH"]][patch_id][scale_level] / 10.0)
        s1_vh.append(s1_vh_coef)

        s2_r = f[city][s2_prefix + s + "_" + band_ref["R"]][patch_id][scale_level] * scale
        s2_g = f[city][s2_prefix + s + "_" + band_ref["G"]][patch_id][scale_level] * scale
        s2_b = f[city][s2_prefix + s + "_" + band_ref["B"]][patch_id][scale_level] * scale

        if rescaled:
            s2_r = rgb_rescale(s2_r)
            s2_g = rgb_rescale(s2_g)
            s2_b = rgb_rescale(s2_b)

        s2_rgb.append(np.concatenate((np.expand_dims(s2_r, axis=2), np.expand_dims(s2_g, axis=2), np.expand_dims(s2_b, axis=2)), axis=-1))
        s2_nir.append(f[city][s2_prefix + s + "_" + band_ref["NIR"]][patch_id][scale_level])

    img = s2_rgb[0]

    plt.imshow(img)
    # plt.imshow(s1_vh[1])
    # plt.imshow(s2_nir[1]*scale)
    plt.show()


# ************************* [2] Model Comparison *************************
def corr_plot(height_ref_file, height_pred_file):
    height_ref = np.load(height_ref_file)
    height_pred = np.load(height_pred_file)

    xy = np.vstack([height_ref, height_pred])
    density = gaussian_kde(xy)(xy)

    fig, ax = plt.subplots(1, 1, figsize=(10, 9))

    n_colors = 20
    v_min = 0.0
    v_max = 0.2
    cm = plt.cm.get_cmap("viridis", n_colors)
    colors = [cm(l) for l in range(0, n_colors)]
    nvals = np.linspace(v_min, v_max, n_colors + 1)
    cmap, norm = pcolor.from_levels_and_colors(nvals, colors)

    s = ax.scatter(height_ref, height_pred, c=density, s=0.1, cmap=cmap, norm=norm, edgecolors=None)

    height_plot = np.linspace(0, max(max(height_pred), max(height_ref)), 10000)
    ax.plot(height_plot, height_plot, color="red", linestyle="--")

    ax.set_xlabel("Reference [m]")
    ax.set_ylabel("Prediction [m]")
    ax.set_aspect(1)

    r2 = metrics.r2_score(y_true=height_ref, y_pred=height_pred)
    ax.set_title("$H_{\mathrm{xgb}}$ ~ $H_{\mathrm{ref}}$ ($R^2$=%.3f)" % r2, fontsize="large")
    position = fig.add_axes([0.91, 0.11, 0.02, 0.75])
    fig.colorbar(s, ax=ax, cax=position)

    # plt.show()
    plt.savefig("xgb_corr.png", dpi=400)


def plot_height_over_city(tif_file, boundary_file):
    image = rasterio.open(tif_file)
    shp_bd = gpd.read_file(boundary_file)

    ref_image = rasterio.open("testCase/infer_test/Glasgow_height_ref.tif")
    ref_data = ref_image.read(1)
    ref_nodata = ref_image.nodata
    ref_data = np.where(ref_data==ref_nodata, np.nan, ref_data)

    fig, ax = plt.subplots(1, 1, figsize=(14, 9))

    # plot prediction results
    h_min = 0.0
    h_max = 40.0

    image_data =image.read(1)

    ref_flatten = ref_data.flatten()
    pred_flatten = image_data.flatten()
    val_loc = np.where(~np.isnan(ref_flatten))[0]
    rmse = metrics.mean_squared_error(y_true=ref_flatten[val_loc], y_pred=pred_flatten[val_loc], squared=False)

    image_bound = list(image.bounds)
    im = ax.imshow(image_data, cmap='Reds', vmin=h_min, vmax=h_max,
                   extent=[image_bound[0], image_bound[2], image_bound[1], image_bound[3]])
    rplt.show(image_data, transform=image.transform, ax=ax, cmap="Reds", vmin=h_min, vmax=h_max)
    ax.set_title("Building Height predicted by Bagging Support Vector Regression [m] (RMSE=%.3f)" % rmse, fontsize="x-large")
    '''
    '''

    '''
    # plot delta value, i.e., h_pred - h_ref
    h_min = -20.0
    h_max = 20.0

    image_data = image.read(1) - ref_data
    image_bound = list(image.bounds)
    im = ax.imshow(image_data, cmap='RdBu', vmin=h_min, vmax=h_max,
                   extent=[image_bound[0], image_bound[2], image_bound[1], image_bound[3]])
    rplt.show(image_data, transform=image.transform, ax=ax, cmap="RdBu", vmin=h_min, vmax=h_max)
    ax.set_title("$H_{\mathrm{baggingSVR}} - H_{\mathrm{ref}}$ [m]", fontsize="x-large")
    '''

    shp_bd.plot(ax=ax, facecolor='none', edgecolor='darkviolet')
    position = fig.add_axes([0.85, 0.11, 0.02, 0.77])
    fig.colorbar(im, ax=ax, cax=position)

    #plt.show()
    plt.savefig("Output_img/Glasgow_bagging_svr.png", dpi=400)


def histogram_compare(model_file, feature_file, height_file, n_bins=100, log_scale=True, saved=True, **kwargs):
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

    # model_est = VVH_model(gamma=0, a=0, b=0, c=0)
    # model_est = RandomForestModel()
    # model_est = BaggingSupportVectorRegressionModel()
    model_est = XGBoostRegressionModel(n_jobs=4)
    model_est.load_model(model_file)

    feature = np.load(feature_file)
    height = np.load(height_file)

    if filter is not None:
        feature = filter.transform(feature)

    if scaler is not None:
        feature = scaler.transform(feature)

    if transformer is not None:
        feature = transformer.transform(feature)

    height_pred = model_est.predict(feature)
    if log_scale:
        height_pred = np.exp(height_pred)

    if saved:
        np.save("height_vvh_est.npy", height_pred)

    h_min = 0.0
    h_max = 20.0
    bins_range = np.linspace(h_min, h_max, n_bins+1)

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.hist(height, bins=bins_range, alpha=0.5, color="firebrick", label="Reference")
    ax.hist(height_pred, bins=bins_range, alpha=0.5, color="royalblue", label="VVH Regression")
    ax.set_xlabel("Height [m]")
    ax.set_title("Building Height Distribution predicted by VVH Regression, $H\in [0, 20]$", fontsize="x-large")

    plt.legend(fontsize="x-large")
    #plt.show()
    plt.savefig("hist_vvh.png", dpi=400)


def plot_confusion_matrix(val_ref_file, val_pred_file, num_quantile=20, num_ticks=None):
    val_ref = np.load(val_ref_file)
    val_pred = np.load(val_pred_file)

    c_mat = get_confusion_matrix(val_true=val_ref, val_pred=val_pred, num_quantile=num_quantile, normed=True)
    q_list = np.linspace(0.0, 1.0, num_quantile + 1)
    q_val_list = [np.quantile(val_ref, q) for q in q_list]

    q_label_list = ["$-\infty$"]
    for i in range(1, len(q_val_list) - 1):
        q_label_list.append("%.2f" % ((q_val_list[i] + q_val_list[i+1])/2.0))
    q_label_list.append("$\infty$")
    q_label_list = np.array(q_label_list)

    fig, ax = plt.subplots(1, figsize=(10, 9))
    c_mat_im = ax.imshow(X=c_mat, cmap="magma_r", vmax=1.0, vmin=0.0)

    if num_ticks is None:
        ax.set_xticks(np.arange(num_quantile + 2))
        ax.set_xticklabels(q_label_list)
        ax.set_yticks(np.arange(num_quantile + 2))
        ax.set_yticklabels(q_label_list)
    else:
        tick_loc = np.linspace(1, num_quantile - 1, num_ticks - 1, dtype=np.int)
        label = q_label_list[tick_loc]
        ax.set_xticks(tick_loc)
        ax.set_xticklabels(label)
        ax.set_yticks(tick_loc)
        ax.set_yticklabels(label)

    position = fig.add_axes([0.88, 0.11, 0.02, 0.77])
    fig.colorbar(c_mat_im, ax=ax, cax=position)
    # plt.show()
    plt.savefig("confusion_mat_xgb.png", dpi=400)


if __name__ == "__main__":
    ds_path = os.path.join("dataset", "patch_data_50pt_s20.h5")
    city = "Beijing@0"

    plot_confusion_matrix("dataset/target_var_s20.npy", "dataset/height_xgb_n50_est.npy", num_quantile=50, num_ticks=10)
    # patch_vis(ds_path, city, "BuildingHeight", 400)

    '''
    plot_height_over_city("testCase/infer_test/Glasgow_height_bagging_svr_s20.tif",
                          "testCase/infer_test/Glasgow/Glasgow_boundary.shp")
    '''

    '''
    model_ref = {"VVH_model": "model_results/lr_VVH_s20.txt",
                 "RandomForestRegression": "model_results/RF_model_s20.pkl",
                 "BaggingSVR": "model_results/bagging_svr_model_s20.pkl",
                 "XGBoostRegression": "model_results/xgb_model_s20.pkl"
                 }

    histogram_compare(model_file=model_ref["VVH_model"], feature_file="dataset/vvh_s20.npy",
                      height_file="dataset/target_var_s20.npy", n_bins=100)
    
    histogram_compare(model_file=model_ref["BaggingSVR"], feature_file="dataset/feature_s20.npy",
                      height_file="dataset/target_var_s20.npy", n_bins=100, filter="model_results/filter_bagging_svr.pkl",
                      scaler="model_results/scaler_bagging_svr.pkl", transformer="model_results/transformer_bagging_svr.pkl")
    '''

    # corr_plot(height_ref_file="dataset/vvh_height_s20.npy", height_pred_file="records/height_vvh_est.npy")


