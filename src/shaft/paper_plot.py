import pandas as pd
import osr
import ogr
from scipy import interpolate
from skimage.metrics import structural_similarity as ssim

import json
import seaborn as sns
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from visualization import *
from DL_train import *
from DL_output import *
from vis_taylor_diagram import *
from dataset import *


def get_NMAD(dta):
    dta_median = np.median(dta)
    nmad = 1.4826 * np.median(np.abs(dta - dta_median))
    return nmad


def symlog_transform(x, base=10, linthresh=2.0):
    x_new = np.where(np.abs(x) < linthresh, x, np.sign(x) * (np.log(np.abs(x) - linthresh + 1) / np.log(base) + linthresh))
    return x_new


def get_LCZ_id(height, footprint):
    other_id = 11
    if footprint < 0.05:
        lcz_id = 0   # non-urban cell
    elif footprint < 0.2:
        if 3 <= height < 10:
            lcz_id = 9   # LCZ-9, sparsely built
        else:
            lcz_id = other_id
    elif 0.2 <= footprint < 0.3 and 5 <= height < 15:
        lcz_id = 10   # LCZ-10, heavy industry
    elif 0.2 <= footprint < 0.4 and 3 <= height < 10:
        lcz_id = 6   # LCZ-6, open low-rise
    elif 0.2 <= footprint < 0.4 and 10 <= height < 25:
        lcz_id = 5   # LCZ-5, open mid-rise
    elif 0.2 <= footprint < 0.4 and height >= 25:
        lcz_id = 4   # LCZ-4, open high-rise
    elif 0.3 <= footprint < 0.5 and 3 <= height < 10:
        lcz_id = 8   # LCZ-8, large low-rise
    elif 0.4 <= footprint < 0.6 and height >= 25:
        lcz_id = 1   # LCZ-1, compact high-rise
    elif 0.4 <= footprint < 0.7 and 3 <= height < 10:
        lcz_id = 2   # LCZ-2, compact low-rise
    elif 0.4 <= footprint < 0.7 and 10 <= height < 25:
        lcz_id = 3   # LCZ-3, compact mid-rise
    elif 0.6 <= footprint < 0.9 and 2 <= height < 4:
        lcz_id = 7   # LCZ-7, lightweight low-rise
    else:
        lcz_id = other_id

    return lcz_id


def get_prediction_ML(base_dir, feature_dir, save_dir, var="height", log_scale=True):
    model_dir_path = {
        "100m": {
            "feature": os.path.join(feature_dir, var, "feature_test_s15_100m.npy"),
            "rf": os.path.join(base_dir, var, "_".join(["rf", var, "100m"])),
            "xgb": os.path.join(base_dir, var, "_".join(["xgb", var, "100m"])),
            "baggingSVR": os.path.join(base_dir, var, "_".join(["baggingSVR", var, "100m"]))
        },
        "250m": {
            "feature": os.path.join(feature_dir, var, "feature_test_s30_250m.npy"),
            "rf": os.path.join(base_dir, var, "_".join(["rf", var, "250m"])),
            "xgb": os.path.join(base_dir, var, "_".join(["xgb", var, "250m"])),
            "baggingSVR": os.path.join(base_dir, var, "_".join(["baggingSVR", var, "250m"]))
        },
        "500m": {
            "feature": os.path.join(feature_dir, var, "feature_test_s60_500m.npy"),
            "rf": os.path.join(base_dir, var, "_".join(["rf", var, "500m"])),
            "xgb": os.path.join(base_dir, var, "_".join(["xgb", var, "500m"])),
            "SVR": os.path.join(base_dir, var, "_".join(["SVR", var, "500m"]))
        },
        "1000m": {
            "feature": os.path.join(feature_dir, var, "feature_test_s120_1000m.npy"),
            "rf": os.path.join(base_dir, var, "_".join(["rf", var, "1000m"])),
            "xgb": os.path.join(base_dir, var, "_".join(["xgb", var, "1000m"])),
            "SVR": os.path.join(base_dir, var, "_".join(["SVR", var, "1000m"]))
        }
    }

    filter_pred = None
    scaler_pred = None
    transformer_pred = None
    model_pred = None

    for resolution in ["100m", "250m", "500m", "1000m"]:
        feat = np.load(model_dir_path[resolution]["feature"])
        for model in ["rf", "xgb", "baggingSVR", "SVR"]:
            if model in model_dir_path[resolution].keys():
                model_dir = model_dir_path[resolution][model]
                model_file_list = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
                # ------load pretrained models
                for f in model_file_list:
                    if f.startswith("filter"):
                        filter_pred = joblib.load(os.path.join(model_dir, f))
                    elif f.startswith("scaler"):
                        scaler_pred = joblib.load(os.path.join(model_dir, f))
                    elif f.startswith("transformer"):
                        transformer_pred = joblib.load(os.path.join(model_dir, f))
                    elif f.startswith(model):
                        if model == "rf":
                            model_pred = RandomForestModel(n_jobs=8)
                        elif model == "xgb":
                            model_pred = XGBoostRegressionModel(n_jobs=8)
                        elif model == "baggingSVR":
                            model_pred = BaggingSupportVectorRegressionModel(n_jobs=4)
                        elif model == "SVR":
                            model_pred = SupportVectorRegressionModel()
                        else:
                            raise NotImplementedError("Unknown ML models")
                        model_pred.load_model(os.path.join(model_dir, f))
                # ------do feature transformation
                if filter_pred is not None:
                    feat_tmp = filter_pred.transform(feat)

                if scaler_pred is not None:
                    feat_tmp = scaler_pred.transform(feat_tmp)

                if transformer_pred is not None:
                    feat_tmp = transformer_pred.transform(feat_tmp)

                # ------do target variable prediction
                var_pred = model_pred.predict(feat_tmp)
                if log_scale:
                    var_pred = np.exp(var_pred)

                # ------save the prediction result
                save_path = os.path.join(save_dir, var, "_".join([model, var, resolution]) + ".npy")
                np.save(save_path, var_pred)

                print("\t".join([model, var, resolution, "finished"]))


def get_prediction_DL(var, resolution, model_name, dataset_path, model_path, input_size, save_dir, activation=None, target_id_shift=None, log_scale=False, cuda_used=True, batch_size=128, num_workers=4, cached=True):
    # ------define data loader
    if model_name.endswith("MTL"):
        val_loader = load_data_lmdb_MTL(dataset_path, batch_size, num_workers, ["50pt"], mode="valid",
                                        cached=cached, log_scale=log_scale)
    else:
        val_loader = load_data_lmdb(dataset_path, batch_size, num_workers, ["50pt"], mode="valid",
                                    cached=cached, target_id_shift=target_id_shift, log_scale=log_scale)

    # ------load pretrained models
    if input_size == 15:
        in_plane = 64
        num_block = 2
    elif input_size == 30:
        in_plane = 64
        num_block = 1
    elif input_size == 60:
        in_plane = 64
        num_block = 1
    else:
        in_plane = 64
        num_block = 1

    if model_name == "senet":
        model_pred = model_SEResNet(in_plane=in_plane, input_channels=6, input_size=input_size,
                                    num_block=num_block, log_scale=log_scale, activation=activation,
                                    cuda_used=cuda_used, trained_record=model_path)
    elif model_name == "senetMTL":
        model_pred = model_SEResNetMTL(in_plane=in_plane, input_channels=6, input_size=input_size,
                                       num_block=num_block, log_scale=log_scale,
                                       cuda_used=cuda_used, trained_record=model_path)
    elif model_name == "cbam":
        model_pred = model_CBAMResNet(in_plane=in_plane, input_channels=6, input_size=input_size, num_block=num_block,
                                      log_scale=log_scale, activation=activation, cuda_used=cuda_used,
                                      trained_record=model_path)
    elif model_name == "cbamMTL":
        model_pred = model_CBAMResNetMTL(in_plane=in_plane, input_channels=6, input_size=input_size,
                                         num_block=num_block,
                                         log_scale=log_scale, cuda_used=cuda_used, trained_record=model_path)
    if cuda_used:
        model_pred = model_pred.cuda()

    model_pred.eval()
    # ------do target variable prediction
    res_list = None
    target_list = None
    if model_name.endswith("MTL"):
        res_list = {"height": [], "footprint": []}
        target_list = {"height": [], "footprint": []}
        for i, sample in enumerate(val_loader):
            input_band, target_footprint, target_height = sample["feature"], sample["footprint"], sample["height"]
            if cuda_used:
                input_band, target_footprint, target_height = input_band.cuda(), target_footprint.cuda(), target_height.cuda()
            with torch.no_grad():
                output_footprint, output_height = model_pred(input_band)
                output_footprint = torch.squeeze(output_footprint)
                output_height = torch.squeeze(output_height)

            pred_footprint = output_footprint.data.cpu().numpy()
            res_list["footprint"].append(pred_footprint)
            target_footprint = target_footprint.cpu().numpy()
            target_list["footprint"].append(target_footprint)
            pred_height = output_height.data.cpu().numpy()
            res_list["height"].append(pred_height)
            target_height = target_height.cpu().numpy()
            target_list["height"].append(target_height)
    else:
        res_list = {var: []}
        target_list = {var: []}
        for i, sample in enumerate(val_loader):
            input_band, target = sample["feature"], sample["value"]
            if cuda_used:
                input_band, target = input_band.cuda(), target.cuda()
            with torch.no_grad():
                output = model_pred(input_band)
                output = torch.squeeze(output)

            output = output.data.cpu().numpy()
            res_list[var].append(output)
            target = target.cpu().numpy()
            target_list[var].append(target)

    # ------save the prediction and target result
    for k in res_list.keys():
        res_list[k] = np.concatenate(res_list[k], axis=0)
        if log_scale:
            res_list[k] = np.exp(res_list[k])
        save_path = os.path.join(save_dir, k, "_".join([model_name, k, resolution]) + ".npy")
        np.save(save_path, res_list[k])
    for k in target_list.keys():
        target_list[k] = np.concatenate(target_list[k], axis=0)
        save_path = os.path.join(save_dir, k, "_".join([k, resolution, "refDL"]) + ".npy")
        np.save(save_path, target_list[k])


def get_prediction_DL_batch(base_dir, ds_dir, save_dir, var="height", log_scale=False, cuda_used=True, batch_size=128, num_workers=4, cached=True):
    backbone = "senet"

    if var == "height":
        target_id_shift = 2
        activation = "relu"
        model_dir_path = {
            "100m": {
                "validation_dataset": os.path.join(ds_dir, "patch_data_50pt_s15_100m_valid.lmdb"),
                "senet": os.path.join(base_dir, var, "check_pt_senet_100m", "experiment_4", "checkpoint.pth.tar"),
                "cbam": os.path.join(base_dir, var, "check_pt_cbam_100m", "experiment_1", "checkpoint.pth.tar"),
                "senetMTL": os.path.join(base_dir, var, "check_pt_senet_100m_MTL", "experiment_11", "checkpoint.pth.tar"),
                "cbamMTL": os.path.join(base_dir, var, "check_pt_cbam_100m_MTL", "experiment_5", "checkpoint.pth.tar")
            },
            "250m": {
                "validation_dataset": os.path.join(ds_dir, "patch_data_50pt_s30_250m_valid.lmdb"),
                "senet": os.path.join(base_dir, var, "check_pt_senet_250m", "experiment_4", "checkpoint.pth.tar"),
                "cbam": os.path.join(base_dir, var, "check_pt_cbam_250m", "experiment_2", "checkpoint.pth.tar"),
                "senetMTL": os.path.join(base_dir, var, "check_pt_senet_250m_MTL", "experiment_11", "checkpoint.pth.tar"),
                "cbamMTL": os.path.join(base_dir, var, "check_pt_cbam_250m_MTL", "experiment_5", "checkpoint.pth.tar")
            },
            "500m": {
                "validation_dataset": os.path.join(ds_dir, "patch_data_50pt_s60_500m_valid.lmdb"),
                "senet": os.path.join(base_dir, var, "check_pt_senet_500m", "experiment_4", "checkpoint.pth.tar"),
                "cbam": os.path.join(base_dir, var, "check_pt_cbam_500m", "experiment_1", "checkpoint.pth.tar"),
                "senetMTL": os.path.join(base_dir, var, "check_pt_senet_500m_MTL", "experiment_11", "checkpoint.pth.tar"),
                "cbamMTL": os.path.join(base_dir, var, "check_pt_cbam_500m_MTL", "experiment_5", "checkpoint.pth.tar")
            },
            "1000m": {
                "validation_dataset": os.path.join(ds_dir, "patch_data_50pt_s120_1000m_valid.lmdb"),
                "senet": os.path.join(base_dir, var, "check_pt_senet_1000m", "experiment_3", "checkpoint.pth.tar"),
                "cbam": os.path.join(base_dir, var, "check_pt_cbam_1000m", "experiment_1", "checkpoint.pth.tar"),
                "senetMTL": os.path.join(base_dir, var, "check_pt_senet_1000m_MTL", "experiment_11", "checkpoint.pth.tar"),
                "cbamMTL": os.path.join(base_dir, var, "check_pt_cbam_1000m_MTL", "experiment_4", "checkpoint.pth.tar")
            }
        }
    elif var == "footprint":
        target_id_shift = 1
        activation = "sigmoid"
        model_dir_path = {
            "100m": {
                "validation_dataset": os.path.join(ds_dir, "patch_data_50pt_s15_100m_valid.lmdb"),
                "senet": os.path.join(base_dir, var, "check_pt_senet_100m", "experiment_4", "checkpoint.pth.tar"),
                "cbam": os.path.join(base_dir, var, "check_pt_cbam_100m", "experiment_1", "checkpoint.pth.tar"),
                "senetMTL": os.path.join(base_dir, var, "check_pt_senet_100m_MTL", "experiment_11", "checkpoint.pth.tar"),
                "cbamMTL": os.path.join(base_dir, var, "check_pt_cbam_100m_MTL", "experiment_5", "checkpoint.pth.tar")
            },
            "250m": {
                "validation_dataset": os.path.join(ds_dir, "patch_data_50pt_s30_250m_valid.lmdb"),
                "senet": os.path.join(base_dir, var, "check_pt_senet_250m", "experiment_3", "checkpoint.pth.tar"),
                "cbam": os.path.join(base_dir, var, "check_pt_cbam_250m", "experiment_0", "checkpoint.pth.tar"),
                "senetMTL": os.path.join(base_dir, var, "check_pt_senet_250m_MTL", "experiment_11", "checkpoint.pth.tar"),
                "cbamMTL": os.path.join(base_dir, var, "check_pt_cbam_250m_MTL", "experiment_5", "checkpoint.pth.tar")
            },
            "500m": {
                "validation_dataset": os.path.join(ds_dir, "patch_data_50pt_s60_500m_valid.lmdb"),
                "senet": os.path.join(base_dir, var, "check_pt_senet_500m", "experiment_3", "checkpoint.pth.tar"),
                "cbam": os.path.join(base_dir, var, "check_pt_cbam_500m", "experiment_1", "checkpoint.pth.tar"),
                "senetMTL": os.path.join(base_dir, var, "check_pt_senet_500m_MTL", "experiment_11", "checkpoint.pth.tar"),
                "cbamMTL": os.path.join(base_dir, var, "check_pt_cbam_500m_MTL", "experiment_5", "checkpoint.pth.tar")
            },
            "1000m": {
                "validation_dataset": os.path.join(ds_dir, "patch_data_50pt_s120_1000m_valid.lmdb"),
                "senet": os.path.join(base_dir, var, "check_pt_senet_1000m", "experiment_5", "checkpoint.pth.tar"),
                "cbam": os.path.join(base_dir, var, "check_pt_cbam_1000m", "experiment_2", "checkpoint.pth.tar"),
                "senetMTL": os.path.join(base_dir, var, "check_pt_senet_1000m_MTL", "experiment_11", "checkpoint.pth.tar"),
                "cbamMTL": os.path.join(base_dir, var, "check_pt_cbam_1000m_MTL", "experiment_4", "checkpoint.pth.tar")
            }
        }
    else:
        raise NotImplementedError("Unknown target variable")

    for resolution, input_size in [("100m", 15), ("250m", 30), ("500m", 60), ("1000m", 120)]:
        val_ds_path = model_dir_path[resolution]["validation_dataset"]
        for model in [backbone, backbone + "MTL"]:
            model_file = model_dir_path[resolution][model]
            if model == backbone:
                get_prediction_DL(var, resolution, model,
                                  dataset_path=val_ds_path,
                                  model_path=model_file,
                                  input_size=input_size,
                                  save_dir=save_dir,
                                  activation=activation, target_id_shift=target_id_shift,
                                  log_scale=log_scale, cuda_used=cuda_used,
                                  batch_size=batch_size,
                                  num_workers=num_workers, cached=cached)
            elif model == backbone + "MTL":
                get_prediction_DL(var, resolution, model,
                                  dataset_path=val_ds_path,
                                  model_path=model_file,
                                  input_size=input_size,
                                  save_dir=save_dir,
                                  activation=None, target_id_shift=None,
                                  log_scale=log_scale, cuda_used=cuda_used,
                                  batch_size=batch_size,
                                  num_workers=num_workers, cached=cached)


# Fig. 1. Methodological flowchart for the estimation of global building height and building footprint


# Fig. 2. Location of sample cities used in this research
def fig_2_dataset_loc_plot(csv_path, path_prefix=None, report=False):
    # ---read the center location, i.e., (lon, lat) of each city
    target_spatialRef = osr.SpatialReference()
    target_spatialRef.ImportFromEPSG(4326)
    target_spatialRef.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    xy_center = {}
    df = pd.read_csv(csv_path)
    for row_id in df.index:
        # ------read basic information of Shapefile
        city_name = df.loc[row_id]["City"]
        dta_path = os.path.join(path_prefix, df.loc[row_id]["SHP_Path"])
        # ---------get Shapefile extent
        shp_ds = ogr.Open(dta_path, 0)
        shp_layer = shp_ds.GetLayer()
        x_min, x_max, y_min, y_max = shp_layer.GetExtent()
        x_center = 0.5 * (x_min + x_max)
        y_center = 0.5 * (y_min + y_max)
        # ---------coordinate transformation
        shp_spatialRef = shp_layer.GetSpatialRef()
        coordTrans = osr.CoordinateTransformation(shp_spatialRef, target_spatialRef)
        lon, lat = coordTrans.TransformPoint(x_center, y_center)[0:2]
        xy_center[city_name] = [lon, lat]

    # ---plot the center location on the world map
    projection = ccrs.PlateCarree()
    ax = plt.axes(projection=projection)
    fig = plt.gcf()
    fig.set_size_inches(20, 10)

    ax.stock_img()

    for city_name, loc in xy_center.items():
        x, y = loc
        if report:
            print(city_name, ", ", str(x), ",", str(y))
        ax.plot(x, y, "or", markersize=3)

    k_test = list(xy_center.keys())[0]
    ax.plot(xy_center[k_test][0], xy_center[k_test][1], "or", markersize=5, label="center of sample cities")
    ax.set_xticks(np.linspace(-180, 180, 5), crs=projection)
    ax.set_yticks(np.linspace(-90, 90, 5), crs=projection)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    plt.legend()
    # plt.show()
    plt.savefig("Fig-1-location.png", dpi=500)


# Fig. 1. Distributions of reference building height and building footprint under diﬀerent target resolutions
# i.e., 100m, 250m, 500m, 1000m
def fig_1_dataset_distribution_plot(plot_type="violin", path_prefix=None):
    # ---read the HDF5 dataset to get target variable
    db_path_list = {"100m": os.path.join(path_prefix, "patch_data_50pt_s15_100m.h5"),
                    "250m": os.path.join(path_prefix, "patch_data_50pt_s30_250m.h5"),
                    "500m": os.path.join(path_prefix, "patch_data_50pt_s60_500m.h5"),
                    "1000m": os.path.join(path_prefix, "patch_data_50pt_s120_1000m.h5")}
    target_height = []
    target_footprint = []
    labels = ["100m", "250m", "500m", "1000m"]
    for res in labels:
        tmp_height = []
        tmp_footprint = []
        hf_db = h5py.File(db_path_list[res], "r")
        for city in sorted(hf_db.keys()):
            tmp_height.append(np.array(hf_db[city]["BuildingHeight"]))
            tmp_footprint.append(np.array(hf_db[city]["BuildingFootprint"]))
        tmp_height = np.concatenate(tmp_height, axis=0)
        tmp_footprint = np.concatenate(tmp_footprint, axis=0)
        target_height.append(tmp_height)
        target_footprint.append(tmp_footprint)

    # ---plot violin plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    if plot_type == "violin":
        sns.violinplot(data=target_height, ax=ax[0], cut=0, gridsize=100)
        sns.violinplot(data=target_footprint, ax=ax[1], cut=0, gridsize=100)
    elif plot_type == "box":
        sns.boxplot(data=target_height, ax=ax[0], fliersize=0.5)
        sns.boxplot(data=target_footprint, ax=ax[1], fliersize=0.5)
    else:
        raise NotImplementedError("Unknown plot type")

    ax[0].get_xaxis().set_tick_params(direction='out')
    ax[0].xaxis.set_ticks_position('bottom')
    ax[0].set_xticks(np.arange(0, len(labels)))
    ax[0].set_xticklabels(labels)
    ax[0].set_yscale('log')
    ax[0].set_ylabel("$H_{\mathrm{ave}}$ [m]")

    ax[1].get_xaxis().set_tick_params(direction='out')
    ax[1].xaxis.set_ticks_position('bottom')
    ax[1].set_xticks(np.arange(0, len(labels)))
    ax[1].set_xticklabels(labels)
    ax[1].set_yscale('log')
    ax[1].set_ylabel("$\lambda_p$ $[\mathrm{m}^2/\mathrm{m}^2]$")

    # plt.show()
    plt.savefig("Fig-1-distribution.pdf", bbox_inches='tight', dpi=600)


# Fig. 4. Performance of models at different resolution (plot pred-ref corr curve, scatter plot with density)
# i.e., [BuildingHeight, BuildingFootprint] x [RandomForest, BaggingSVR, XGBoost, CNN] x [100m, 250m, 500m, 1000m]
def fig_4_model_metric_corr(res_prefix, var="height"):
    if var == "height":
        unit = "m"
        ref_label = "$H_{\mathrm{ave}}$"
        pred_label = "$\widehat{H_{\mathrm{ave}}}$"
        density_min = 1e-3
        density_max = 1.0
        vmin_mapping = {"100m": 1.0, "250m": 1.0, "500m": 1.0, "1000m": 1.0}
        vmax_mapping = {"100m": 500.0, "250m": 500.0, "500m": 500.0, "1000m": 500.0}
    elif var == "footprint":
        unit = "$\mathrm{m}^2$/$\mathrm{m}^2$"
        ref_label = "$\lambda_p$"
        pred_label = "$\widehat{\lambda_p}$"
        density_min = 1e-4
        density_max = 1.0
        # vmin_mapping = {"100m": 1e-3, "250m": 1e-4, "500m": 1e-4, "1000m": 1e-6}
        vmin_mapping = {"100m": 1e-6, "250m": 1e-6, "500m": 1e-6, "1000m": 1e-6}
        vmax_mapping = {"100m": 1.0, "250m": 1.0, "500m": 1.0, "1000m": 1.0}
    else:
        raise NotImplementedError("Unknown target variable")

    res_path = {
        "100m": {
            "DL_reference": os.path.join(res_prefix, var, "_".join([var, "100m", "refDL"]) + ".npy"),
            "ML_reference": os.path.join(res_prefix, var, "_".join([var, "100m", "ref"]) + ".npy"),
            "rf": os.path.join(res_prefix, var, "_".join(["rf", var, "100m"]) + ".npy"),
            "xgb": os.path.join(res_prefix, var, "_".join(["xgb", var, "100m"]) + ".npy"),
            "baggingSVR": os.path.join(res_prefix, var, "_".join(["baggingSVR", var, "100m"]) + ".npy"),
            "senet": os.path.join(res_prefix, var, "_".join(["senet", var, "100m"]) + ".npy"),
            "senetMTL": os.path.join(res_prefix, var, "_".join(["senetMTL", var, "100m"]) + ".npy")
        },
        "250m": {
            "DL_reference": os.path.join(res_prefix, var, "_".join([var, "250m", "refDL"]) + ".npy"),
            "ML_reference": os.path.join(res_prefix, var, "_".join([var, "250m", "ref"]) + ".npy"),
            "rf": os.path.join(res_prefix, var, "_".join(["rf", var, "250m"]) + ".npy"),
            "xgb": os.path.join(res_prefix, var, "_".join(["xgb", var, "250m"]) + ".npy"),
            "baggingSVR": os.path.join(res_prefix, var, "_".join(["baggingSVR", var, "250m"]) + ".npy"),
            "senet": os.path.join(res_prefix, var, "_".join(["senet", var, "250m"]) + ".npy"),
            "senetMTL": os.path.join(res_prefix, var, "_".join(["senetMTL", var, "250m"]) + ".npy")
        },
        "500m": {
            "DL_reference": os.path.join(res_prefix, var, "_".join([var, "500m", "refDL"]) + ".npy"),
            "ML_reference": os.path.join(res_prefix, var, "_".join([var, "500m", "ref"]) + ".npy"),
            "rf": os.path.join(res_prefix, var, "_".join(["rf", var, "500m"]) + ".npy"),
            "xgb": os.path.join(res_prefix, var, "_".join(["xgb", var, "500m"]) + ".npy"),
            "SVR": os.path.join(res_prefix, var, "_".join(["SVR", var, "500m"]) + ".npy"),
            "senet": os.path.join(res_prefix, var, "_".join(["senet", var, "500m"]) + ".npy"),
            "senetMTL": os.path.join(res_prefix, var, "_".join(["senetMTL", var, "500m"]) + ".npy")
        },
        "1000m": {
            "DL_reference": os.path.join(res_prefix, var, "_".join([var, "1000m", "refDL"]) + ".npy"),
            "ML_reference": os.path.join(res_prefix, var, "_".join([var, "1000m", "ref"]) + ".npy"),
            "rf": os.path.join(res_prefix, var, "_".join(["rf", var, "1000m"]) + ".npy"),
            "xgb": os.path.join(res_prefix, var, "_".join(["xgb", var, "1000m"]) + ".npy"),
            "SVR": os.path.join(res_prefix, var, "_".join(["SVR", var, "1000m"]) + ".npy"),
            "senet": os.path.join(res_prefix, var, "_".join(["senet", var, "1000m"]) + ".npy"),
            "senetMTL": os.path.join(res_prefix, var, "_".join(["senetMTL", var, "1000m"]) + ".npy")
        }
    }

    report_df = pd.DataFrame(columns=["Model", "Variable", "Resolution", "RMSE", "MAE", "ME", "NMAD", "CC", "R^2"])

    name_mapping = {"rf": "RFR", "xgb": "XGBoostR", "SVR": "(bagging)SVR", "baggingSVR": "(bagging)SVR",
                    "senet": "SENet-STL", "senetMTL": "SENet-MTL"}

    col_mapping = {"rf": 0, "SVR": 1, "baggingSVR": 1, "xgb": 2, "senet": 3, "senetMTL": 4}
    col_label_mapping = {0: "RFR", 1: "(bagging)SVR", 2: "XGBoostR", 3: "SENet-STL", 4: "SENet-MTL"}
    row_mapping = {"100m": 0, "250m": 1, "500m": 2, "1000m": 3}
    row_label_mapping = {0: "100m", 1: "250m", 2: "500m", 3: "1000m"}

    fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(16, 12))

    for model in ["rf", "xgb", "baggingSVR", "SVR", "senet", "senetMTL"]:
        for resolution in ["100m", "250m", "500m", "1000m"]:
            var_ref_dl = np.load(res_path[resolution]["DL_reference"])
            var_ref_ml = np.load(res_path[resolution]["ML_reference"])
            row_id = row_mapping[resolution]
            if model in res_path[resolution].keys():
                if model in ["senet", "senetMTL"]:
                    var_ref = var_ref_dl
                else:
                    var_ref = var_ref_ml
                col_id = col_mapping[model]
                var_pred = np.load(res_path[resolution][model])
                # ------density estimation for scatter plot
                xy = np.vstack([var_ref, var_pred])
                density = gaussian_kde(xy)(xy)
                print(resolution, np.max(density), np.min(density))
                density = (density - np.min(density)) / (np.max(density) - np.min(density))
                # ------calculate some metrics
                num_sample = len(var_ref)
                cc = np.corrcoef(x=var_ref, y=var_pred)[0, 1]
                r2 = metrics.r2_score(y_true=var_ref, y_pred=var_pred)
                r2_weighted = metrics.r2_score(y_true=var_ref, y_pred=var_pred, sample_weight=density)
                rmse = np.sqrt(metrics.mean_squared_error(y_true=var_ref, y_pred=var_pred))
                rmse_weighted = np.sqrt(metrics.mean_squared_error(y_true=var_ref, y_pred=var_pred, sample_weight=density))
                mae = metrics.mean_absolute_error(y_true=var_ref, y_pred=var_pred)
                err = var_pred - var_ref
                se = np.mean(err)
                nmad = get_NMAD(err)
                result_summary = {"Model": name_mapping[model],
                                  "Variable": var,
                                  "Resolution": resolution,
                                  "RMSE": rmse,
                                  "MAE": mae,
                                  "ME": se,
                                  "NMAD": nmad,
                                  "CC": cc,
                                  "R^2": r2}
                report_df = report_df.append(result_summary, ignore_index=True)
                # ------scatter plot
                s = ax[row_id, col_id].scatter(var_ref, var_pred, c=density, s=0.05, cmap="jet",
                                               norm=colors.LogNorm(vmin=density_min, vmax=density_max), edgecolors=None)

                height_plot = np.linspace(0, max(max(var_pred), max(var_ref)), 10000)
                ax[row_id, col_id].plot(height_plot, height_plot, color="red", linestyle="--")

                ax[row_id, col_id].set_xscale("log")
                ax[row_id, col_id].set_yscale("log")
                ax[row_id, col_id].set_xlim([vmin_mapping[resolution], vmax_mapping[resolution]])
                ax[row_id, col_id].set_ylim([vmin_mapping[resolution], vmax_mapping[resolution]])

                if row_id == 0:
                    ax[row_id, col_id].set_title("{0}".format(col_label_mapping[col_id]), size="large")
                if row_id == 3:
                    ax[row_id, col_id].set_xlabel("{0} [{1}]".format(ref_label, unit), size="large")
                else:
                    ax[row_id, col_id].tick_params(axis="x", which="both", bottom=False, labelbottom=False)

                if col_id == 0:
                    ax[row_id, col_id].set_ylabel("{0} (N={1})\n{2} [{3}]".format(row_label_mapping[row_id], num_sample,
                                                                                  pred_label, unit), fontsize="large")
                else:
                    ax[row_id, col_id].tick_params(axis="y", which="both", left=False, labelleft=False)

                if row_id == 3 and col_id == 2:
                    position = ax[row_id, col_id].inset_axes([-0.7, -0.3, 2.40, 0.04], transform=ax[row_id, col_id].transAxes)
                    cbar = fig.colorbar(s, ax=ax[row_id, col_id], cax=position, orientation="horizontal")

                # ------add metric text
                metric_str = "\n".join(["RMSE: %.2f, MAE: %.2f" % (rmse, mae),
                                        "ME: %.2f, NMAD: %.2f" % (se, nmad),
                                        "CC: %.2f, $R^2$: %.3f" % (cc, r2),
                                        "wRMSE: %.2f, w$R^2$: %.3f" % (rmse_weighted, r2_weighted)])
                ax[row_id, col_id].text(0.42, 0.25, metric_str, ha="left", va="top",
                                        fontsize="x-small", transform=ax[row_id, col_id].transAxes)

    report_df.to_csv("Fig-4-metrics_{0}.csv".format(var), index=False)

    plt.savefig("Fig-4-metrics_corr_{0}.pdf".format(var), bbox_inches='tight', dpi=600)


def fig_4_model_metric_hist_2x4(res_prefix, num_bin=1000, log_scale=True):
    res_summary = {}
    for var in ["height", "footprint"]:
        res_path = {
            "100m": {
                "DL_reference": os.path.join(res_prefix, var, "_".join([var, "100m", "refDL"]) + ".npy"),
                "ML_reference": os.path.join(res_prefix, var, "_".join([var, "100m", "ref"]) + ".npy"),
                "rf": os.path.join(res_prefix, var, "_".join(["rf", var, "100m"]) + ".npy"),
                "xgb": os.path.join(res_prefix, var, "_".join(["xgb", var, "100m"]) + ".npy"),
                "baggingSVR": os.path.join(res_prefix, var, "_".join(["baggingSVR", var, "100m"]) + ".npy"),
                "senet": os.path.join(res_prefix, var, "_".join(["senet", var, "100m"]) + ".npy"),
                "senetMTL": os.path.join(res_prefix, var, "_".join(["senetMTL", var, "100m"]) + ".npy")
            },
            "250m": {
                "DL_reference": os.path.join(res_prefix, var, "_".join([var, "250m", "refDL"]) + ".npy"),
                "ML_reference": os.path.join(res_prefix, var, "_".join([var, "250m", "ref"]) + ".npy"),
                "rf": os.path.join(res_prefix, var, "_".join(["rf", var, "250m"]) + ".npy"),
                "xgb": os.path.join(res_prefix, var, "_".join(["xgb", var, "250m"]) + ".npy"),
                "baggingSVR": os.path.join(res_prefix, var, "_".join(["baggingSVR", var, "250m"]) + ".npy"),
                "senet": os.path.join(res_prefix, var, "_".join(["senet", var, "250m"]) + ".npy"),
                "senetMTL": os.path.join(res_prefix, var, "_".join(["senetMTL", var, "250m"]) + ".npy")
            },
            "500m": {
                "DL_reference": os.path.join(res_prefix, var, "_".join([var, "500m", "refDL"]) + ".npy"),
                "ML_reference": os.path.join(res_prefix, var, "_".join([var, "500m", "ref"]) + ".npy"),
                "rf": os.path.join(res_prefix, var, "_".join(["rf", var, "500m"]) + ".npy"),
                "xgb": os.path.join(res_prefix, var, "_".join(["xgb", var, "500m"]) + ".npy"),
                "SVR": os.path.join(res_prefix, var, "_".join(["SVR", var, "500m"]) + ".npy"),
                "senet": os.path.join(res_prefix, var, "_".join(["senet", var, "500m"]) + ".npy"),
                "senetMTL": os.path.join(res_prefix, var, "_".join(["senetMTL", var, "500m"]) + ".npy")
            },
            "1000m": {
                "DL_reference": os.path.join(res_prefix, var, "_".join([var, "1000m", "refDL"]) + ".npy"),
                "ML_reference": os.path.join(res_prefix, var, "_".join([var, "1000m", "ref"]) + ".npy"),
                "rf": os.path.join(res_prefix, var, "_".join(["rf", var, "1000m"]) + ".npy"),
                "xgb": os.path.join(res_prefix, var, "_".join(["xgb", var, "1000m"]) + ".npy"),
                "SVR": os.path.join(res_prefix, var, "_".join(["SVR", var, "1000m"]) + ".npy"),
                "senet": os.path.join(res_prefix, var, "_".join(["senet", var, "1000m"]) + ".npy"),
                "senetMTL": os.path.join(res_prefix, var, "_".join(["senetMTL", var, "1000m"]) + ".npy")
            }
        }
        res_summary[var] = res_path

    palette_pdf = {"Reference": [105 / 255, 105 / 255, 105 / 255, 105 / 255]}
    palette_model = {"RFR": "darkgreen", "XGBoostR": "purple", "(bagging)SVR": "darkblue",
                     "SENet-STL": "firebrick", "SENet-MTL": "darkorange"}

    name_mapping = {"rf": "RFR", "xgb": "XGBoostR", "SVR": "(bagging)SVR", "baggingSVR": "(bagging)SVR",
                    "senet": "SENet-STL", "senetMTL": "SENet-MTL"}

    col_mapping = {"100m": 0, "250m": 1, "500m": 2, "1000m": 3}
    col_label_mapping = {0: "100m", 1: "250m", 2: "500m", 3: "1000m"}
    row_mapping = {"height": 0, "footprint": 1}
    row_label_mapping = {0: "$H_{\mathrm{ave}}$", 1: "$\lambda_p$"}

    vmin_mapping = {
        "height": {"100m": 1.0, "250m": 1.0, "500m": 1.0, "1000m": 1.0},
        "footprint": {"100m": 1e-5, "250m": 1e-5, "500m": 1e-5, "1000m": 1e-5}
    }
    vmax_mapping = {
        "height":  {"100m": 500.0, "250m": 500.0, "500m": 500.0, "1000m": 500.0},
        "footprint": {"100m": 1.0, "250m": 1.0, "500m": 1.0, "1000m": 1.0}
    }

    num_max_mapping = {
        "height": {"100m": 0.05, "250m": 0.05, "500m": 0.05, "1000m": 0.05},
        "footprint": {"100m": 0.02, "250m": 0.02, "500m": 0.02, "1000m": 0.02}
    }

    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(8, 4))
    for var in ["height", "footprint"]:
        row_id = row_mapping[var]
        res_name = row_label_mapping[row_id]
        for resolution in ["100m", "250m", "500m", "1000m"]:
            col_id = col_mapping[resolution]
            # ---plot histogram for reference
            var_ref = np.load(res_summary[var][resolution]["DL_reference"])
            var_record = var_ref
            num_sample = len(var_ref)
            model_list = ["Reference"]
            label = ["Reference" for i in range(0, len(var_record))]
            summary_df = pd.DataFrame({res_name: var_ref, "category": label})
            g = sns.histplot(data=summary_df, x=res_name, hue="category", hue_order=model_list,
                             stat="probability",
                             bins=num_bin, log_scale=log_scale, legend=False, palette=palette_pdf, element="step",
                             ax=ax[row_id, col_id], linewidth=1.0)
            if col_id != 0:
                g.set(ylabel=None)
            g.set(xlabel=None)

            for model in ["rf", "xgb", "baggingSVR", "SVR", "senet", "senetMTL"]:
                if model in res_summary[var][resolution].keys():
                    model_list = [name_mapping[model]]
                    var_pred = np.load(res_summary[var][resolution][model])
                    if log_scale:
                        if var == "height":
                            var_pred = np.where(var_pred < 2.0, 2.0, var_pred)
                        else:
                            var_pred = np.where(var_pred < 1e-6, 1e-6, var_pred)
                    label = [name_mapping[model] for i in range(0, len(var_pred))]

                    summary_df = pd.DataFrame({res_name: var_pred, "category": label})
                    g = sns.histplot(data=summary_df, x=row_label_mapping[row_id], hue="category", hue_order=model_list,
                                     stat="probability",
                                     bins=num_bin, log_scale=log_scale, legend=False, palette=palette_model, element="step",
                                     fill=False, ax=ax[row_id, col_id], linewidth=0.6)

                ax[row_id, col_id].set_xlabel(res_name, size="large")

                if col_id != 0:
                    g.set(ylabel=None)
                g.set(xlabel=None)

            if row_id == 0:
                ax[row_id, col_id].set_title("{0} (N={1})".format(col_label_mapping[col_id], int(num_sample)), size="large")

            if col_id == 0:
                ax[row_id, col_id].set_ylabel("Prob. ({0})".format(row_label_mapping[row_id]), size="large")
            else:
                ax[row_id, col_id].tick_params(axis="y", which="both", left=False, labelleft=False)

            if row_id == 1 and col_id == 0:
                legend_list = [Patch(facecolor=palette_pdf["Reference"], edgecolor=palette_pdf["Reference"],
                                     label="Reference")]
                for model in ["rf", "xgb", "SVR", "senet", "senetMTL"]:
                    legend_list.append(Line2D([], [], color=palette_model[name_mapping[model]], label=name_mapping[model]))

                ax[row_id, col_id].legend(handles=legend_list, bbox_to_anchor=[5.15, -0.5], fontsize='medium',
                                          loc='lower right', ncol=6)

            ax[row_id, col_id].set_xlim([vmin_mapping[var][resolution], vmax_mapping[var][resolution]])
            ax[row_id, col_id].set_ylim([0, num_max_mapping[var][resolution]])

    # plt.show()
    plt.savefig("Fig-4-metrics_hist_2x4.pdf", bbox_inches='tight', dpi=600)


def fig_4_r2_partition(res_prefix):
    report_df = pd.DataFrame(columns=["Model", "Variable", "Resolution", "1-R^2", "nME^2", "nRMSE_centered^2"])

    for var in ["height", "footprint"]:
        res_path = {
            "100 m": {
                "DL_reference": os.path.join(res_prefix, var, "_".join([var, "100m", "refDL"]) + ".npy"),
                "ML_reference": os.path.join(res_prefix, var, "_".join([var, "100m", "ref"]) + ".npy"),
                "rf": os.path.join(res_prefix, var, "_".join(["rf", var, "100m"]) + ".npy"),
                "xgb": os.path.join(res_prefix, var, "_".join(["xgb", var, "100m"]) + ".npy"),
                "baggingSVR": os.path.join(res_prefix, var, "_".join(["baggingSVR", var, "100m"]) + ".npy"),
                "senet": os.path.join(res_prefix, var, "_".join(["senet", var, "100m"]) + ".npy"),
                "senetMTL": os.path.join(res_prefix, var, "_".join(["senetMTL", var, "100m"]) + ".npy")
            },
            "250 m": {
                "DL_reference": os.path.join(res_prefix, var, "_".join([var, "250m", "refDL"]) + ".npy"),
                "ML_reference": os.path.join(res_prefix, var, "_".join([var, "250m", "ref"]) + ".npy"),
                "rf": os.path.join(res_prefix, var, "_".join(["rf", var, "250m"]) + ".npy"),
                "xgb": os.path.join(res_prefix, var, "_".join(["xgb", var, "250m"]) + ".npy"),
                "baggingSVR": os.path.join(res_prefix, var, "_".join(["baggingSVR", var, "250m"]) + ".npy"),
                "senet": os.path.join(res_prefix, var, "_".join(["senet", var, "250m"]) + ".npy"),
                "senetMTL": os.path.join(res_prefix, var, "_".join(["senetMTL", var, "250m"]) + ".npy")
            },
            "500 m": {
                "DL_reference": os.path.join(res_prefix, var, "_".join([var, "500m", "refDL"]) + ".npy"),
                "ML_reference": os.path.join(res_prefix, var, "_".join([var, "500m", "ref"]) + ".npy"),
                "rf": os.path.join(res_prefix, var, "_".join(["rf", var, "500m"]) + ".npy"),
                "xgb": os.path.join(res_prefix, var, "_".join(["xgb", var, "500m"]) + ".npy"),
                "SVR": os.path.join(res_prefix, var, "_".join(["SVR", var, "500m"]) + ".npy"),
                "senet": os.path.join(res_prefix, var, "_".join(["senet", var, "500m"]) + ".npy"),
                "senetMTL": os.path.join(res_prefix, var, "_".join(["senetMTL", var, "500m"]) + ".npy")
            },
            "1000 m": {
                "DL_reference": os.path.join(res_prefix, var, "_".join([var, "1000m", "refDL"]) + ".npy"),
                "ML_reference": os.path.join(res_prefix, var, "_".join([var, "1000m", "ref"]) + ".npy"),
                "rf": os.path.join(res_prefix, var, "_".join(["rf", var, "1000m"]) + ".npy"),
                "xgb": os.path.join(res_prefix, var, "_".join(["xgb", var, "1000m"]) + ".npy"),
                "SVR": os.path.join(res_prefix, var, "_".join(["SVR", var, "1000m"]) + ".npy"),
                "senet": os.path.join(res_prefix, var, "_".join(["senet", var, "1000m"]) + ".npy"),
                "senetMTL": os.path.join(res_prefix, var, "_".join(["senetMTL", var, "1000m"]) + ".npy")
            }
        }

        name_mapping = {"rf": "RFR", "xgb": "XGBoostR", "SVR": "(bagging)SVR", "baggingSVR": "(bagging)SVR",
                        "senet": "SENet-STL", "senetMTL": "SENet-MTL"}

        print("\t".join(["Resolution", "Model", "1-R^2", "MSE_centered^2", "ME^2"]))
        for resolution in ["100 m", "250 m", "500 m", "1000 m"]:
            var_ref_dl = np.load(res_path[resolution]["DL_reference"])
            var_ref_ml = np.load(res_path[resolution]["ML_reference"])
            for model in ["rf", "xgb", "baggingSVR", "SVR", "senet", "senetMTL"]:
                if model in res_path[resolution].keys():
                    if model in ["senet", "senetMTL"]:
                        var_ref = var_ref_dl
                    else:
                        var_ref = var_ref_ml
                    var_pred = np.load(res_path[resolution][model])

                    # ------calculate some metrics
                    ref_mean = np.mean(var_ref)
                    pred_mean = np.mean(var_pred)
                    ref_std = np.std(var_ref, ddof=1)

                    r2 = metrics.r2_score(y_true=var_ref, y_pred=var_pred)

                    err = var_pred - var_ref
                    se = np.mean(err)
                    se_nsq = (se / ref_std) ** 2
                    mse_centered = np.mean((err - pred_mean + ref_mean)**2)
                    mse_centered_nsq = mse_centered / ref_std**2
                    logging = [resolution, name_mapping[model]] + list(map(lambda x: "%.4f" % x, [1.0-r2, mse_centered_nsq, se_nsq]))
                    print("\t".join(logging))

                    result_summary = {"Model": name_mapping[model],
                                      "Variable": var,
                                      "Resolution": resolution,
                                      "1-R^2": 1.0-r2,
                                      "nME^2": se_nsq,
                                      "nRMSE_centered^2": mse_centered_nsq}
                    report_df = report_df.append(result_summary, ignore_index=True)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
    col_mapping = {"height": 0, "footprint": 1}
    title_mapping = {"height": "$H_{\mathrm{ave}}$", "footprint": "$\lambda_p$"}
    color_mapping = {"RFR": "darkcyan", "XGBoostR": "navy", "(bagging)SVR": "darkmagenta",
                     "SENet-STL": "darkorange", "SENet-MTL": "firebrick"}

    bar_width = 0.10
    model_namelist = ["RFR", "XGBoostR", "(bagging)SVR", "SENet-STL", "SENet-MTL"]
    resolution_list = ["100 m", "250 m", "500 m", "1000 m"]
    num_model = len(model_namelist)
    num_resolution = len(resolution_list)

    bar_position_list = [np.arange(num_resolution) + bar_width * i for i in range(0, num_model)]

    for var in ["height", "footprint"]:
        col_id = col_mapping[var]
        legend_list = []
        for model_id in range(0, num_model):
            model_name = model_namelist[model_id]
            bar_pos = bar_position_list[model_id]

            model_test = report_df["Model"] == model_name
            var_test = report_df["Variable"] == var
            model_test = model_test & var_test
            se_nsq_list = []
            mse_centered_nsq_list = []
            for res in resolution_list:
                res_test = report_df["Resolution"] == res
                se_nsq_list.append(report_df[model_test & res_test]["nME^2"].values[0])
                mse_centered_nsq_list.append(report_df[model_test & res_test]["nRMSE_centered^2"].values[0])

            b1 = ax[col_id].bar(bar_pos, se_nsq_list, bar_width, color=color_mapping[model_name],
                                alpha=1.0, label=model_name+", $\mathrm{ME}_n^2$")
            b2 = ax[col_id].bar(bar_pos, mse_centered_nsq_list, bar_width, bottom=se_nsq_list,
                                color=color_mapping[model_name], alpha=0.5, label=model_name+", ${\mathrm{RMSE}'}_n^2$")
            legend_list.append(b1)
            legend_list.append(b2)

        ax[col_id].set_xticks(bar_position_list[num_model // 2])
        ax[col_id].set_xticklabels(resolution_list, size="large")
        ax[col_id].set_title(title_mapping[var], size="large")

        if col_id == 0:
            ax[col_id].set_ylabel("$1-R^2$", size="large")
            ax[col_id].set_ylim(0.0, 1.0)
            lgd = ax[col_id].legend(handles=legend_list, bbox_to_anchor=[-0.1, -0.15, 0.02, 0.02], ncol=num_model, loc="upper left", fontsize="x-small")
        else:
            ax[col_id].tick_params(axis="y", which="both", left=False, labelleft=False)

    plt.savefig("Fig-4-r2_partition.pdf", bbox_extra_artists=(lgd, ), bbox_inches='tight', dpi=600)


# Fig. 4. Comparison metrics varying with scale using Taylor Diagram
# ref to: https://cdat.llnl.gov/Jupyter-notebooks/vcs/Taylor_Diagrams/Taylor_Diagrams.html
def fig_4_metric_scale(res_prefix):
    res_summary = {}
    for var in ["height", "footprint"]:
        res_path = {
            "100m": {
                "DL_reference": os.path.join(res_prefix, var, "_".join([var, "100m", "refDL"]) + ".npy"),
                "ML_reference": os.path.join(res_prefix, var, "_".join([var, "100m", "ref"]) + ".npy"),
                "rf": os.path.join(res_prefix, var, "_".join(["rf", var, "100m"]) + ".npy"),
                "xgb": os.path.join(res_prefix, var, "_".join(["xgb", var, "100m"]) + ".npy"),
                "baggingSVR": os.path.join(res_prefix, var, "_".join(["baggingSVR", var, "100m"]) + ".npy"),
                "senet": os.path.join(res_prefix, var, "_".join(["senet", var, "100m"]) + ".npy"),
                "senetMTL": os.path.join(res_prefix, var, "_".join(["senetMTL", var, "100m"]) + ".npy"),
                "cbam": os.path.join(res_prefix, var, "_".join(["cbam", var, "100m"]) + ".npy"),
                "cbamMTL": os.path.join(res_prefix, var, "_".join(["cbamMTL", var, "100m"]) + ".npy")
            },
            "250m": {
                "DL_reference": os.path.join(res_prefix, var, "_".join([var, "250m", "refDL"]) + ".npy"),
                "ML_reference": os.path.join(res_prefix, var, "_".join([var, "250m", "ref"]) + ".npy"),
                "rf": os.path.join(res_prefix, var, "_".join(["rf", var, "250m"]) + ".npy"),
                "xgb": os.path.join(res_prefix, var, "_".join(["xgb", var, "250m"]) + ".npy"),
                "baggingSVR": os.path.join(res_prefix, var, "_".join(["baggingSVR", var, "250m"]) + ".npy"),
                "senet": os.path.join(res_prefix, var, "_".join(["senet", var, "250m"]) + ".npy"),
                "senetMTL": os.path.join(res_prefix, var, "_".join(["senetMTL", var, "250m"]) + ".npy"),
                "cbam": os.path.join(res_prefix, var, "_".join(["cbam", var, "250m"]) + ".npy"),
                "cbamMTL": os.path.join(res_prefix, var, "_".join(["cbamMTL", var, "250m"]) + ".npy")
            },
            "500m": {
                "DL_reference": os.path.join(res_prefix, var, "_".join([var, "500m", "refDL"]) + ".npy"),
                "ML_reference": os.path.join(res_prefix, var, "_".join([var, "500m", "ref"]) + ".npy"),
                "rf": os.path.join(res_prefix, var, "_".join(["rf", var, "500m"]) + ".npy"),
                "xgb": os.path.join(res_prefix, var, "_".join(["xgb", var, "500m"]) + ".npy"),
                "SVR": os.path.join(res_prefix, var, "_".join(["SVR", var, "500m"]) + ".npy"),
                "senet": os.path.join(res_prefix, var, "_".join(["senet", var, "500m"]) + ".npy"),
                "senetMTL": os.path.join(res_prefix, var, "_".join(["senetMTL", var, "500m"]) + ".npy"),
                "cbam": os.path.join(res_prefix, var, "_".join(["cbam", var, "500m"]) + ".npy"),
                "cbamMTL": os.path.join(res_prefix, var, "_".join(["cbamMTL", var, "500m"]) + ".npy"),
            },
            "1000m": {
                "DL_reference": os.path.join(res_prefix, var, "_".join([var, "1000m", "refDL"]) + ".npy"),
                "ML_reference": os.path.join(res_prefix, var, "_".join([var, "1000m", "ref"]) + ".npy"),
                "rf": os.path.join(res_prefix, var, "_".join(["rf", var, "1000m"]) + ".npy"),
                "xgb": os.path.join(res_prefix, var, "_".join(["xgb", var, "1000m"]) + ".npy"),
                "SVR": os.path.join(res_prefix, var, "_".join(["SVR", var, "1000m"]) + ".npy"),
                "senet": os.path.join(res_prefix, var, "_".join(["senet", var, "1000m"]) + ".npy"),
                "senetMTL": os.path.join(res_prefix, var, "_".join(["senetMTL", var, "1000m"]) + ".npy"),
                "cbam": os.path.join(res_prefix, var, "_".join(["cbam", var, "1000m"]) + ".npy"),
                "cbamMTL": os.path.join(res_prefix, var, "_".join(["cbamMTL", var, "1000m"]) + ".npy"),
            }
        }
        res_summary[var] = res_path

    metric_summary = {}
    for var in ["height", "footprint"]:
        metric_dict = {
            "100m": {"ref": None, "RFR": None, "XGBoostR": None, "(bagging)SVR": None,
                     "SENet-STL": None, "SENet-MTL": None,
                     #"CBAM-STL": None, "CBAM-MTL": None,
            },
            "250m": {"ref": None, "RFR": None, "XGBoostR": None, "(bagging)SVR": None,
                     "SENet-STL": None, "SENet-MTL": None,
                     #"CBAM-STL": None, "CBAM-MTL": None,
            },
            "500m": {"ref": None, "RFR": None, "XGBoostR": None, "(bagging)SVR": None,
                     "SENet-STL": None, "SENet-MTL": None,
                     #"CBAM-STL": None, "CBAM-MTL": None,
            },
            "1000m": {"ref": None, "RFR": None, "XGBoostR": None, "(bagging)SVR": None,
                      "SENet-STL": None, "SENet-MTL": None,
                      #"CBAM-STL": None, "CBAM-MTL": None,
            }
        }

        name_mapping = {"rf": "RFR", "xgb": "XGBoostR", "SVR": "(bagging)SVR", "baggingSVR": "(bagging)SVR",
                        "senet": "SENet-STL", "senetMTL": "SENet-MTL",
                        "cbam": "CBAM-STL", "cbamMTL": "CBAM-MTL"}

        # ------calculate std and R
        for model in ["rf", "xgb", "baggingSVR", "SVR", "senet", "senetMTL"]:
        #for model in ["rf", "xgb", "baggingSVR", "SVR", "cbam", "cbamMTL"]:
            for resolution in ["100m", "250m", "500m", "1000m"]:
                var_ref_dl = np.load(res_summary[var][resolution]["DL_reference"])
                var_ref_ml = np.load(res_summary[var][resolution]["ML_reference"])
                std_ref = np.std(var_ref_dl, ddof=1)
                metric_dict[resolution]["ref"] = {"std": std_ref, "R": 1.0}
                if model in res_summary[var][resolution].keys():
                    if model in ["senet", "senetMTL", "cbam", "cbamMTL"]:
                        var_ref = var_ref_dl
                    else:
                        var_ref = var_ref_ml
                    var_pred = np.load(res_summary[var][resolution][model])

                    r_pred = np.corrcoef(x=var_ref, y=var_pred)[0, 1]
                    print(model, resolution, r_pred)
                    std_pred = np.std(var_pred, ddof=1)
                    metric_dict[resolution][name_mapping[model]] = {"std": std_pred, "R": r_pred}

        metric_summary[var] = metric_dict

    model_style_dict = {
        "100m": {
            "RFR": {"color": "indianred", "symbol": "^", "facecolor": 'none'},
            "XGBoostR": {"color": "indianred", "symbol": "P", "facecolor": 'none'},
            "(bagging)SVR": {"color": "indianred", "symbol": "X", "facecolor": 'none'},
            "SENet-STL": {"color": "indianred", "symbol": "s", "facecolor": "indianred"},
            "SENet-MTL": {"color": "indianred", "symbol": "H", "facecolor": "indianred"},
            #"CBAM-STL": {"color": "indianred", "symbol": "s", "facecolor": "indianred"},
            #"CBAM-MTL": {"color": "indianred", "symbol": "H", "facecolor": "indianred"},
        },
        "250m": {
            "RFR": {"color": "darkorange", "symbol": "^", "facecolor": 'none'},
            "XGBoostR": {"color": "darkorange", "symbol": "P", "facecolor": 'none'},
            "(bagging)SVR": {"color": "darkorange", "symbol": "X", "facecolor": 'none'},
            "SENet-STL": {"color": "darkorange", "symbol": "s", "facecolor": "darkorange"},
            "SENet-MTL": {"color": "darkorange", "symbol": "H", "facecolor": "darkorange"},
            #"CBAM-STL": {"color": "darkorange", "symbol": "s", "facecolor": "darkorange"},
            #"CBAM-MTL": {"color": "darkorange", "symbol": "H", "facecolor": "darkorange"},
        },
        "500m": {
            "RFR": {"color": "darkgreen", "symbol": "^", "facecolor": 'none'},
            "XGBoostR": {"color": "darkgreen", "symbol": "P", "facecolor": 'none'},
            "(bagging)SVR": {"color": "darkgreen", "symbol": "X", "facecolor": 'none'},
            "SENet-STL": {"color": "darkgreen", "symbol": "s", "facecolor": "darkgreen"},
            "SENet-MTL": {"color": "darkgreen", "symbol": "H", "facecolor": "darkgreen"},
            #"CBAM-STL": {"color": "darkgreen", "symbol": "s", "facecolor": "darkgreen"},
            #"CBAM-MTL": {"color": "darkgreen", "symbol": "H", "facecolor": "darkgreen"},
        },
        "1000m": {
            "RFR": {"color": "darkmagenta", "symbol": "^", "facecolor": 'none'},
            "XGBoostR": {"color": "darkmagenta", "symbol": "P", "facecolor": 'none'},
            "(bagging)SVR": {"color": "darkmagenta", "symbol": "X", "facecolor": 'none'},
            "SENet-STL": {"color": "darkmagenta", "symbol": "s", "facecolor": "darkmagenta"},
            "SENet-MTL": {"color": "darkmagenta", "symbol": "H", "facecolor": "darkmagenta"},
            #"CBAM-STL": {"color": "darkmagenta", "symbol": "s", "facecolor": "darkmagenta"},
            #"CBAM-MTL": {"color": "darkmagenta", "symbol": "H", "facecolor": "darkmagenta"},
        }
    }

    fig = plt.figure(figsize=(8, 3.8))

    rect_mapping = {"height": 121, "footprint": 122}
    '''
    color_mapping = {"RFR": "indianred", "XGBoostR": "darkorange", "(bagging)SVR": "c",
                     "SENet-STL": "navy", "SENet-MTL": "darkmagenta"}
    marker_mapping = {"RFR": "s", "XGBoostR": "s", "(bagging)SVR": "s", "SENet-STL": "D", "SENet-MTL": "D"}
    '''
    std_mapping = {"height": {"min": 0.0, "max": 1.0}, "footprint": {"min": 0.0, "max": 1.0}}
    title_mapping = {"height": "$H_{\mathrm{ave}}$", "footprint": "$\lambda_p$"}

    for var in ["height", "footprint"]:
        rect_id = rect_mapping[var]
        dia = TaylorDiagram(std_ref=1.0, fig=fig, rect=rect_id,
                            label="reference", std_min=std_mapping[var]["min"], std_max=std_mapping[var]["max"],
                            std_label_format='%.1f', normalized=True)
        for resolution in ["100m", "250m", "500m", "1000m"]:
            for model in ["RFR", "XGBoostR", "(bagging)SVR", "SENet-STL", "SENet-MTL"]:
            #for model in ["RFR", "XGBoostR", "(bagging)SVR", "CBAM-STL", "CBAM-MTL"]:
                #print(resolution, model, metric_summary[var][resolution][model]["std"] / metric_summary[var][resolution]["ref"]["std"])
                dia.add_sample(stddev=metric_summary[var][resolution][model]["std"] / metric_summary[var][resolution]["ref"]["std"],
                               corrcoef=metric_summary[var][resolution][model]["R"],
                               ms=4, ls="", markeredgewidth=0.5, marker=model_style_dict[resolution][model]["symbol"],
                               markerfacecolor=model_style_dict[resolution][model]["facecolor"],
                               markeredgecolor=model_style_dict[resolution][model]["color"],
                               label="{0}, {1}".format(model, resolution))
        dia.add_grid()
        contours = dia.add_contours(colors='0.5', levels=5)
        plt.clabel(contours, inline=1, fontsize="medium", fmt='%.2f')
        dia._ax.set_title(title_mapping[var], size="large")

    ax_legend = fig.axes[1]
    lgd = ax_legend.legend(handles=dia.samplePoints, bbox_to_anchor=[1.05, 0.8], ncol=2, loc="upper left", fontsize="x-small")

    plt.savefig("Fig-4-taylor_diagram.pdf", bbox_inches='tight', dpi=600)


# Fig. 4. Comparison training and validation curve between STL and MTL
# i.e., [training, validation] x [100m, 250m, 500m, 1000m]
def fig_4_curve_STL_MTL(record_prefix):
    backbone = "senet"
    record_path = {
        "height": {
            "100m": {
                "STL": os.path.join(record_prefix, "height", "check_pt_{0}_100m".format(backbone), "experiment_5", "out_{0}.txt".format(backbone)),
                "MTL": os.path.join(record_prefix, "height", "check_pt_{0}_100m_MTL".format(backbone), "experiment_11", "out_{0}_MTL.txt".format(backbone)),
            },
            "250m": {
                "STL": os.path.join(record_prefix, "height", "check_pt_{0}_250m".format(backbone), "experiment_5", "out_{0}.txt".format(backbone)),
                "MTL": os.path.join(record_prefix, "height", "check_pt_{0}_250m_MTL".format(backbone), "experiment_11", "out_{0}_MTL.txt".format(backbone)),
            },
            "500m": {
                "STL": os.path.join(record_prefix, "height", "check_pt_{0}_500m".format(backbone), "experiment_5", "out_{0}.txt".format(backbone)),
                "MTL": os.path.join(record_prefix, "height", "check_pt_{0}_500m_MTL".format(backbone), "experiment_11", "out_{0}_MTL.txt".format(backbone)),
            },
            "1000m": {
                "STL": os.path.join(record_prefix, "height", "check_pt_{0}_1000m".format(backbone), "experiment_5", "out_{0}.txt".format(backbone)),
                "MTL": os.path.join(record_prefix, "height", "check_pt_{0}_1000m_MTL".format(backbone), "experiment_11", "out_{0}_MTL.txt".format(backbone)),
            }
        },
        "footprint": {
            "100m": {
                "STL": os.path.join(record_prefix, "footprint", "check_pt_{0}_100m".format(backbone), "experiment_5", "out_{0}.txt".format(backbone)),
                "MTL": os.path.join(record_prefix, "footprint", "check_pt_{0}_100m_MTL".format(backbone), "experiment_11", "out_{0}_MTL.txt".format(backbone)),
            },
            "250m": {
                "STL": os.path.join(record_prefix, "footprint", "check_pt_{0}_250m".format(backbone), "experiment_5", "out_{0}.txt".format(backbone)),
                "MTL": os.path.join(record_prefix, "footprint", "check_pt_{0}_250m_MTL".format(backbone), "experiment_11", "out_{0}_MTL.txt".format(backbone)),
            },
            "500m": {
                "STL": os.path.join(record_prefix, "footprint", "check_pt_{0}_500m".format(backbone), "experiment_5", "out_{0}.txt".format(backbone)),
                "MTL": os.path.join(record_prefix, "footprint", "check_pt_{0}_500m_MTL".format(backbone), "experiment_11", "out_{0}_MTL.txt".format(backbone)),
            },
            "1000m": {
                "STL": os.path.join(record_prefix, "footprint", "check_pt_{0}_1000m".format(backbone), "experiment_5", "out_{0}.txt".format(backbone)),
                "MTL": os.path.join(record_prefix, "footprint", "check_pt_{0}_1000m_MTL".format(backbone), "experiment_11", "out_{0}_MTL.txt".format(backbone)),
            }
        }
    }

    row_mapping = {"height": 0, "footprint": 1}
    row_label_mapping = {0: "$(H_{\mathrm{ave}})$", 1: "$(\lambda_p)$"}
    col_mapping = {"100m": 0, "250m": 1, "500m": 2, "1000m": 3}
    col_label_mapping = {0: "100m", 1: "250m", 2: "500m", 3: "1000m"}
    var_name_mapping = {"height": "BuildingHeight", "footprint": "BuildingFootprint"}

    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(8, 4))
    epoch = np.linspace(0, 99, 100, dtype=int)
    y_label = np.linspace(0.0, 1.0, 6)
    x_label = np.linspace(0, 100, 6, dtype=int)
    metric = "$R^2$"

    for var in ["height", "footprint"]:
        row_id = row_mapping[var]
        for resolution in ["100m", "250m", "500m", "1000m"]:
            print(record_path[var][resolution]["STL"])
            res_stl = extract_record(output_file=record_path[var][resolution]["STL"])
            r2_stl_training = res_stl["Training"]["R^2"]
            r2_stl_test = res_stl["Validation"]["R^2"]
            print(record_path[var][resolution]["MTL"])
            res_mtl = extract_record_MTL(output_file=record_path[var][resolution]["MTL"])
            r2_mtl_training = res_mtl[var_name_mapping[var]]["Training"]["R^2"]
            r2_mtl_test = res_mtl[var_name_mapping[var]]["Validation"]["R^2"]
            num_epoch = min(len(epoch), len(r2_stl_training), len(r2_stl_test), len(r2_mtl_training), len(r2_mtl_test))
            col_id = col_mapping[resolution]
            l1, = ax[row_id, col_id].plot(epoch[0:num_epoch], r2_stl_training[0:num_epoch], label="STL, training", color="royalblue", linestyle="dashed")
            l2, = ax[row_id, col_id].plot(epoch[0:num_epoch], r2_mtl_training[0:num_epoch], label="MTL, training", color="indianred", linestyle="dashed")
            l3, = ax[row_id, col_id].plot(epoch[0:num_epoch], r2_stl_test[0:num_epoch], label="STL, validation", color="navy")
            l4, = ax[row_id, col_id].plot(epoch[0:num_epoch], r2_mtl_test[0:num_epoch], label="MTL, validation", color="firebrick")

            ax[row_id, col_id].set_ylim(0.0, 1.0)

            if row_id == 0:
                ax[row_id, col_id].set_title("{0}".format(col_label_mapping[col_id]), size="large")
            if row_id == 1:
                ax[row_id, col_id].set_xlabel("Epoch")
                ax[row_id, col_id].set_xticks(x_label)
                ax[row_id, col_id].set_xticklabels([str(l) for l in x_label])
            else:
                ax[row_id, col_id].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            if col_id == 0:
                ax[row_id, col_id].set_ylabel("{0} {1}".format(metric, row_label_mapping[row_id]), size="large")
                ax[row_id, col_id].set_yticks(y_label)
                ax[row_id, col_id].set_yticklabels(["%.1f" % m for m in y_label])
            else:
                ax[row_id, col_id].tick_params(axis="y", which="both", left=False, labelleft=False)

            if row_id == 1 and col_id == 1:
                ax[row_id, col_id].legend(handles=[l1, l2, l3, l4], bbox_to_anchor=[-1.35, -0.55, 2.0, 0.04],
                                          ncol=4, loc="lower left")

    plt.savefig("Fig-4-curve.pdf", bbox_inches='tight')


# Fig. 5. 1km Building Height close-ups
# i.e., [Glasgow, Beijing, Chicago] x [Sentinel-2, ref_data, SENet-STL, SENet-MTL, Li’s result]
# ---extent for plotting: xmin ymin xmax ymax
# [1] Glasgow: -4.380 55.760 -4.100 55.930 (old)
# [1] Glasgow: -4.4786 55.6756 -3.8832 56.0197 (new)
# [2] Beijing: 116.204 39.823 116.575 40.038
# [3] LosAngeles: -118.6811 33.6742 -117.7362 34.3498
# [4] Chicago: -87.740 41.733 -87.545 41.996
# gdalwarp -dstnodata -255 -wo CUTLINE_ALL_TOUCHED=TRUE -cutline ../boundary/Chicago_bd.shp src.tif target.tif
def fig_5_model_close_up(var="height"):
    s2_prefix = "testCase"
    scale = 0.0001
    backbone = "senet"
    model_name = "SENet"

    city_shp = {
        "Glasgow": os.path.join(s2_prefix, "infer_test_Glasgow", "raw_data", "Glasgow_2020_sentinel_2_clip.tif"),
        "Beijing": os.path.join(s2_prefix, "infer_test_Beijing", "raw_data", "BeijingC6_2020_sentinel_2_50pt_clip.tif"),
        #"LosAngeles": os.path.join(s2_prefix, "infer_test_LosAngeles", "raw_data", "LosAngeles_2018_sentinel_2_clip.tif"),
        "Chicago": os.path.join(s2_prefix, "infer_test_Chicago", "raw_data", "Chicago_2018_sentinel_2_50pt_clip.tif")
    }

    li_prefix = "testCase"
    li_res = {
        "Glasgow": os.path.join(li_prefix, "infer_test_Glasgow", "1000m", "Glasgow_{0}_li2020_clip.tif".format(var)),
        "Beijing": os.path.join(li_prefix, "infer_test_Beijing", "1000m", "BeijingC6_{0}_li2020_clip.tif".format(var)),
        #"LosAngeles": os.path.join(li_prefix, "infer_test_LosAngeles", "1000m", "LosAngeles_{0}_li2020_clip.tif".format(var)),
        "Chicago": os.path.join(li_prefix, "infer_test_Chicago", "1000m", "Chicago_{0}_li2020_clip.tif".format(var))
    }

    senet_prefix = "testCase"
    res = {
        "Glasgow": os.path.join(senet_prefix, "infer_test_Glasgow", "1000m", "Glasgow_{0}_{1}.tif".format(var, backbone)),
        "Beijing": os.path.join(senet_prefix, "infer_test_Beijing", "1000m", "Beijing_{0}_{1}.tif".format(var, backbone)),
        #"LosAngeles": os.path.join(senet_prefix, "infer_test_LosAngeles", "1000m", "LosAngeles_{0}_senet.tif".format(var)),
        "Chicago": os.path.join(senet_prefix, "infer_test_Chicago", "1000m", "Chicago_{0}_{1}_clip.tif".format(var, backbone))
    }

    senetMTL_prefix = "testCase"
    res_MTL = {
        "Glasgow": os.path.join(senetMTL_prefix, "infer_test_Glasgow", "1000m", "Glasgow_{0}_{1}_MTL.tif".format(var, backbone)),
        "Beijing": os.path.join(senetMTL_prefix, "infer_test_Beijing", "1000m", "Beijing_{0}_{1}_MTL.tif".format(var, backbone)),
        #"LosAngeles": os.path.join(senetMTL_prefix, "infer_test_LosAngeles", "1000m", "LosAngeles_{0}_senet_MTL.tif".format(var)),
        "Chicago": os.path.join(senetMTL_prefix, "infer_test_Chicago", "1000m", "Chicago_{0}_{1}_MTL_clip.tif".format(var, backbone))
    }

    ref_prefix = "testCase"
    ref_data = {
        "Glasgow": os.path.join(ref_prefix, "infer_test_Glasgow", "1000m", "Glasgow_2020_building_{0}_clip.tif".format(var)),
        "Beijing": os.path.join(ref_prefix, "infer_test_Beijing", "1000m", "Beijing_2020_building_{0}_ref_clip.tif".format(var)),
        #"LosAngeles": os.path.join(ref_prefix, "infer_test_LosAngeles", "1000m", "LosAngeles_2017_building_{0}_clip.tif".format(var)),
        "Chicago": os.path.join(ref_prefix, "infer_test_Chicago", "1000m", "Chicago_2015_building_{0}_clip.tif".format(var))
    }

    if var == "height":
        ref_label = "$H_{\mathrm{ave}}$ [m]"
        var_limit = {
            "Glasgow": [0.0, 30.0],
            "Beijing": [0.0, 60.0],
            #"LosAngeles": [0.0, 50.0],
            "Chicago": [0.0, 60.0]
        }
        y_label = [1.16, 1.15, 1.08]
        pad_label = [-20, -20, -20]
    elif var == "footprint":
        ref_label = "$\lambda_p$ [$\mathrm{m}^2$/$\mathrm{m}^2$]"
        var_limit = {
            "Glasgow": [0.0, 0.5],
            "Beijing": [0.0, 0.5],
            #"LosAngeles": [0.0, 0.5],
            "Chicago": [0.0, 0.5]
        }
        y_label = [1.17, 1.18, 1.08]
        pad_label = [-20, -20, -20]
    else:
        raise NotImplementedError("Unknown target variable")

    city_list = ["Glasgow", "Beijing", "Chicago"]
    #city_list = ["Glasgow", "Beijing", "LosAngeles", "Chicago"]

    n_city = len(city_list)
    res_list = [city_shp, ref_data, res, res_MTL, li_res]
    n_model = len(res_list)

    widths = [2, 2, 2, 2, 2]
    heights = [1.5, 1.5, 3.5]

    gs_kw = dict(width_ratios=widths, height_ratios=heights)
    fig, ax = plt.subplots(nrows=n_city, ncols=n_model, gridspec_kw=gs_kw, figsize=(16, 7))
    for res_id in range(0, n_model):
        for city_id in range(0, n_city):
            city_name = city_list[city_id]
            res_file = res_list[res_id][city_name]
            image = rasterio.open(res_file)
            image_nodata = image.nodata
            image_bound = list(image.bounds)
            # ------Sentinel-2 data
            if res_id == 0:
                r_band = np.expand_dims(rgb_rescale_band(image.read(1)), axis=0)
                g_band = np.expand_dims(rgb_rescale_band(image.read(2)), axis=0)
                b_band = np.expand_dims(rgb_rescale_band(image.read(3)), axis=0)
                '''
                if np.isnan(r_band).any():
                    test = np.where(np.isnan(r_band))
                    r_band = fill_nan_nearest(r_band)

                if np.isnan(g_band).any():
                    test = np.where(np.isnan(g_band))
                    g_band = fill_nan_nearest(g_band)

                if np.isnan(b_band).any():
                    test = np.where(np.isnan(b_band))
                    b_band = fill_nan_nearest(b_band)

                r_band = np.expand_dims(r_band, axis=0) * scale
                g_band = np.expand_dims(g_band, axis=0) * scale
                b_band = np.expand_dims(b_band, axis=0) * scale
                '''
                image_dta = np.concatenate((r_band, g_band, b_band), axis=0)
                rplt.show(image_dta, transform=image.transform, ax=ax[city_id, res_id])
                if city_id == n_city - 1:
                    labels_loc = [-87.70, -87.60]
                    ax[city_id, res_id].set_xticks(labels_loc)
                    ax[city_id, res_id].set_xticklabels([str(l) for l in labels_loc])
            # ------model prediction data
            else:
                h_min, h_max = var_limit[city_name]
                image_dta = image.read(1)
                image_dta = np.where(image_dta == image_nodata, np.nan, image_dta)
                image_dta = np.where(image_dta == 0.0, np.nan, image_dta)
                im = ax[city_id, res_id].imshow(image_dta, cmap='rainbow', vmin=h_min, vmax=h_max,
                                                extent=[image_bound[0], image_bound[2], image_bound[1], image_bound[3]])
                rplt.show(image_dta, transform=image.transform, ax=ax[city_id, res_id], cmap="rainbow",
                          vmin=h_min, vmax=h_max)

                if res_id == n_model - 1:
                    position = ax[city_id, res_id].inset_axes([1.05, 0.02, 0.03, 0.96], transform=ax[city_id, res_id].transAxes)
                    cbar = fig.colorbar(im, ax=ax[city_id, res_id], cax=position)
                    cbar.ax.set_ylabel(ref_label, rotation=0,
                                       y=y_label[city_id], labelpad=pad_label[city_id], fontsize="small")

                if city_id == n_city - 1:
                    labels_loc = [-87.70, -87.60]
                    ax[city_id, res_id].set_xticks(labels_loc)
                    ax[city_id, res_id].set_xticklabels([str(l) for l in labels_loc])

    col_label = ["Sentinel-2 R/G/B", "Reference", model_name + "-STL", model_name + "-MTL", "L20"]
    row_label = city_list

    for a, col in zip(ax[0], col_label):
        a.set_title(col, size='large')

    for a, row in zip(ax[:, 0], row_label):
        a.set_ylabel(row, size='large', rotation=90)

    # plt.show()
    plt.savefig("Fig-5-close_ups_{0}.pdf".format(var), bbox_inches='tight', dpi=1000)


# Fig. 6. Compare the building height results in Glasgow from different models
# i.e., [100m, 250m, 500m, 1000m] x [ref_data, SENet-STL, SENet-MTL]
# ---extent for plotting: xmin ymin xmax ymax
# [1] Glasgow, 100m: -4.4786 55.6756 -3.8832 56.0197
def fig_6_Glasgow_compare(ref_prefix, res_prefix, var="height"):
    backbone = "senet"
    model_name = "SENet"

    if var == "height":
        linthresh = 10.0
        var_min = 0.0
        var_max = 30.0
        ref_label = "$H_{\mathrm{ave}}$ [m]"
        pad_label = 25
        p_max = 0.3
        err_max = 50.0
    elif var == "footprint":
        linthresh = 0.1
        var_min = 0.0
        var_max = 0.5
        ref_label = "$\lambda_p$ [$\mathrm{m}^2$/$\mathrm{m}^2$]"
        pad_label = 35
        p_max = 0.2
        err_max = 0.5
    else:
        raise NotImplementedError("Unknown target variable")

    res_path = {
        "100m": {
            "reference": os.path.join(ref_prefix, "100m", "Glasgow_2020_building_{0}_clip_th.tif".format(var)),
            backbone: os.path.join(res_prefix, "100m", "Glasgow_{0}_{1}_th.tif".format(var, backbone)),
            backbone + "MTL": os.path.join(res_prefix, "100m", "Glasgow_{0}_{1}_MTL_th.tif".format(var, backbone))
        },
        "250m": {
            "reference": os.path.join(ref_prefix, "250m", "Glasgow_2020_building_{0}_clip_th.tif".format(var)),
            backbone: os.path.join(res_prefix, "250m", "Glasgow_{0}_{1}_th.tif".format(var, backbone)),
            backbone + "MTL": os.path.join(res_prefix, "250m", "Glasgow_{0}_{1}_MTL_th.tif".format(var, backbone))
        },
        "500m": {
            "reference": os.path.join(ref_prefix, "500m", "Glasgow_2020_building_{0}_clip_th.tif".format(var)),
            backbone: os.path.join(res_prefix, "500m", "Glasgow_{0}_{1}_th.tif".format(var, backbone)),
            backbone + "MTL": os.path.join(res_prefix, "500m", "Glasgow_{0}_{1}_MTL_th.tif".format(var, backbone))
        },
        "1000m": {
            "reference": os.path.join(ref_prefix, "1000m", "Glasgow_2020_building_{0}_clip_th.tif".format(var)),
            backbone: os.path.join(res_prefix, "1000m", "Glasgow_{0}_{1}_th.tif".format(var, backbone)),
            backbone + "MTL": os.path.join(res_prefix, "1000m", "Glasgow_{0}_{1}_MTL_th.tif".format(var, backbone))
        }
    }

    res_dta = {
        "100m": {
            "reference": None, backbone: None, backbone + "MTL": None
        },
        "250m": {
            "reference": None, backbone: None, backbone + "MTL": None
        },
        "500m": {
            "reference": None, backbone: None, backbone + "MTL": None
        },
        "1000m": {
            "reference": None, backbone: None, backbone + "MTL": None
        }
    }

    fig = plt.figure(figsize=(16, 9))

    num_rows = 4
    num_cols = 4

    num_spaces = 1
    row_height = 4
    space_height = 1

    grid = (row_height * num_rows + space_height * num_spaces, num_cols)

    ax = []
    space_id = 0
    for row_id in range(0, num_rows):
        ax.append([])
        if row_id == num_rows - 1:
            space_id += 1
        for col_id in range(0, num_cols):
            grid_row = row_height * row_id + space_height * space_id
            grid_col = col_id
            ax[row_id].append(plt.subplot2grid(grid, (grid_row, grid_col), rowspan=row_height, fig=fig))

    col_mapping = {"100m": 0, "250m": 1, "500m": 2, "1000m": 3}
    row_mapping = {"reference": 0, backbone: 1, backbone + "MTL": 2}
    subplot_id_mapping = {
        0: {0: "(a)", 1: "(b)", 2: "(c)", 3: "(d)"},
        1: {0: "(e)", 1: "(f)", 2: "(g)", 3: "(h)"},
        2: {0: "(i)", 1: "(j)", 2: "(k)", 3: "(l)"},
    }

    for model in ["reference", backbone, backbone + "MTL"]:
        row_id = row_mapping[model]
        for res in ["100m", "250m", "500m", "1000m"]:
            col_id = col_mapping[res]
            image = rasterio.open(res_path[res][model])
            image_nodata = image.nodata
            image_bound = list(image.bounds)
            # ------plot prediction result
            image_dta = image.read(1)
            image_dta = np.where(image_dta == image_nodata, np.nan, image_dta)
            image_dta = np.where(image_dta == 0.0, np.nan, image_dta)
            im = ax[row_id][col_id].imshow(image_dta, cmap='rainbow', vmin=var_min, vmax=var_max,
                                            extent=[image_bound[0], image_bound[2], image_bound[1], image_bound[3]])
            rplt.show(image_dta, transform=image.transform, ax=ax[row_id][col_id], cmap="rainbow",
                      vmin=var_min, vmax=var_max)
            # ------save result data
            res_dta[res][model] = np.where(np.isnan(image_dta), 0.0, image_dta)

            subplot_id = subplot_id_mapping[row_id][col_id]
            ax[row_id][col_id].text(0.03, 0.95, subplot_id, ha="left", va="top",
                                    fontsize="x-small", transform=ax[row_id][col_id].transAxes)

            if row_id != 2:
                ax[row_id][col_id].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            else:
                ax[row_id][col_id].set_xlabel("Longitude")

            if col_id != 0:
                ax[row_id][col_id].tick_params(axis="y", which="both", left=False, labelleft=False)
            else:
                ax[row_id][col_id].set_ylabel("Latitude")

    col_label = ["100m", "250m", "500m", "1000m"]

    for a, col in zip(ax[0], col_label):
        a.set_title(col, size='large')

    # ---prepare a DataFrame for violin plot
    palette_model = {model_name + "-STL": "dodgerblue", model_name + "-MTL": "salmon"}
    name_mapping = {backbone: model_name + "-STL", backbone + "MTL": model_name + "-MTL"}
    report_df = pd.DataFrame(columns=["Model", "Resolution", "RMSE", "MAE", "NMAD", "BIAS-min", "BIAS-max",
                                      "R^2", "nME^2", "nRMSE_centered^2", "CC", "SSIM"])

    for res in ["100m", "250m", "500m", "1000m"]:
        for model in [backbone, backbone + "MTL"]:
            col_id = col_mapping[res]
            height = min(res_dta[res][model].shape[0], res_dta[res]["reference"].shape[0])
            width = min(res_dta[res][model].shape[1], res_dta[res]["reference"].shape[1])
            print(height, width)
            img_pred_clip = res_dta[res][model][0:height, 0:width]
            img_ref_clip = res_dta[res]["reference"][0:height, 0:width]

            # ------applying the mask derived from prediction
            img_ref_clip_1d = img_ref_clip.flatten()
            img_pred_clip_1d = img_pred_clip.flatten()

            model_error = img_pred_clip_1d - img_ref_clip_1d

            pt_id_select = np.logical_and(img_ref_clip_1d != 0, img_pred_clip_1d != 0)
            model_error_tmp = model_error[pt_id_select]

            np.save("tmp/{0}/{1}_{2}_ref.npy".format(var, model, res), img_ref_clip_1d[pt_id_select])
            np.save("tmp/{0}/{1}_{2}_pred.npy".format(var, model, res), img_pred_clip_1d[pt_id_select])

            # ------fill the report DataFrame
            ref_mean = np.mean(img_ref_clip_1d[pt_id_select])
            ref_std = np.std(img_ref_clip_1d[pt_id_select], ddof=1)
            pred_mean = np.mean(img_pred_clip_1d[pt_id_select])

            result_summary = {"Model": name_mapping[model],
                              "Resolution": res,
                              "RMSE": np.sqrt(np.mean(model_error_tmp**2)),
                              "MAE": np.mean(np.abs(model_error_tmp)),
                              "NMAD": get_NMAD(model_error_tmp),
                              "BIAS-min": np.min(model_error_tmp),
                              "BIAS-max": np.max(model_error_tmp),
                              "R^2": metrics.r2_score(y_true=img_ref_clip_1d[pt_id_select],
                                                      y_pred=img_pred_clip_1d[pt_id_select]),
                              "nME^2": (np.mean(model_error_tmp) / ref_std) ** 2,
                              "nRMSE_centered^2": np.mean((model_error_tmp - pred_mean + ref_mean)**2) / ref_std**2,
                              "CC": np.corrcoef(img_ref_clip_1d[pt_id_select], img_pred_clip_1d[pt_id_select])[0, 1],
                              "SSIM": ssim(im1=img_pred_clip, im2=img_ref_clip)}
            report_df = report_df.append(result_summary, ignore_index=True)

            n_sample = len(model_error_tmp)

            resolution_list = [res] * n_sample
            if model == backbone:
                mtl_list = [model_name + "-STL"] * n_sample
            else:
                mtl_list = [model_name + "-MTL"] * n_sample

            if var == "height":
                bias_col = "$\mathrm{bias}_{H_{\mathrm{ave}}}$"
                err_df = pd.DataFrame({"resolution": resolution_list, bias_col: model_error_tmp, "learning strategy": mtl_list})
                g = sns.histplot(x=bias_col, hue="learning strategy", data=err_df, legend=False,
                                 ax=ax[-1][col_id], bins=100, binrange=(-err_max, err_max), stat="probability", palette=palette_model, element="step")
            elif var == "footprint":
                bias_col = "$\mathrm{bias}_{\lambda_p}$"
                err_df = pd.DataFrame({"resolution": resolution_list, bias_col: model_error_tmp, "learning strategy": mtl_list})
                g = sns.histplot(x=bias_col, hue="learning strategy", data=err_df, legend=False,
                                 ax=ax[-1][col_id], bins=100, binrange=(-err_max, err_max), stat="probability", palette=palette_model, element="step")
            else:
                raise NotImplementedError("Unknown target variable")

            if col_id != 0:
                g.set(ylabel=None)
                ax[-1][col_id].tick_params(axis="y", which="both", left=False, labelleft=False)
            else:
                ax[-1][col_id].set_ylabel("Prob", size="large")

            ax[-1][col_id].set_xscale('symlog', linthresh=linthresh)
            ax[-1][col_id].set_ylim(0, p_max)

    position = ax[3][1].inset_axes([0.0, -0.4, 1.0, 0.04], transform=ax[3][1].transAxes)
    cbar = fig.colorbar(im, ax=ax[3][1], cax=position, orientation="horizontal")
    cbar.ax.tick_params(labelsize="medium")
    cbar.ax.set_ylabel(ref_label, labelpad=pad_label, y=-1.2, rotation=0)

    legend_list = []
    for model in [model_name + "-STL", model_name + "-MTL"]:
        legend_list.append(Patch(facecolor=palette_model[model], edgecolor=palette_model[model],
                                 label=model, alpha=0.5))

    ax[-1][2].legend(handles=legend_list, title="learning strategy", bbox_to_anchor=[0.0, -0.63], fontsize='medium', loc='lower left', ncol=2)

    report_df.to_csv("Fig-6-Glasgow-{0}.csv".format(var), index=False)

    plt.savefig("Fig-6-Glasgow-box_{0}.pdf".format(var), bbox_inches='tight', dpi=1000)


# Fig. 6. Compare the building height results in Los Angeles from different models
# i.e., [100m, 250m, 500m, 1000m] x [ref_data, SENet-STL, SENet-MTL]
# ---extent for plotting: xmin ymin xmax ymax
# [1] LosAngeles, 100m: -118.6811 33.6742 -117.7362 34.3498
def fig_6_LosAngeles_compare(ref_prefix, res_prefix, var="height"):
    if var == "height":
        linthresh = 10.0
        var_min = 0.0
        var_max = 50.0
    elif var == "footprint":
        linthresh = 0.2
        var_min = 0.0
        var_max = 1.0
    else:
        raise NotImplementedError("Unknown target variable")

    res_path = {
        "100m": {
            "reference": os.path.join(ref_prefix, "100m", "LosAngeles_2017_building_{0}_clip.tif".format(var)),
            "senet": os.path.join(res_prefix, "100m", "LosAngeles_{0}_senet.tif".format(var)),
            "senetMTL": os.path.join(res_prefix, "100m", "LosAngeles_{0}_senet_MTL.tif".format(var))
        },
        "250m": {
            "reference": os.path.join(ref_prefix, "250m", "LosAngeles_2017_building_{0}_clip.tif".format(var)),
            "senet": os.path.join(res_prefix, "250m", "LosAngeles_{0}_senet.tif".format(var)),
            "senetMTL": os.path.join(res_prefix, "250m", "LosAngeles_{0}_senet_MTL.tif".format(var))
        },
        "500m": {
            "reference": os.path.join(ref_prefix, "500m", "LosAngeles_2017_building_{0}_clip.tif".format(var)),
            "senet": os.path.join(res_prefix, "500m", "LosAngeles_{0}_senet.tif".format(var)),
            "senetMTL": os.path.join(res_prefix, "500m", "LosAngeles_{0}_senet_MTL.tif".format(var))
        },
        "1000m": {
            "reference": os.path.join(ref_prefix, "1000m", "LosAngeles_2017_building_{0}_clip.tif".format(var)),
            "senet": os.path.join(res_prefix, "1000m", "LosAngeles_{0}_senet.tif".format(var)),
            "senetMTL": os.path.join(res_prefix, "1000m", "LosAngeles_{0}_senet_MTL.tif".format(var))
        }
    }

    res_dta = {
        "100m": {
            "reference": None, "senet": None, "senetMTL": None
        },
        "250m": {
            "reference": None, "senet": None, "senetMTL": None
        },
        "500m": {
            "reference": None, "senet": None, "senetMTL": None
        },
        "1000m": {
            "reference": None, "senet": None, "senetMTL": None
        }
    }

    fig, ax = plt.subplots(nrows=4, ncols=4, constrained_layout=True, figsize=(20, 10))
    # ------the last row used for error plotting
    gs = ax[3, 0].get_gridspec()
    for sa in ax[3, :]:
        sa.remove()
    ax_error = fig.add_subplot(gs[3, :])

    col_mapping = {"100m": 0, "250m": 1, "500m": 2, "1000m": 3}
    row_mapping = {"reference": 0, "senet": 1, "senetMTL": 2}

    for model in ["reference", "senet", "senetMTL"]:
        row_id = row_mapping[model]
        for res in ["100m", "250m", "500m", "1000m"]:
            col_id = col_mapping[res]
            image = rasterio.open(res_path[res][model])
            image_nodata = image.nodata
            image_bound = list(image.bounds)
            # ------plot prediction result
            image_dta = image.read(1)
            image_dta = np.where(image_dta == image_nodata, np.nan, image_dta)
            image_dta = np.where(image_dta == 0.0, np.nan, image_dta)
            im = ax[row_id, col_id].imshow(image_dta, cmap='rainbow', vmin=var_min, vmax=var_max,
                                            extent=[image_bound[0], image_bound[2], image_bound[1], image_bound[3]])
            rplt.show(image_dta, transform=image.transform, ax=ax[row_id, col_id], cmap="rainbow",
                      vmin=var_min, vmax=var_max)
            position = ax[row_id, col_id].inset_axes([1.05, 0.02, 0.03, 0.96], transform=ax[row_id, col_id].transAxes)
            fig.colorbar(im, ax=ax[row_id, col_id], cax=position)
            # ------save result data
            res_dta[res][model] = np.where(np.isnan(image_dta), 0.0, image_dta)

            if row_id != 2:
                ax[row_id, col_id].tick_params(axis="x", which="both", bottom=False, labelbottom=False)

            if col_id != 0:
                ax[row_id, col_id].tick_params(axis="y", which="both", left=False, labelleft=False)

    col_label = ["100m", "250m", "500m", "1000m"]
    row_label = ["reference", "SENet-STL", "SENet-MTL"]

    for a, col in zip(ax[0], col_label):
        a.set_title(col, size='xx-large')

    for a, row in zip(ax[:-1, 0], row_label):
        a.set_ylabel(row, size='xx-large', rotation=90)

    # ---prepare a DataFrame for violin plot
    error_list = []
    resolution_list = []
    mtl_list = []

    name_mapping = {"senet": "SENet-STL", "senetMTL": "SENet-MTL"}
    report_df = pd.DataFrame(columns=["Model", "Resolution", "RMSE", "MAE", "NMAD", "BIAS-min", "BIAS-max",
                                      "R^2", "nME^2", "nRMSE_centered^2", "CC", "SSIM"])

    for res in ["100m", "250m", "500m", "1000m"]:
        for model in ["senet", "senetMTL"]:
            height = min(res_dta[res][model].shape[0], res_dta[res]["reference"].shape[0])
            width = min(res_dta[res][model].shape[1], res_dta[res]["reference"].shape[1])
            print(height, width)
            img_pred_clip = res_dta[res][model][0:height, 0:width]
            img_ref_clip = res_dta[res]["reference"][0:height, 0:width]

            # ------applying the mask derived from prediction
            img_ref_clip_1d = img_ref_clip.flatten()
            img_pred_clip_1d = img_pred_clip.flatten()

            model_error = img_pred_clip_1d - img_ref_clip_1d

            pt_id_select = (img_ref_clip_1d != 0)
            model_error_tmp = model_error[pt_id_select]

            np.save("tmp/{0}/{1}_{2}_ref.npy".format(var, model, res), img_ref_clip_1d[pt_id_select])
            np.save("tmp/{0}/{1}_{2}_pred.npy".format(var, model, res), img_pred_clip_1d[pt_id_select])

            # ------fill the report DataFrame
            ref_mean = np.mean(img_ref_clip_1d[pt_id_select])
            ref_std = np.std(img_ref_clip_1d[pt_id_select], ddof=1)
            pred_mean = np.mean(img_pred_clip_1d[pt_id_select])

            result_summary = {"Model": name_mapping[model],
                              "Resolution": res,
                              "RMSE": np.sqrt(np.mean(model_error_tmp**2)),
                              "MAE": np.mean(np.abs(model_error_tmp)),
                              "NMAD": get_NMAD(model_error_tmp),
                              "BIAS-min": np.min(model_error_tmp),
                              "BIAS-max": np.max(model_error_tmp),
                              "R^2": metrics.r2_score(y_true=img_ref_clip_1d[pt_id_select],
                                                      y_pred=img_pred_clip_1d[pt_id_select]),
                              "nME^2": (np.mean(model_error_tmp) / ref_std) ** 2,
                              "nRMSE_centered^2": np.mean((model_error_tmp - pred_mean + ref_mean)**2) / ref_std**2,
                              "CC": np.corrcoef(img_ref_clip_1d[pt_id_select], img_pred_clip_1d[pt_id_select])[0, 1],
                              "SSIM": ssim(im1=img_pred_clip, im2=img_ref_clip)}
            report_df = report_df.append(result_summary, ignore_index=True)

            error_list += model_error.tolist()
            n_sample = len(model_error)
            resolution_list += [res] * n_sample
            if model == "senet":
                mtl_list += ["STL"] * n_sample
            else:
                mtl_list += ["MTL"] * n_sample

    if var == "height":
        err_df = pd.DataFrame({"resolution": resolution_list, "$\widehat{H_{\mathrm{ave}}}-H_{\mathrm{ave}}$": error_list, "learning strategy": mtl_list})
        sns.boxplot(x="resolution", y="$\widehat{H_{\mathrm{ave}}}-H_{\mathrm{ave}}$", hue="learning strategy",
                    data=err_df, ax=ax_error, fliersize=0.1, whis=1.5)
    elif var == "footprint":
        err_df = pd.DataFrame({"resolution": resolution_list, "$\widehat{\lambda_p}-\lambda_p$": error_list, "learning strategy": mtl_list})
        sns.boxplot(x="resolution", y="$\widehat{\lambda_p}-\lambda_p$", hue="learning strategy",
                    data=err_df, ax=ax_error, fliersize=0.1, whis=1.5)
    else:
        raise NotImplementedError("Unknown target variable")
    # ---do violin plot
    # sns.violinplot(x="resolution", y="error", hue="learning strategy", data=err_df, ax=ax_error, gridsize=50)
    ax_error.set_yscale('symlog', linthresh=linthresh)

    report_df.to_csv("Fig-6-LosAngeles-{0}.csv".format(var), index=False)
    # plt.show()
    plt.savefig("Fig-6-LosAngeles-box_{0}.png".format(var), dpi=1000)


# Fig. 7. Compare the building height results in Glasgow at the resolution of 1km with Li's result
# i.e., a scatter plot, [BuildingHeight, BuildingFootprint] x [SENet-STL, SENet-MTL, Li's result]
# ---extent for plotting: xmin ymin xmax ymax
# [1] Glasgow, 100m: -4.380 55.760 -4.100 55.930
# ---we may use the following command via cmd first:
# `gdalwarp -te -4.4786 55.6777 -3.8846 56.0197 -tr 0.009 -0.009 -r bilinear -dstnodata -100000 srcfile dstfile`
def fig_7_Glasgow_compareLi(ref_prefix, res_prefix):
    backbone = "senet"
    model_name = "SENet"
    res_path = {
        "height": {
            "reference": os.path.join(ref_prefix, "1000m", "Li_comparison", "Glasgow_2020_building_height_LiClip_th.tif"),
            backbone: os.path.join(res_prefix, "1000m", "Li_comparison", "Glasgow_height_{0}_LiClip_th.tif".format(backbone)),
            backbone + "MTL": os.path.join(res_prefix, "1000m", "Li_comparison", "Glasgow_height_{0}_MTL_LiClip_th.tif".format(backbone)),
            "Li-RFR": os.path.join(res_prefix, "1000m", "Li_comparison", "Glasgow_height_li2020_clip_th.tif"),
        },
        "footprint": {
            "reference": os.path.join(ref_prefix, "1000m", "Li_comparison", "Glasgow_2020_building_footprint_LiClip_th.tif"),
            backbone: os.path.join(res_prefix, "1000m", "Li_comparison", "Glasgow_footprint_{0}_LiClip_th.tif".format(backbone)),
            backbone + "MTL": os.path.join(res_prefix, "1000m", "Li_comparison", "Glasgow_footprint_{0}_MTL_LiClip_th.tif".format(backbone)),
            "Li-RFR": os.path.join(res_prefix, "1000m", "Li_comparison", "Glasgow_footprint_li2020_clip_th.tif"),
        }
    }

    fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(20, 12))
    col_mapping = {"height": 0, "footprint": 1}

    res_dta = {"height": {}, "footprint": {}}

    for model in ["reference", backbone, backbone + "MTL", "Li-RFR"]:
        for var in ["height", "footprint"]:
            image = rasterio.open(res_path[var][model])
            image_nodata = image.nodata
            image_dta = image.read(1)
            image_dta = np.where(image_dta == image_nodata, np.nan, image_dta)
            image_dta = np.where(image_dta == 0.0, np.nan, image_dta)
            # ------save result data
            res_dta[var][model] = np.where(np.isnan(image_dta), 0.0, image_dta)

    model_index = [backbone, backbone + "MTL", "Li-RFR"]
    name_mapping = {backbone: model_name + "-STL", backbone + "MTL": model_name + "-MTL", "Li-RFR": "Li-RFR"}
    model_style_dict = {
        "RFR": {"color": "darkmagenta", "symbol": "^"},
        "SENet-STL": {"color": "indianred", "symbol": "s"},
        "SENet-MTL": {"color": "darkgreen", "symbol": "H"},
    }

    report_df = pd.DataFrame(columns=["Model", "Variable", "RMSE", "MAE", "NMAD", "BIAS-min", "BIAS-max",
                                      "R^2", "nME^2", "nRMSE_centered^2", "CC", "SSIM"])

    col_dta = {0: {}, 1: {}}
    for var in ["height", "footprint"]:
        col_id = col_mapping[var]
        for model in model_index:
            img_pred = res_dta[var][model]
            img_ref = res_dta[var]["reference"]

            img_ref_1d = img_ref.flatten()
            img_pred_1d = img_pred.flatten()

            model_error = img_pred_1d - img_ref_1d

            pt_id_select = np.logical_and(img_ref_1d != 0, img_pred_1d != 0)

            model_error_tmp = model_error[pt_id_select]
            print(len(model_error_tmp))

            # ------fill the report DataFrame
            ref_mean = np.mean(img_ref_1d[pt_id_select])
            ref_std = np.std(img_ref_1d[pt_id_select], ddof=1)
            pred_mean = np.mean(img_pred_1d[pt_id_select])

            name = name_mapping[model]
            result_summary = {"Model": name,
                              "Variable": var,
                              "RMSE": np.sqrt(np.mean(model_error_tmp**2)),
                              "MAE": np.mean(np.abs(model_error_tmp)),
                              "NMAD": get_NMAD(model_error_tmp),
                              "BIAS-min": np.min(model_error_tmp),
                              "BIAS-max": np.max(model_error_tmp),
                              "R^2": metrics.r2_score(y_true=img_ref_1d[pt_id_select],
                                                      y_pred=img_pred_1d[pt_id_select]),
                              "nME^2": (np.mean(model_error_tmp) / ref_std) ** 2,
                              "nRMSE_centered^2": np.mean((model_error_tmp - pred_mean + ref_mean) ** 2) / ref_std ** 2,
                              "CC": np.corrcoef(img_ref_1d[pt_id_select], img_pred_1d[pt_id_select])[0, 1],
                              "SSIM": ssim(im1=img_pred, im2=img_ref)}
            report_df = report_df.append(result_summary, ignore_index=True)

            col_dta[col_id][name] = {}
            col_dta[col_id][name]["ref"] = img_ref_1d[pt_id_select]
            col_dta[col_id][name]["pred"] = img_pred_1d[pt_id_select]

    report_df.to_csv("Fig-7-Glasgow-1km-comparison.csv", index=False)

    for i in range(0, 2):
        v_max = max([max(col_dta[i][model][type_name]) for model in [model_name + "-STL", model_name + "-MTL", "Li-RFR"] for type_name in ["ref", "pred"]])
        ax[i].scatter(col_dta[i]["Li-RFR"]["ref"], col_dta[i]["Li-RFR"]["pred"],
                      c="navy", marker="P", s=6, label="Li-RFR", alpha=0.2)
        ax[i].scatter(col_dta[i][model_name + "-STL"]["ref"], col_dta[i][model_name + "-STL"]["pred"],
                      c="darkorange", marker="s", s=6, label=model_name + "-STL", alpha=0.3)
        ax[i].scatter(col_dta[i][model_name + "-MTL"]["ref"], col_dta[i][model_name + "-MTL"]["pred"],
                      c="firebrick", marker="H", s=6, label=model_name + "-MTL", alpha=0.3)

        val = np.linspace(0, v_max, 10000)
        ax[i].plot(val, val, color="red", linestyle="--")

        if i == 0:
            scale = 10
            v_min = 1.0
            unit = "m"
            ref_label = "$H_{\mathrm{ave}}$"
            pred_label = "$\widehat{H_{\mathrm{ave}}}$"
        else:
            scale = 0.1
            v_min = 1e-4
            unit = "$\mathrm{m}^2$/$\mathrm{m}^2$"
            ref_label = "$\lambda_p$"
            pred_label = "$\widehat{\lambda_p}$"

        ax[i].set_xlim([v_min, np.ceil(v_max / scale) * scale])
        ax[i].set_ylim([v_min, np.ceil(v_max / scale) * scale])
        ax[i].set_xlabel("{0} [{1}]".format(ref_label, unit))
        ax[i].set_ylabel("{0} [{1}]".format(pred_label, unit))
        ax[i].legend(loc=2)

        ax[i].set_xscale("log")
        ax[i].set_yscale("log")

    # plt.show()
    plt.savefig("Fig-7-Glasgow-1km_comparison.png", dpi=1000)


# Fig. 8. Compare the aerodynamic parameters derived from different sources
# i.e., [zd, z0] x [reference, CBAM-STL, CBAM-MTL, Li's result]
def fig_8_Glasgow_aero_compareLi(ref_prefix, res_prefix):
    backbone = "senet"
    res_path = {
        "zd": {
            "reference": os.path.join(ref_prefix, "Glasgow_2020_ref_zd.tif"),
            backbone: os.path.join(res_prefix, "Glasgow_{0}_zd.tif".format(backbone)),
            backbone + "MTL": os.path.join(res_prefix, "Glasgow_{0}_MTL_zd.tif".format(backbone)),
            "L20": os.path.join(res_prefix, "Glasgow_li2020_zd.tif"),
        },
        "z0": {
            "reference": os.path.join(ref_prefix, "Glasgow_2020_ref_z0.tif"),
            backbone: os.path.join(res_prefix, "Glasgow_{0}_z0.tif".format(backbone)),
            backbone + "MTL": os.path.join(res_prefix, "Glasgow_{0}_MTL_z0.tif".format(backbone)),
            "L20": os.path.join(res_prefix, "Glasgow_li2020_z0.tif"),
        }
    }

    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(16, 4.5))
    range_mapping = {"zd": {"v_min": 0.0, "v_max": 15.0},
                     "z0": {"v_min": 0.0, "v_max": 3.0}}
    row_mapping = {"zd": 0, "z0": 1}
    pad_mapping = {"zd": -25, "z0": -20}
    col_mapping = {"reference": 0, backbone: 1, backbone + "MTL": 2, "L20": 3}

    res_dta = {"zd": {}, "z0": {}}
    var_name_mapping = {"zd": "$z_d$ [m]", "z0": "$z_0$ [m]"}

    for model in ["reference", backbone, backbone + "MTL", "L20"]:
        col_id = col_mapping[model]
        for var in ["zd", "z0"]:
            row_id = row_mapping[var]
            var_min = range_mapping[var]["v_min"]
            var_max = range_mapping[var]["v_max"]

            image = rasterio.open(res_path[var][model])
            image_nodata = image.nodata
            image_bound = list(image.bounds)
            image_dta = image.read(1)
            image_dta = np.where(image_dta == image_nodata, np.nan, image_dta)
            image_dta = np.where(image_dta == 0.0, np.nan, image_dta)
            im = ax[row_id, col_id].imshow(image_dta, cmap='rainbow', vmin=var_min, vmax=var_max,
                                           extent=[image_bound[0], image_bound[2], image_bound[1], image_bound[3]])
            rplt.show(image_dta, transform=image.transform, ax=ax[row_id, col_id], cmap="rainbow",
                      vmin=var_min, vmax=var_max)

            # ------save result data
            res_dta[var][model] = np.where(np.isnan(image_dta), 0.0, image_dta)

            if col_id == 3:
                position = ax[row_id, col_id].inset_axes([1.05, 0.02, 0.03, 0.96],
                                                          transform=ax[row_id, col_id].transAxes)
                cbar = fig.colorbar(im, ax=ax[row_id, col_id], cax=position)
                cbar.ax.set_ylabel(var_name_mapping[var], rotation=0, y=1.15, labelpad=pad_mapping[var])

            if row_id != 1:
                ax[row_id, col_id].tick_params(axis="x", which="both", bottom=False, labelbottom=False)

            if col_id != 0:
                ax[row_id, col_id].tick_params(axis="y", which="both", left=False, labelleft=False)

    report_df = pd.DataFrame(columns=["Model", "Variable", "RMSE", "MAE", "NMAD", "BIAS-min", "BIAS-max",
                                      "R^2", "nME^2", "nRMSE_centered^2", "CC", "SSIM"])

    name_mapping = {"cbam": "CBAM-STL", "cbamMTL": "CBAM-MTL", "L20": "L20",
                    "senet": "SENet-STL", "senetMTL": "SENet-MTL"}
    for model in [backbone, backbone + "MTL", "L20"]:
        for var in ["zd", "z0"]:
            var_pred = res_dta[var][model]
            var_ref = res_dta[var]["reference"]

            var_ref_1d = var_ref.flatten()
            var_pred_1d = var_pred.flatten()

            model_error = var_pred_1d - var_ref_1d

            pt_id_select = np.logical_and(var_ref_1d != 0, var_pred_1d != 0)

            model_error_tmp = model_error[pt_id_select]
            print(len(model_error_tmp))

            # ------fill the report DataFrame
            ref_mean = np.mean(var_ref_1d[pt_id_select])
            ref_std = np.std(var_ref_1d[pt_id_select], ddof=1)
            pred_mean = np.mean(var_pred_1d[pt_id_select])

            model_name = name_mapping[model]
            result_summary = {"Model": model_name,
                              "Variable": var,
                              "RMSE": np.sqrt(np.mean(model_error_tmp ** 2)),
                              "MAE": np.mean(np.abs(model_error_tmp)),
                              "NMAD": get_NMAD(model_error_tmp),
                              "BIAS-min": np.min(model_error_tmp),
                              "BIAS-max": np.max(model_error_tmp),
                              "R^2": metrics.r2_score(y_true=var_ref_1d[pt_id_select],
                                                      y_pred=var_pred_1d[pt_id_select]),
                              "nME^2": (np.mean(model_error_tmp) / ref_std) ** 2,
                              "nRMSE_centered^2": np.mean((model_error_tmp - pred_mean + ref_mean) ** 2) / ref_std ** 2,
                              "CC": np.corrcoef(var_ref_1d[pt_id_select], var_pred_1d[pt_id_select])[0, 1],
                              "SSIM": ssim(im1=var_pred, im2=var_ref)}
            report_df = report_df.append(result_summary, ignore_index=True)

    report_df.to_csv("Fig-8-aero_compare.csv", index=False)

    title_list = ["reference", "SENet-STL", "SENet-MTL", "L20"]
    col_label = ["Longitude", "Longitude", "Longitude", "Longitude"]
    row_label = ["Latitude", "Latitude"]

    for a, title in zip(ax[0], title_list):
        a.set_title(title, size='large')

    for a, col in zip(ax[-1], col_label):
        a.set_xlabel(col, size='large')

    for a, row in zip(ax[:, 0], row_label):
        a.set_ylabel(row, size='large', rotation=90)

    # plt.show()
    plt.savefig("Fig-8-aero_compare.pdf", bbox_inches='tight', dpi=1000)


if __name__ == "__main__":
    # get_prediction_DL_batch(base_dir="DL_run", ds_dir="/data/lyy/BuildingProject/dataset", save_dir="records", var="height", log_scale=False)
    # get_prediction_DL_batch(base_dir="DL_run", ds_dir="/data/lyy/BuildingProject/dataset", save_dir="records", var="footprint", log_scale=False)
    # get_prediction_ML(base_dir="ML_run", feature_dir="dataset", save_dir="records", var="footprint", log_scale=True)
    
    fig_2_dataset_loc_plot(csv_path="utils/HeightGen.csv",
                           path_prefix="/Volumes/ForLyy/Temp/ReferenceData", report=True)
    '''
    '''
    #fig_1_dataset_distribution_plot(plot_type="box", path_prefix="/data/lyy/BuildingProject/dataset")
    #fig_4_model_metric_corr(res_prefix="records", var="height")
    #fig_4_curve_STL_MTL(record_prefix="DL_run")
    #fig_4_model_metric_hist_2x4(res_prefix="records", num_bin=200, log_scale=True)
    #fig_4_r2_partition(res_prefix="records")
    #fig_4_metric_scale(res_prefix="records")
    #fig_5_model_close_up(var="footprint")
    #fig_5_model_close_up(var="height")
    #fig_6_Glasgow_compare(ref_prefix="testCase/infer_test_Glasgow", res_prefix="testCase/infer_test_Glasgow", var="footprint")
    #fig_6_Glasgow_compare(ref_prefix="testCase/infer_test_Glasgow", res_prefix="testCase/infer_test_Glasgow", var="height")
    #fig_6_LosAngeles_compare(ref_prefix="testCase/infer_test_LosAngeles", res_prefix="testCase/infer_test_LosAngeles",
                             #var="footprint")
    #fig_6_LosAngeles_compare(ref_prefix="testCase/infer_test_LosAngeles", res_prefix="testCase/infer_test_LosAngeles",
                             #var="height")
    #fig_7_Glasgow_compareLi(ref_prefix="testCase/infer_test_Glasgow", res_prefix="testCase/infer_test_Glasgow")
    #fig_8_Glasgow_aero_compareLi(ref_prefix="testCase/infer_test_Glasgow/1000m/Li_comparison/aero",
                                 #res_prefix="testCase/infer_test_Glasgow/1000m/Li_comparison/aero")
