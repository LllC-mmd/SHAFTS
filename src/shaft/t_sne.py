from sklearn import manifold
from paper_plot import *
from train import *


def get_feature_ML(dataset_path, save_dir, target_variable, resolution, aggregate_suffix,
                   filter_path, scaler_path, transformer_path,
                   scale_level=0, saved=True, num_cpu=24, chunk_size=50000, aux_namelist=None, **kwargs):
    if "save_suffix" in kwargs.keys():
        save_suffix = kwargs["save_suffix"]
    else:
        save_suffix = None

    feature, var_dta = GetDataPairFromDataset(dataset_path, target_variable, aggregate_suffix,
                                                scale_level, saved, num_cpu, chunk_size,
                                                save_suffix=save_suffix, aux_namelist=aux_namelist)

    feat_filter = joblib.load(filter_path)
    feat_scaler = joblib.load(scaler_path)
    feat_transformer = joblib.load(transformer_path)
    feature = feat_filter.transform(feature)
    feature = feat_scaler.transform(feature)
    feature = feat_transformer.transform(feature)

    var_mapping = {"BuildingHeight": "height", "BuildingFootprint": "footprint"}

    save_path = os.path.join(save_dir, "_".join(["ml", var_mapping[target_variable], "feat", resolution]) + ".npy")
    np.save(save_path, feature)

    save_path = os.path.join(save_dir, "_".join([var_mapping[target_variable], resolution, "ref"]) + ".npy")
    np.save(save_path, var_dta)


def get_feature_ML_batch(var, base_dir, ds_dir, save_dir):
    aux_namelist = ["DEM"]
    num_cpu = 18

    model_dir_path = {
        "100m": {
            "test_dataset": os.path.join(ds_dir, "patch_data_50pt_s15_100m_test.h5"),
            "filter": os.path.join(base_dir, "rf_{0}_100m".format(var), "filter_rf_n194_s15_100m.pkl"),
            "scaler": os.path.join(base_dir, "rf_{0}_100m".format(var), "scaler_rf_n194_s15_100m.pkl"),
            "transformer": os.path.join(base_dir, "rf_{0}_100m".format(var), "transformer_rf_n194_s15_100m.pkl"),
        },
        "250m": {
            "test_dataset": os.path.join(ds_dir, "patch_data_50pt_s30_250m_test.h5"),
            "filter": os.path.join(base_dir, "rf_{0}_250m".format(var), "filter_rf_n194_s30_250m.pkl"),
            "scaler": os.path.join(base_dir, "rf_{0}_250m".format(var), "scaler_rf_n194_s30_250m.pkl"),
            "transformer": os.path.join(base_dir, "rf_{0}_250m".format(var), "transformer_rf_n194_s30_250m.pkl"),
        },
        "500m": {
            "test_dataset": os.path.join(ds_dir, "patch_data_50pt_s60_500m_test.h5"),
            "filter": os.path.join(base_dir, "rf_{0}_500m".format(var), "filter_rf_n194_s60_500m.pkl"),
            "scaler": os.path.join(base_dir, "rf_{0}_500m".format(var), "scaler_rf_n194_s60_500m.pkl"),
            "transformer": os.path.join(base_dir, "rf_{0}_500m".format(var), "transformer_rf_n194_s60_500m.pkl"),
        },
        "1000m": {
            "test_dataset": os.path.join(ds_dir, "patch_data_50pt_s120_1000m_test.h5"),
            "filter": os.path.join(base_dir, "rf_{0}_1000m".format(var), "filter_rf_n194_s120_1000m.pkl"),
            "scaler": os.path.join(base_dir, "rf_{0}_1000m".format(var), "scaler_rf_n194_s120_1000m.pkl"),
            "transformer": os.path.join(base_dir, "rf_{0}_1000m".format(var), "transformer_rf_n194_s120_1000m.pkl"),
        }
    }

    name_mapping = {"height": "BuildingHeight", "footprint": "BuildingFootprint"}

    for resolution, input_size in [("100m", 15), ("250m", 30), ("500m", 60), ("1000m", 120)]:
        val_ds_path = model_dir_path[resolution]["test_dataset"]
        var_name = name_mapping[var]
        get_feature_ML(dataset_path=val_ds_path,
                       save_dir=save_dir,
                       target_variable=var_name,
                       resolution=resolution,
                       aggregate_suffix=["50pt"],
                       filter_path=model_dir_path[resolution]["filter"],
                       scaler_path=model_dir_path[resolution]["scaler"],
                       transformer_path=model_dir_path[resolution]["transformer"], save_suffix="_ml",
                       aux_namelist=aux_namelist, num_cpu=num_cpu)


def get_feature_DL(var, resolution, model_name, dataset_path, model_path, input_size, save_dir, aux_feat_info=None, activation=None, target_id_shift=None, log_scale=False, cuda_used=True, batch_size=128, num_workers=4, cached=True):
    if aux_feat_info is not None:
        aux_namelist = sorted(aux_feat_info.keys())
        aux_size = int(aux_feat_info[aux_namelist[0]] * input_size)

    # ------define data loader
    if model_name.endswith("MTL"):
        val_loader = load_data_lmdb_MTL(dataset_path, batch_size, num_workers, ["50pt"], mode="valid",
                                        cached=cached, log_scale=log_scale, aux_namelist=aux_namelist)
    else:
        val_loader = load_data_lmdb(dataset_path, batch_size, num_workers, ["50pt"], mode="valid",
                                    cached=cached, target_id_shift=target_id_shift, log_scale=log_scale,
                                    aux_namelist=aux_namelist)

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
        if aux_namelist is None:
            model_pred = model_SEResNet(in_plane=in_plane, input_channels=6, input_size=input_size,
                                            num_block=num_block, log_scale=log_scale, activation=activation,
                                            cuda_used=cuda_used, trained_record=model_path)
        else:
            model_pred = model_SEResNet_aux(in_plane=in_plane, input_channels=6, input_size=input_size,
                                                aux_input_size=aux_size, num_aux=len(aux_namelist),
                                                num_block=num_block, log_scale=log_scale, activation=activation,
                                                cuda_used=cuda_used, trained_record=model_path)
    elif model_name == "senetMTL":
        if aux_namelist is None:
            model_pred = model_SEResNetMTL(in_plane=in_plane, input_channels=6, input_size=input_size,
                                                num_block=num_block, log_scale=log_scale,
                                                cuda_used=cuda_used, trained_record=model_path)
        else:
            model_pred = model_SEResNetMTL_aux(in_plane=in_plane, input_channels=6, input_size=input_size,
                                                aux_input_size=aux_size, num_aux=len(aux_namelist),
                                                num_block=num_block, log_scale=log_scale,
                                                cuda_used=cuda_used, trained_record=model_path)
    elif model_name == "cbam":
        if aux_namelist is None:
            model_pred = model_CBAMResNet(in_plane=in_plane, input_channels=6, input_size=input_size,
                                            num_block=num_block, log_scale=log_scale, activation=activation,
                                            cuda_used=cuda_used, trained_record=model_path)
        else:
            model_pred = model_CBAMResNet_aux(in_plane=in_plane, input_channels=6, input_size=input_size,
                                                aux_input_size=aux_size, num_aux=len(aux_namelist),
                                                num_block=num_block, log_scale=log_scale, activation=activation,
                                                cuda_used=cuda_used, trained_record=model_path)
    elif model_name == "cbamMTL":
        if aux_namelist is None:
            model_pred = model_CBAMResNetMTL(in_plane=in_plane, input_channels=6, input_size=input_size,
                                                num_block=num_block, log_scale=log_scale,
                                                cuda_used=cuda_used, trained_record=model_path)
        else:
            model_pred = model_CBAMResNetMTL_aux(in_plane=in_plane, input_channels=6, input_size=input_size,
                                                    aux_input_size=aux_size, num_aux=len(aux_namelist),
                                                    num_block=num_block, log_scale=log_scale,
                                                    cuda_used=cuda_used, trained_record=model_path)
    else:
        raise NotImplementedError
    
    if cuda_used:
        model_pred = model_pred.cuda()

    model_pred.eval()
    # ------extract features after the calculation of NN's backbone part
    feat_list = []
    target_list = []
    if model_name.endswith("MTL"):
        if aux_namelist is None:
            for i, sample in enumerate(val_loader):
                input_band, target = sample["feature"], sample[var]
                if cuda_used:
                    input_band = input_band.cuda()
                with torch.no_grad():
                    feat = model_pred.get_feature(input_band)
        else:
            for i, sample in enumerate(val_loader):
                input_band, aux_feat, target = sample["feature"], sample["aux_feature"], sample[var]
                if cuda_used:
                    input_band, aux_feat = input_band.cuda(), aux_feat.cuda()
                with torch.no_grad():
                    feat = model_pred.get_feature(input_band, aux_feat)
    else:
        if aux_namelist is None:
            for i, sample in enumerate(val_loader):
                input_band, target = sample["feature"], sample["value"]
                if cuda_used:
                    input_band = input_band.cuda()
                with torch.no_grad():
                    feat = model_pred.get_feature(input_band)
        else:
            for i, sample in enumerate(val_loader):
                input_band, aux_feat, target = sample["feature"], sample["aux_feature"], sample["value"], 
                if cuda_used:
                    input_band, aux_feat = input_band.cuda(), aux_feat.cuda()
                with torch.no_grad():
                    feat = model_pred.get_feature(input_band, aux_feat)

    feat = feat.data.cpu().numpy()
    feat_list.append(feat)
    target_list.append(target)

    # ------save the prediction and target result
    feat_dta = np.concatenate(feat_list, axis=0)
    save_path = os.path.join(save_dir, var, "_".join([model_name, var, "feat", resolution]) + ".npy")
    np.save(save_path, feat_dta)

    target_dta = np.concatenate(target_list, axis=0)
    save_path = os.path.join(save_dir, var, "_".join([var, resolution, "refDL"]) + ".npy")
    np.save(save_path, target_dta)


def get_feature_DL_batch(base_dir, ds_dir, save_dir, var="height", log_scale=False, cuda_used=True, batch_size=128, num_workers=4, cached=True):
    backbone = "senet"
    aux_feat_info = {"DEM": 1.0}

    torch.multiprocessing.set_sharing_strategy('file_system')
    if var == "height":
        target_id_shift = 2
        activation = "relu"
        model_dir_path = {
            "100m": {
                "test_dataset": os.path.join(ds_dir, "patch_data_50pt_s15_100m_test.lmdb"),
                "senet": os.path.join(base_dir, var, "check_pt_senet_100m", "experiment_0", "checkpoint.pth.tar"),
                "cbam": os.path.join(base_dir, var, "check_pt_cbam_100m", "experiment_0", "checkpoint.pth.tar"),
                "senetMTL": os.path.join(base_dir, var, "check_pt_senet_100m_MTL", "experiment_1", "checkpoint.pth.tar"),
                "cbamMTL": os.path.join(base_dir, var, "check_pt_cbam_100m_MTL", "experiment_1", "checkpoint.pth.tar")
            },
            "250m": {
                "test_dataset": os.path.join(ds_dir, "patch_data_50pt_s30_250m_test.lmdb"),
                "senet": os.path.join(base_dir, var, "check_pt_senet_250m", "experiment_0", "checkpoint.pth.tar"),
                "cbam": os.path.join(base_dir, var, "check_pt_cbam_250m", "experiment_0", "checkpoint.pth.tar"),
                "senetMTL": os.path.join(base_dir, var, "check_pt_senet_250m_MTL", "experiment_1", "checkpoint.pth.tar"),
                "cbamMTL": os.path.join(base_dir, var, "check_pt_cbam_250m_MTL", "experiment_1", "checkpoint.pth.tar")
            },
            "500m": {
                "test_dataset": os.path.join(ds_dir, "patch_data_50pt_s60_500m_test.lmdb"),
                "senet": os.path.join(base_dir, var, "check_pt_senet_500m", "experiment_0", "checkpoint.pth.tar"),
                "cbam": os.path.join(base_dir, var, "check_pt_cbam_500m", "experiment_0", "checkpoint.pth.tar"),
                "senetMTL": os.path.join(base_dir, var, "check_pt_senet_500m_MTL", "experiment_1", "checkpoint.pth.tar"),
                "cbamMTL": os.path.join(base_dir, var, "check_pt_cbam_500m_MTL", "experiment_1", "checkpoint.pth.tar")
            },
            "1000m": {
                "test_dataset": os.path.join(ds_dir, "patch_data_50pt_s120_1000m_test.lmdb"),
                "senet": os.path.join(base_dir, var, "check_pt_senet_1000m", "experiment_0", "checkpoint.pth.tar"),
                "cbam": os.path.join(base_dir, var, "check_pt_cbam_1000m", "experiment_0", "checkpoint.pth.tar"),
                "senetMTL": os.path.join(base_dir, var, "check_pt_senet_1000m_MTL", "experiment_1", "checkpoint.pth.tar"),
                "cbamMTL": os.path.join(base_dir, var, "check_pt_cbam_1000m_MTL", "experiment_1", "checkpoint.pth.tar")
            }
        }
    elif var == "footprint":
        target_id_shift = 1
        activation = "sigmoid"
        model_dir_path = {
            "100m": {
                "test_dataset": os.path.join(ds_dir, "patch_data_50pt_s15_100m_test.lmdb"),
                "senet": os.path.join(base_dir, var, "check_pt_senet_100m", "experiment_0", "checkpoint.pth.tar"),
                "cbam": os.path.join(base_dir, var, "check_pt_cbam_100m", "experiment_0", "checkpoint.pth.tar"),
                "senetMTL": os.path.join(base_dir, var, "check_pt_senet_100m_MTL", "experiment_1", "checkpoint.pth.tar"),
                "cbamMTL": os.path.join(base_dir, var, "check_pt_cbam_100m_MTL", "experiment_1", "checkpoint.pth.tar")
            },
            "250m": {
                "test_dataset": os.path.join(ds_dir, "patch_data_50pt_s30_250m_test.lmdb"),
                "senet": os.path.join(base_dir, var, "check_pt_senet_250m", "experiment_0", "checkpoint.pth.tar"),
                "cbam": os.path.join(base_dir, var, "check_pt_cbam_250m", "experiment_0", "checkpoint.pth.tar"),
                "senetMTL": os.path.join(base_dir, var, "check_pt_senet_250m_MTL", "experiment_1", "checkpoint.pth.tar"),
                "cbamMTL": os.path.join(base_dir, var, "check_pt_cbam_250m_MTL", "experiment_1", "checkpoint.pth.tar")
            },
            "500m": {
                "test_dataset": os.path.join(ds_dir, "patch_data_50pt_s60_500m_test.lmdb"),
                "senet": os.path.join(base_dir, var, "check_pt_senet_500m", "experiment_0", "checkpoint.pth.tar"),
                "cbam": os.path.join(base_dir, var, "check_pt_cbam_500m", "experiment_0", "checkpoint.pth.tar"),
                "senetMTL": os.path.join(base_dir, var, "check_pt_senet_500m_MTL", "experiment_1", "checkpoint.pth.tar"),
                "cbamMTL": os.path.join(base_dir, var, "check_pt_cbam_500m_MTL", "experiment_1", "checkpoint.pth.tar")
            },
            "1000m": {
                "test_dataset": os.path.join(ds_dir, "patch_data_50pt_s120_1000m_test.lmdb"),
                "senet": os.path.join(base_dir, var, "check_pt_senet_1000m", "experiment_0", "checkpoint.pth.tar"),
                "cbam": os.path.join(base_dir, var, "check_pt_cbam_1000m", "experiment_0", "checkpoint.pth.tar"),
                "senetMTL": os.path.join(base_dir, var, "check_pt_senet_1000m_MTL", "experiment_1", "checkpoint.pth.tar"),
                "cbamMTL": os.path.join(base_dir, var, "check_pt_cbam_1000m_MTL", "experiment_1", "checkpoint.pth.tar")
            }
        }
    else:
        raise NotImplementedError("Unknown target variable")

    for resolution, input_size in [("100m", 15), ("250m", 30), ("500m", 60), ("1000m", 120)]:
        val_ds_path = model_dir_path[resolution]["test_dataset"]
        for model in [backbone, backbone + "MTL"]:
            model_file = model_dir_path[resolution][model]
            if model == backbone:
                get_feature_DL(var, resolution, model,
                               dataset_path=val_ds_path,
                               model_path=model_file,
                               input_size=input_size,
                               save_dir=save_dir,
                               aux_feat_info=aux_feat_info,
                               activation=activation, target_id_shift=target_id_shift,
                               log_scale=log_scale, cuda_used=cuda_used,
                               batch_size=batch_size,
                               num_workers=num_workers, cached=cached)
            elif model == backbone + "MTL":
                get_feature_DL(var, resolution, model,
                               dataset_path=val_ds_path,
                               model_path=model_file,
                               input_size=input_size,
                               save_dir=save_dir,
                               aux_feat_info=aux_feat_info,
                               activation=None, target_id_shift=None,
                               log_scale=log_scale, cuda_used=cuda_used,
                               batch_size=batch_size,
                               num_workers=num_workers, cached=cached)


def vis_feature_tSNE(var, feature, var_dta, fig, ax, marker_size=0.05, alpha=1.0, show_legend=False):
    if var == "height":
        ref_label = "$H_{\mathrm{ave}}$ [m]"
        pad_label = 25
        y_label = -0.6
    else:
        ref_label = "$\lambda_p$ [$\mathrm{m}^2$/$\mathrm{m}^2$]"
        pad_label = 30
        y_label = -1.2

    # ---use t-SNE for visualization on 2D hyperplane
    tsne_view = manifold.TSNE(n_components=2, init='pca', perplexity=30, learning_rate=10.0, n_iter=10000, n_jobs=16)
    y_reduced = tsne_view.fit_transform(feature)

    s = ax.scatter(y_reduced[:, 0], y_reduced[:, 1], c=var_dta, cmap="jet", s=marker_size, alpha=alpha, norm=colors.LogNorm())

    if show_legend:
        position = ax.inset_axes([-0.1, -0.2, 2.40, 0.04], transform=ax.transAxes)
        cbar = fig.colorbar(s, ax=ax, cax=position, orientation="horizontal")
        cbar.ax.tick_params(labelsize="medium")
        cbar.ax.set_ylabel(ref_label, labelpad=pad_label, y=y_label, rotation=0)


# [100m, 250m, 500m, 1000m] x [SENet-STL, SENet-MTL, ML]
def fig_4_model_feature_tSNE_3x4(var, res_prefix, save_suffix=None):
    feat_ref_pair = {
        "100m": {
            "SENet-STL": {
                "feature": os.path.join(res_prefix, "senet_{0}_feat_100m.npy".format(var)),
                var: os.path.join(res_prefix, "{0}_100m_refDL.npy".format(var))
            },
            "SENet-MTL": {
                "feature": os.path.join(res_prefix, "senetMTL_{0}_feat_100m.npy".format(var)),
                var: os.path.join(res_prefix, "{0}_100m_refDL.npy".format(var))
            },
            "ML": {
                "feature": os.path.join(res_prefix, "ml_{0}_feat_100m.npy".format(var)),
                var: os.path.join(res_prefix, "{0}_100m_ref.npy".format(var))
            }
        },
        "250m": {
            "SENet-STL": {
                "feature": os.path.join(res_prefix, "senet_{0}_feat_250m.npy".format(var)),
                var: os.path.join(res_prefix, "{0}_250m_refDL.npy".format(var))
            },
            "SENet-MTL": {
                "feature": os.path.join(res_prefix, "senetMTL_{0}_feat_250m.npy".format(var)),
                var: os.path.join(res_prefix, "{0}_250m_refDL.npy".format(var))
            },
            "ML": {
                "feature": os.path.join(res_prefix, "ml_{0}_feat_250m.npy".format(var)),
                var: os.path.join(res_prefix, "{0}_250m_ref.npy".format(var))
            }
        },
        "500m": {
            "SENet-STL": {
                "feature": os.path.join(res_prefix, "senet_{0}_feat_500m.npy".format(var)),
                var: os.path.join(res_prefix, "{0}_500m_refDL.npy".format(var))
            },
            "SENet-MTL": {
                "feature": os.path.join(res_prefix, "senetMTL_{0}_feat_500m.npy".format(var)),
                var: os.path.join(res_prefix, "{0}_500m_refDL.npy".format(var))
            },
            "ML": {
                "feature": os.path.join(res_prefix, "ml_{0}_feat_500m.npy".format(var)),
                var: os.path.join(res_prefix, "{0}_500m_ref.npy".format(var))
            }
        },
        "1000m": {
            "SENet-STL": {
                "feature": os.path.join(res_prefix, "senet_{0}_feat_1000m.npy".format(var)),
                var: os.path.join(res_prefix, "{0}_1000m_refDL.npy".format(var))
            },
            "SENet-MTL": {
                "feature": os.path.join(res_prefix, "senetMTL_{0}_feat_1000m.npy".format(var)),
                var: os.path.join(res_prefix, "{0}_1000m_refDL.npy".format(var))
            },
            "ML": {
                "feature": os.path.join(res_prefix, "ml_{0}_feat_1000m.npy".format(var)),
                var: os.path.join(res_prefix, "{0}_1000m_ref.npy".format(var))
            }
        },
    }

    num_100 = len(np.load(feat_ref_pair["100m"]["ML"]["feature"]))
    num_250 = len(np.load(feat_ref_pair["250m"]["ML"]["feature"]))
    num_500 = len(np.load(feat_ref_pair["500m"]["ML"]["feature"]))
    num_1000 = len(np.load(feat_ref_pair["1000m"]["ML"]["feature"]))

    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(8, 6))
    row_id_mapping = {"SENet-STL": 0, "SENet-MTL": 1, "ML": 2}
    row_label_mapping = {0: "SENet-STL", 1: "SENet-MTL", 2: "ML"}
    col_id_mapping = {"100m": 0, "250m": 1, "500m": 2, "1000m": 3}
    col_label_mapping = {0: "100m (N={0})".format(int(num_100)),
                         1: "250m (N={0})".format(int(num_250)),
                         2: "500m (N={0})".format(int(num_500)),
                         3: "1000m (N={0})".format(int(num_1000))}

    marker_size_mapping = {"100m": 0.01, "250m": 0.01, "500m": 0.1, "1000m": 0.2}
    alpha_mapping = {"100m": 1.0, "250m": 1.0, "500m": 1.0, "1000m": 1.0}

    for resolution in ["100m", "250m", "500m", "1000m"]:
        col_id = col_id_mapping[resolution]
        for res_type in ["SENet-STL", "SENet-MTL", "ML"]:
            row_id = row_id_mapping[res_type]
            val_ref = feat_ref_pair[resolution][res_type]
            feat_dta = np.load(val_ref["feature"])
            var_dta = np.load(val_ref[var])

            if col_id == 1 and row_id == 2:
                show_legend = True
            else:
                show_legend = False

            vis_feature_tSNE(var, feat_dta, var_dta, fig, ax[row_id, col_id], marker_size=marker_size_mapping[resolution], alpha=alpha_mapping[resolution], show_legend=show_legend)

            if row_id == 0:
                ax[row_id, col_id].set_title(col_label_mapping[col_id], size="medium")

            if col_id == 0:
                ax[row_id, col_id].set_ylabel(row_label_mapping[row_id], size="medium")

            ax[row_id, col_id].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            ax[row_id, col_id].tick_params(axis="y", which="both", left=False, labelleft=False)

    if save_suffix is not None:
        plt.savefig("Fig-4-tSNE_{0}.pdf".format(save_suffix), bbox_inches='tight', dpi=600)
    else:
        plt.savefig("Fig-4-tSNE.pdf", bbox_inches='tight', dpi=600)


if __name__ == "__main__":
    get_feature_DL_batch(base_dir="DL_run", ds_dir="/data/lyy/BuildingProject/dataset", save_dir="records_feat", var="height", log_scale=False)
    get_feature_DL_batch(base_dir="DL_run", ds_dir="/data/lyy/BuildingProject/dataset", save_dir="records_feat", var="footprint", log_scale=False)
    get_feature_ML_batch(var="height", base_dir="ML_run/height", ds_dir="/data/lyy/BuildingProject/dataset", save_dir="records_feat/height")
    #fig_4_model_feature_tSNE_3x4(var="height", res_prefix="records_feat/height", save_suffix="height")
    #get_feature_ML_batch(var="footprint", base_dir="ML_run/footprint", ds_dir="/data/lyy/BuildingProject/dataset", save_dir="records_feat/footprint")
    #fig_4_model_feature_tSNE_3x4(var="footprint", res_prefix="records_feat/footprint", save_suffix="footprint")
