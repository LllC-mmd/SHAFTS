from re import I
from dataset import *
from DL_dataset import *
from DL_model import *
from train import *


def split_city_from_dataset(src_dataset_path, out_dataset_dir):
    src_db = h5py.File(src_dataset_path, "r")

    city_namelist = np.unique([c.split("@")[0] for c in src_db.keys()]).tolist()
    for city in city_namelist:
        city_key = [c for c in src_db.keys() if c.startswith(city)]
        out_db = h5py.File(os.path.join(out_dataset_dir, city+".h5"), "w")
        out_group_db = out_db.create_group(city)

        db_tmp = {}
        for c in city_key:
            for dataset_name in src_db[c].keys():
                dataset = src_db[c][dataset_name][()]
                if dataset_name not in db_tmp.keys():
                    db_tmp[dataset_name] = []

                db_tmp[dataset_name].append(dataset)

        for ds_name, ds_val in db_tmp.items():
            out_group_db.create_dataset(ds_name, data=np.concatenate(ds_val, axis=0))

    src_db.close()
    out_db.close()


def get_city_prediction_ML(base_dir, ds_dir, save_dir, var="height", log_scale=True):
    model_dir_path = {
        "100m": {
            "rf": os.path.join(base_dir, var, "_".join(["rf", var, "100m"])),
            "xgb": os.path.join(base_dir, var, "_".join(["xgb", var, "100m"])),
            "baggingSVR": os.path.join(base_dir, var, "_".join(["baggingSVR", var, "100m"]))
        },
        "250m": {
            "rf": os.path.join(base_dir, var, "_".join(["rf", var, "250m"])),
            "xgb": os.path.join(base_dir, var, "_".join(["xgb", var, "250m"])),
            "baggingSVR": os.path.join(base_dir, var, "_".join(["baggingSVR", var, "250m"]))
        },
        "500m": {
            "rf": os.path.join(base_dir, var, "_".join(["rf", var, "500m"])),
            "xgb": os.path.join(base_dir, var, "_".join(["xgb", var, "500m"])),
            "SVR": os.path.join(base_dir, var, "_".join(["SVR", var, "500m"]))
        },
        "1000m": {
            "rf": os.path.join(base_dir, var, "_".join(["rf", var, "1000m"])),
            "xgb": os.path.join(base_dir, var, "_".join(["xgb", var, "1000m"])),
            "SVR": os.path.join(base_dir, var, "_".join(["SVR", var, "1000m"]))
        }
    }

    filter_pred = None
    scaler_pred = None
    transformer_pred = None
    model_pred = None

    if var == "height":
        target_variable = "BuildingHeight"
    else:
        target_variable = "BuildingFootprint"
    
    aggregate_suffix = ["50pt"]
    scale_level = 0
    num_cpu = 2
    chunk_size = 50000
    aux_namelist = ["DEM"]
    saved = False
    
    for resolution, input_size in [("100m", 15), ("250m", 30), ("500m", 60), ("1000m", 120)]:
        path_prefix_tmp = os.path.join(ds_dir, resolution+"_tmp")
        ds_list = [f for f in os.listdir(path_prefix_tmp) if f.endswith(".h5")]

        for ds in ds_list:
            ds_base = os.path.splitext(ds)[0]
            val_ds_path = os.path.join(path_prefix_tmp, ds)

            save_dir_tmp = os.path.join(save_dir, ds_base, var)
            if not os.path.exists(save_dir_tmp):
                os.makedirs(save_dir_tmp)

            test_suffix = "_s" + str(input_size) + "_" + resolution
            feat, target_var = GetDataPairFromDataset(val_ds_path, target_variable, aggregate_suffix,
                                                        scale_level, saved, num_cpu, chunk_size,
                                                        save_suffix=test_suffix, aux_namelist=aux_namelist)

            # ------save the reference value
            save_path = os.path.join(save_dir_tmp, "_".join([var, resolution, "ref"]) + ".npy")
            np.save(save_path, target_var)

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
                                model_pred = RandomForestModel(n_jobs=10)
                            elif model == "xgb":
                                model_pred = XGBoostRegressionModel(n_jobs=10)
                            elif model == "baggingSVR":
                                model_pred = BaggingSupportVectorRegressionModel(n_jobs=20)
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
                    save_path = os.path.join(save_dir_tmp, "_".join([model, var, resolution]) + ".npy")
                    np.save(save_path, var_pred)

                    print("\t".join([model, var, resolution, "finished"]))


def get_city_prediction_DL(var, resolution, model_name, dataset_path, model_path, input_size, save_dir, aux_feat_info=None, activation=None, target_id_shift=None, log_scale=False, cuda_used=True, batch_size=128, num_workers=4, cached=True):
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

    '''
    if target_id_shift == 1:
        var = "BuildingFootprint"
    else:
        var = "BuildingHeight"
    if model_name.endswith("MTL"):
        val_loader = load_data_hdf5_MTL(dataset_path, batch_size, num_workers, aggregation_list=["50pt"], mode="valid",
                                            cached=cached, log_scale=log_scale, aux_namelist=aux_namelist)
    else:
        val_loader = load_data_hdf5(dataset_path, batch_size, num_workers, target_variable=var, aggregation_list=["50pt"], mode="valid",
                                        cached=cached, log_scale=log_scale, aux_namelist=aux_namelist)
    '''

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
    # ------do target variable prediction
    res_list = None
    target_list = None
    if model_name.endswith("MTL"):
        res_list = {"height": [], "footprint": []}
        target_list = {"height": [], "footprint": []}
        for i, sample in enumerate(val_loader):
            if aux_namelist is None:
                input_band, target_footprint, target_height = sample["feature"], sample["footprint"], sample["height"]
                if cuda_used:
                    input_band, target_footprint, target_height = input_band.cuda(), target_footprint.cuda(), target_height.cuda()
                with torch.no_grad():
                    output_footprint, output_height = model_pred(input_band)
            else:
                input_band, aux_feat, target_footprint, target_height = sample["feature"], sample["aux_feature"], sample["footprint"], sample["height"]
                if cuda_used:
                    input_band, aux_feat, target_footprint, target_height = input_band.cuda(), aux_feat.cuda(), target_footprint.cuda(), target_height.cuda()
                with torch.no_grad():
                    output_footprint, output_height = model_pred(input_band, aux_feat)
            
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
            if aux_namelist is None:
                input_band, target = sample["feature"], sample["value"]
                if cuda_used:
                    input_band, target = input_band.cuda(), target.cuda()
                with torch.no_grad():
                    output = model_pred(input_band)
            else:
                input_band, target, aux_feat = sample["feature"], sample["value"], sample["aux_feature"]
                if cuda_used:
                    input_band, target, aux_feat = input_band.cuda(), target.cuda(), aux_feat.cuda()
                with torch.no_grad():
                    output = model_pred(input_band, aux_feat)
            
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
        save_dir_tmp = os.path.join(save_dir, k)
        if not os.path.exists(save_dir_tmp):
            os.makedirs(save_dir_tmp)
        save_path = os.path.join(save_dir, k, "_".join([model_name, k, resolution]) + ".npy")
        np.save(save_path, res_list[k])
    for k in target_list.keys():
        target_list[k] = np.concatenate(target_list[k], axis=0)
        save_dir_tmp = os.path.join(save_dir, k)
        if not os.path.exists(save_dir_tmp):
            os.makedirs(save_dir_tmp)
        save_path = os.path.join(save_dir, k, "_".join([k, resolution, "refDL"]) + ".npy")
        np.save(save_path, target_list[k])


def get_city_prediction_DL_batch(base_dir, ds_dir, save_dir, var="height", log_scale=False, cuda_used=True, batch_size=128, num_workers=4, cached=True):
    backbone = "senet"
    aux_feat_info = {"DEM": 1.0}

    if var == "height":
        target_id_shift = 2
        activation = "relu"
        model_dir_path = {
            "100m": {
                "senet": os.path.join(base_dir, var, "check_pt_senet_100m", "experiment_0", "checkpoint.pth.tar"),
                "cbam": os.path.join(base_dir, var, "check_pt_cbam_100m", "experiment_0", "checkpoint.pth.tar"),
                "senetMTL": os.path.join(base_dir, var, "check_pt_senet_100m_MTL", "experiment_1", "checkpoint.pth.tar"),
                "cbamMTL": os.path.join(base_dir, var, "check_pt_cbam_100m_MTL", "experiment_1", "checkpoint.pth.tar")
            },
            "250m": {
                "senet": os.path.join(base_dir, var, "check_pt_senet_250m", "experiment_0", "checkpoint.pth.tar"),
                "cbam": os.path.join(base_dir, var, "check_pt_cbam_250m", "experiment_0", "checkpoint.pth.tar"),
                "senetMTL": os.path.join(base_dir, var, "check_pt_senet_250m_MTL", "experiment_1", "checkpoint.pth.tar"),
                "cbamMTL": os.path.join(base_dir, var, "check_pt_cbam_250m_MTL", "experiment_1", "checkpoint.pth.tar")
            },
            "500m": {
                "senet": os.path.join(base_dir, var, "check_pt_senet_500m", "experiment_0", "checkpoint.pth.tar"),
                "cbam": os.path.join(base_dir, var, "check_pt_cbam_500m", "experiment_0", "checkpoint.pth.tar"),
                "senetMTL": os.path.join(base_dir, var, "check_pt_senet_500m_MTL", "experiment_1", "checkpoint.pth.tar"),
                "cbamMTL": os.path.join(base_dir, var, "check_pt_cbam_500m_MTL", "experiment_1", "checkpoint.pth.tar")
            },
            "1000m": {
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
                "senet": os.path.join(base_dir, var, "check_pt_senet_100m", "experiment_0", "checkpoint.pth.tar"),
                "cbam": os.path.join(base_dir, var, "check_pt_cbam_100m", "experiment_0", "checkpoint.pth.tar"),
                "senetMTL": os.path.join(base_dir, var, "check_pt_senet_100m_MTL", "experiment_1", "checkpoint.pth.tar"),
                "cbamMTL": os.path.join(base_dir, var, "check_pt_cbam_100m_MTL", "experiment_1", "checkpoint.pth.tar")
            },
            "250m": {
                "senet": os.path.join(base_dir, var, "check_pt_senet_250m", "experiment_0", "checkpoint.pth.tar"),
                "cbam": os.path.join(base_dir, var, "check_pt_cbam_250m", "experiment_0", "checkpoint.pth.tar"),
                "senetMTL": os.path.join(base_dir, var, "check_pt_senet_250m_MTL", "experiment_1", "checkpoint.pth.tar"),
                "cbamMTL": os.path.join(base_dir, var, "check_pt_cbam_250m_MTL", "experiment_1", "checkpoint.pth.tar")
            },
            "500m": {
                "senet": os.path.join(base_dir, var, "check_pt_senet_500m", "experiment_0", "checkpoint.pth.tar"),
                "cbam": os.path.join(base_dir, var, "check_pt_cbam_500m", "experiment_0", "checkpoint.pth.tar"),
                "senetMTL": os.path.join(base_dir, var, "check_pt_senet_500m_MTL", "experiment_1", "checkpoint.pth.tar"),
                "cbamMTL": os.path.join(base_dir, var, "check_pt_cbam_500m_MTL", "experiment_1", "checkpoint.pth.tar")
            },
            "1000m": {
                "senet": os.path.join(base_dir, var, "check_pt_senet_1000m", "experiment_0", "checkpoint.pth.tar"),
                "cbam": os.path.join(base_dir, var, "check_pt_cbam_1000m", "experiment_0", "checkpoint.pth.tar"),
                "senetMTL": os.path.join(base_dir, var, "check_pt_senet_1000m_MTL", "experiment_1", "checkpoint.pth.tar"),
                "cbamMTL": os.path.join(base_dir, var, "check_pt_cbam_1000m_MTL", "experiment_1", "checkpoint.pth.tar")
            }
        }
    else:
        raise NotImplementedError("Unknown target variable")

    for resolution, input_size in [("100m", 15), ("250m", 30), ("500m", 60), ("1000m", 120)]:
        path_prefix_tmp = os.path.join(ds_dir, resolution+"_tmp")
        ds_list = [f for f in os.listdir(path_prefix_tmp) if f.endswith(".lmdb")]
        for ds in ds_list:
            ds_base = os.path.splitext(ds)[0]
            val_ds_path = os.path.join(path_prefix_tmp, ds)

            save_dir_tmp = os.path.join(save_dir, ds_base)
            if not os.path.exists(save_dir_tmp):
                os.makedirs(save_dir_tmp)
            
            for model in [backbone, backbone + "MTL"]:
                model_file = model_dir_path[resolution][model]
                if model == backbone:
                    get_city_prediction_DL(var, resolution, model,
                                            dataset_path=val_ds_path,
                                            model_path=model_file,
                                            input_size=input_size,
                                            save_dir=save_dir_tmp,
                                            aux_feat_info=aux_feat_info,
                                            activation=activation, target_id_shift=target_id_shift,
                                            log_scale=log_scale, cuda_used=cuda_used,
                                            batch_size=batch_size,
                                            num_workers=num_workers, cached=cached)
                elif model == backbone + "MTL":
                    get_city_prediction_DL(var, resolution, model,
                                            dataset_path=val_ds_path,
                                            model_path=model_file,
                                            input_size=input_size,
                                            save_dir=save_dir_tmp,
                                            aux_feat_info=aux_feat_info,
                                            activation=None, target_id_shift=None,
                                            log_scale=log_scale, cuda_used=cuda_used,
                                            batch_size=batch_size,
                                            num_workers=num_workers, cached=cached)                   


def check_split_content(src_dataset_path, out_dataset_path, key, group_name="BuildingHeight", lmdb_id=3, sub_lmdb_id=None):
    src_db = h5py.File(src_dataset_path, "r")
    val_ref = src_db[key][group_name][()]

    if lmdb_id == 0 or lmdb_id == 1:
        if "sentinel_1" in group_name:
            v_tmp = []
            for v in val_ref:
                v = get_backscatterCoef(v) * 255
                v = np.where(v > 255, 255, v)
                v = np.transpose(v, (1, 2, 0))
                v_tmp.append(v)
            val_ref = np.concatenate(v_tmp, axis=-1)
        elif "sentinel_2" in group_name:
            v_tmp = []
            for v in val_ref:
                v = v * 1.0 * 255
                v = np.transpose(v, (1, 2, 0))
                v_tmp.append(v)
            val_ref = np.concatenate(v_tmp, axis=-1)
        else:
            v_tmp = []
            for v in val_ref:
                v = np.transpose(v, (1, 2, 0))
                v_tmp.append(v)
            val_ref = np.concatenate(v_tmp, axis=-1)
    
    if out_dataset_path.endswith("h5"):
        out_db = h5py.File(out_dataset_path, "r")
        val_compare = out_db[key][group_name][()]
        if lmdb_id == 0 or lmdb_id == 1:
            if "sentinel_1" in group_name:
                v_tmp = []
                for v in val_ref:
                    v = get_backscatterCoef(v) * 255
                    v = np.where(v > 255, 255, v)
                    v = np.transpose(v, (1, 2, 0))
                    v_tmp.append(v)
                val_ref = np.concatenate(v_tmp, axis=-1)
            elif "sentinel_2" in group_name:
                v_tmp = []
                for v in val_ref:
                    v = v * 1.0 * 255
                    v = np.transpose(v, (1, 2, 0))
                    v_tmp.append(v)
                val_ref = np.concatenate(v_tmp, axis=-1)
            else:
                v_tmp = []
                for v in val_ref:
                    v = np.transpose(v, (1, 2, 0))
                    v_tmp.append(v)
                val_compare = np.concatenate(v_tmp, axis=-1)
    elif out_dataset_path.endswith("lmdb"):
        env = lmdb.open(out_dataset_path, subdir=os.path.isdir(out_dataset_path),
                            readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            db_length = pa.deserialize(txn.get(b'__len__'))
            db_keys = pa.deserialize(txn.get(b'__keys__'))
        
        val_tmp = []
        for ids in range(0, db_length):
            with env.begin(write=False) as txn:
                byteflow = txn.get(db_keys[ids])

            dta_unpacked = pa.deserialize(byteflow)
            if sub_lmdb_id is None:
                dta_tmp = dta_unpacked[lmdb_id]
            else:
                dta_tmp = dta_unpacked[lmdb_id][:, :, sub_lmdb_id]

            if lmdb_id == 0 or lmdb_id == 1:
                dta_tmp = np.expand_dims(dta_tmp, axis=-1)
                val_tmp.append(dta_tmp)
            else:
                val_tmp.append(dta_tmp)

        if not isinstance(dta_tmp, np.ndarray):
            val_compare = np.array(val_tmp)
        else:
            val_compare = np.concatenate(val_tmp, axis=-1)
    
    err = val_ref - val_compare
    err_abs = np.abs(err)
    
    print(err.shape[2])
    for i in range(0, err.shape[2]):
        print(np.mean(err[:, :, i]), np.max(err_abs[:, :, i]))


def check_combined_result(res_dir, var="height"):
    city_list = os.listdir(res_dir)

    res_combined = {"100m": [], "250m": [], "500m": [], "1000m": []}
    for resolution in ["100m", "250m", "500m", "1000m"]:
        res_combined[resolution] = {}
        for model in ["rf", "xgb", "baggingSVR", "SVR", "senet", "senetMTL", "ref", "refDL"]:
            res_combined[resolution][model] = []
    
    for resolution in ["100m", "250m", "500m", "1000m"]:
        for city in city_list:
            res_prefix_tmp = os.path.join(res_dir, city, var)
            res_list = [f for f in os.listdir(res_prefix_tmp) if f.endswith(".npy") and resolution in f]
            for f in res_list:
                dta_path = os.path.join(res_prefix_tmp, f)
                dta_tmp = np.load(dta_path)

                f_resolution = re.findall(re.compile(r"\d+"), f)[0] + "m"
                ref_flag = re.findall(re.compile(r"(ref\w*)."), f)
                if len(ref_flag) == 0:
                    f_model = f.split("_")[0]
                else:
                    f_model = ref_flag[0]

                res_combined[f_resolution][f_model].append(dta_tmp)
    
    for resolution in ["100m", "250m", "500m", "1000m"]:
        for k in res_combined[resolution].keys():
            if len(res_combined[resolution][k]) > 0:
                res_combined[resolution][k] = np.concatenate(res_combined[resolution][k], axis=0)
                if "ref" in k:
                    output_path = os.path.join(res_dir, "_".join([var, resolution, k])+".npy")
                else:
                    output_path = os.path.join(res_dir, "_".join([k, var, resolution])+".npy")
                np.save(output_path, res_combined[resolution][k])
                


if __name__ == "__main__":
    path_prefix = "/data/lyy/BuildingProject"
    '''
    src_dir = os.path.join(path_prefix, "dataset")

    res_scale_mapping = {100: 15, 250: 30, 500: 60, 1000: 120}

    for rs_res in [100, 250, 500, 1000]:
        scale = [res_scale_mapping[rs_res]]
        h5_path = os.path.join(src_dir, "patch_data_50pt_s%d_%dm_valid.h5" % (scale[0], rs_res))
        
        dst_dir = os.path.join(src_dir, str(rs_res)+"m"+"_tmp")
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        
        split_city_from_dataset(src_dataset_path=h5_path, out_dataset_dir=dst_dir)
    '''
    
    '''
    aux_feat = ["DEM"]
    for res in ["100m", "250m", "500m", "1000m"]:
        path_prefix_tmp = os.path.join(path_prefix, "dataset", res+"_tmp")
        h5_list = [f for f in os.listdir(path_prefix_tmp) if f.endswith(".h5")]

        for f in h5_list:
            f_base = os.path.splitext(f)[0]
            HDF5_to_LMDB(h5_dataset_path=os.path.join(path_prefix_tmp, f),
                            lmdb_dataset_path=os.path.join(path_prefix_tmp, f_base+".lmdb"),
                            target_list=["BuildingFootprint", "BuildingHeight"], aggregation_list=["50pt"],
                            s1_prefix="sentinel_1", s2_prefix="sentinel_2", 
                            scale=1.0, max_size=1099511627776, aux_namelist=aux_feat)
    '''
    
    # get_city_prediction_ML(base_dir="ML_run", ds_dir="/data/lyy/BuildingProject/dataset", save_dir="records/cities_tmp", var="height", log_scale=True)
    # get_city_prediction_ML(base_dir="ML_run", ds_dir="/data/lyy/BuildingProject/dataset", save_dir="records/cities", var="footprint", log_scale=True)

    # get_city_prediction_DL_batch(base_dir="DL_run", ds_dir="/data/lyy/BuildingProject/dataset", save_dir="records/cities_tmp", var="height", log_scale=False, cuda_used=True, batch_size=2, num_workers=2, cached=True)
    # get_city_prediction_DL_batch(base_dir="DL_run", ds_dir="/data/lyy/BuildingProject/dataset", save_dir="records/cities", var="footprint", log_scale=False, cuda_used=True, batch_size=2, num_workers=2, cached=True)

    '''
    check_split_content(src_dataset_path=os.path.join(path_prefix, "dataset", "patch_data_50pt_s15_100m_test.h5"), 
                            out_dataset_path=os.path.join(path_prefix, "dataset/100m", "Airdrie.lmdb"), 
                            key="Airdrie@0", group_name="DEM", lmdb_id=1, sub_lmdb_id=0)
    '''

    check_combined_result(res_dir="records/cities", var="footprint")

