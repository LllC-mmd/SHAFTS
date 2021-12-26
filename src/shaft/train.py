import argparse
import h5py
from sklearn import feature_selection as fs
from sklearn import model_selection as ms
from sklearn import preprocessing
from sklearn import decomposition

from .model import *


# ************************* [1] Feature Collection *************************
def get_feature(raw_band, band_ref, s2_band_use, band_statistics, s2_morph_ops, DMP_size, GLCM_angle, TF_list, rescaled=True, reported=True, num_cpu=None):
    band_stat = get_band_statistics_func(band_statistics)
    bs_key = sorted(band_stat.keys())

    texture_func = get_texture_statistics(TF_list)

    city_feature = []

    for suffix in sorted(band_ref.keys()):
        if isinstance(band_ref[suffix], dict):
            # ---------------[1] Sentinel-1-VV
            s1_vv = raw_band[band_ref[suffix]["VV"]]
            s1_vv_stat = [band_stat[stat](s1_vv) for stat in bs_key]  # shape(s1_vv_stat) = (n_statistics, n_sample_city)
            city_feature.append(np.transpose(s1_vv_stat))
            # ---------------[2] Sentinel-1-VH
            s1_vh = raw_band[band_ref[suffix]["VH"]]
            s1_vh_stat = [band_stat[stat](s1_vh) for stat in bs_key]
            city_feature.append(np.transpose(s1_vh_stat))
            # ---------------[3] Sentinel-2-R/G/B/NIR
            s2_r = raw_band[band_ref[suffix]["R"]]
            s2_g = raw_band[band_ref[suffix]["G"]]
            s2_b = raw_band[band_ref[suffix]["B"]]
            s2_nir = raw_band[band_ref[suffix]["NIR"]]
            # ------------------if rescaled = True, we use Instance Normalization for R/G/B and NIR band
            if rescaled:
                s2_r = rgb_rescale(dta=s2_r, axis=(-1, -2))
                s2_g = rgb_rescale(dta=s2_g, axis=(-1, -2))
                s2_b = rgb_rescale(dta=s2_b, axis=(-1, -2))
                s2_nir = rgb_rescale(dta=s2_nir, axis=(-1, -2))

            luminance = get_luminance(s2_r, s2_g, s2_b, reduced=False)

            s2_r_stat = [band_stat[stat](s2_r) for stat in bs_key]
            city_feature.append(np.transpose(s2_r_stat))

            s2_g_stat = [band_stat[stat](s2_g) for stat in bs_key]
            city_feature.append(np.transpose(s2_g_stat))

            s2_b_stat = [band_stat[stat](s2_b) for stat in bs_key]
            city_feature.append(np.transpose(s2_b_stat))

            s2_nir_stat = [band_stat[stat](s2_nir) for stat in bs_key]
            city_feature.append(np.transpose(s2_nir_stat))

            # ------------------set a dict of bands for further feature calculation
            band_dict = {"Red": s2_r, "Green": s2_g, "Blue": s2_b, "NIR": s2_nir, "Brightness": luminance}

            # ------------------3.1 Spectral Features (SFs) where # of SFs = 7
            ndvi_avg = get_normalized_index(s2_nir, s2_r, reduced=True)
            n_nir_green = get_normalized_index(s2_nir, s2_g, reduced=True)
            n_nir_blue = get_normalized_index(s2_nir, s2_b, reduced=True)
            n_red_green = get_normalized_index(s2_r, s2_g, reduced=True)
            n_red_blue = get_normalized_index(s2_r, s2_b, reduced=True)
            n_green_blue = get_normalized_index(s2_g, s2_b, reduced=True)
            luminance_avg = np.mean(luminance, axis=(-1, -2))
            sf = [ndvi_avg, n_nir_green, n_nir_blue, n_red_green, n_red_blue, n_green_blue, luminance_avg]
            city_feature.append(np.transpose(sf))

            # ------------------3.2 Morphological Features (MFs) where # of MFs = 5 * 4 * (n_SE - 1)
            # --------------------- n_SE is the number of the size of the Structural Element
            MF_dict = {}
            for b in s2_band_use:
                for op in s2_morph_ops:
                    MF_dict[b + "@" + op] = get_DMP_mean_batch(img_list=band_dict[b], size_list=DMP_size, method=op, num_cpu=num_cpu)

            for mk in sorted(MF_dict.keys()):
                city_feature.append(np.transpose(MF_dict[mk]))

            # ------------------3.3 Texture Features (TFs) where # of TFs = 8 * 5 * n_SE
            TF_dict = {}
            GLCM_dict = {}
            for b in s2_band_use:
                for s in DMP_size:
                    # shape(GLCM_dict[b + "@" + str(s)]) = (n_sample_city, n_level, n_level)
                    GLCM_dict[b + "@" + str(s)] = get_avg_GLCM_batch(band_dict[b], s, GLCM_angle, normed=True, num_cpu=num_cpu)

            for gk in sorted(GLCM_dict.keys()):
                for func in sorted(texture_func.keys()):
                    TF_dict[gk + "@" + func] = texture_func[func](GLCM_dict[gk])

            for tk in sorted(TF_dict.keys()):
                city_feature.append(np.reshape(TF_dict[tk], (-1, 1)))
        else:
            aux_feat = raw_band[band_ref[suffix]]
            aux_stat = [band_stat[stat](aux_feat) for stat in bs_key]
            city_feature.append(np.transpose(aux_stat))

    # ------------give a summary for recorded features
    if reported:
        feature_summary = open("feature_summary.txt", "w")
        feature_summary.write("*"*10 + "Feature Recorded" + "*"*10 + "\n")
        for suffix in sorted(band_ref.keys()):
            if isinstance(band_ref[suffix], dict):
                # ------------raw band statistics
                for raw_band_name in ["VV", "VH", "R", "G", "B", "NIR"]:
                    for stat in bs_key:
                        feature_summary.write("@".join([suffix, raw_band_name, stat]) + "\n")
                # ------------SFs
                for sf in ["NIR-R", "NIR-G", "NIR-B", "R-G", "R-B", "G-B", "Brightness"]:
                    feature_summary.write("@".join([suffix, sf, "Mean"]) + "\n")
                # ------------MFs
                MF_dict_keys = [b + "-" + op for b in s2_band_use for op in s2_morph_ops]
                for mk in sorted(MF_dict_keys):
                    for i in range(0, len(DMP_size)-1):
                        feature_summary.write("@".join([suffix, mk, str(DMP_size[i+1]) + "-" + str(DMP_size[i])]) + "\n")
                # ------------TFs
                GLCM_dict_keys = [b + "-" + str(s) for b in s2_band_use for s in DMP_size]
                TF_dict_keys = [gk + "@" + func for gk in sorted(GLCM_dict_keys) for func in sorted(texture_func.keys())]
                for tk in sorted(TF_dict_keys):
                    feature_summary.write("@".join([suffix, tk]) + "\n")
            else:
                for stat in bs_key:
                    feature_summary.write("@".join([suffix, stat]) + "\n")

        feature_summary.write("*" * 30 + "\n")

    return city_feature


def GetDataPairFromDataset(dataset_path, target_variable, aggregate_suffix, scale_level=0, saved=True, num_cpu=1, chunk_size=50000, save_suffix=None, aux_namelist=None):
    # ------Experiment Settings
    # scale = 0.0001
    scale = 1.0
    s1_prefix = "sentinel_1_"
    s2_prefix = "sentinel_2_"
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

    height = []
    feature_list = []

    city_feature = {}
    city_band_ref = {}

    for city in city_name:
        n_sample_city = f[city][target_variable].shape[0]
        print("*" * 10, city, "Starts", "*" * 10)
        print(city, n_sample_city, "sample(s)")

        n_chunk = int(np.floor(n_sample_city / chunk_size))
        for chunk_id in range(0, n_chunk + 1):
            id_start = chunk_id * chunk_size
            n_subsample = min(n_sample_city - id_start, chunk_size)

            if n_subsample == 0:
                continue

            id_end = id_start + n_subsample

            print("sampling interval: [{0}, {1})".format(id_start, id_end))

            height_avg = f[city][target_variable][id_start:id_end]
            height.append(height_avg)

            for suffix in aggregate_suffix:
                city_band_ref[suffix] = {}
                # ---------------[1] Sentinel-1-VV
                s1_vv_ref = s1_prefix + suffix + "_" + band_ref["VV"]
                vv_raw = f[city][s1_vv_ref][id_start:id_end, scale_level, :, :]
                s1_vv = get_backscatterCoef(vv_raw)
                city_feature[s1_vv_ref] = s1_vv
                city_band_ref[suffix]["VV"] = s1_vv_ref
                # ---------------[2] Sentinel-1-VH
                s1_vh_ref = s1_prefix + suffix + "_" + band_ref["VH"]
                vh_raw = f[city][s1_vh_ref][id_start:id_end, scale_level, :, :]
                s1_vh = get_backscatterCoef(vh_raw)
                city_feature[s1_vh_ref] = s1_vh
                city_band_ref[suffix]["VH"] = s1_vh_ref
                # ---------------[3] Sentinel-2-R
                s2_r_ref = s2_prefix + suffix + "_" + band_ref["R"]
                s2_r = f[city][s2_r_ref][id_start:id_end, scale_level, :, :] * scale
                city_feature[s2_r_ref] = s2_r
                city_band_ref[suffix]["R"] = s2_r_ref
                # ---------------[4] Sentinel-2-G
                s2_g_ref = s2_prefix + suffix + "_" + band_ref["G"]
                s2_g = f[city][s2_g_ref][id_start:id_end, scale_level, :, :] * scale
                city_feature[s2_g_ref] = s2_g
                city_band_ref[suffix]["G"] = s2_g_ref
                # ---------------[5] Sentinel-2-B
                s2_b_ref = s2_prefix + suffix + "_" + band_ref["B"]
                s2_b = f[city][s2_b_ref][id_start:id_end, scale_level, :, :] * scale
                city_feature[s2_b_ref] = s2_b
                city_band_ref[suffix]["B"] = s2_b_ref
                # ---------------[6] Sentinel-2-NIR
                s2_nir_ref = s2_prefix + suffix + "_" + band_ref["NIR"]
                s2_nir = f[city][s2_nir_ref][id_start:id_end, scale_level, :, :] * scale
                city_feature[s2_nir_ref] = s2_nir
                city_band_ref[suffix]["NIR"] = s2_nir_ref
            
            if aux_namelist is not None:
                for k in aux_namelist:
                    city_feature[k] = f[city][k][id_start:id_end, scale_level, :, :]
                    city_band_ref[k] = k

            res = get_feature(city_feature, city_band_ref, s2_band_use, band_statistics, s2_morph_ops, DMP_size,
                              GLCM_angle, texture_list, rescaled=False, reported=True, num_cpu=num_cpu)

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

    height = np.concatenate(height)

    if saved:
        np.save("feature" + save_suffix + ".npy", feature)
        np.save(target_variable + save_suffix + ".npy", height)

    f.close()

    return feature, height


def GetBackScatterCoefFromDataset(dataset_path, target_variable, aggregate_suffix, scale_level=0, saved=True,
                                  chunk_size=50000, save_suffix=None):
    s1_prefix = "sentinel_1_"
    band_ref = {"VV": "B1", "VH": "B2"}

    f = h5py.File(dataset_path, mode="r")
    city_name = [k for k in f.keys()]

    s1_vv_list = []
    s1_vh_list = []
    height = []

    for city in city_name:
        n_sample_city = f[city][target_variable].shape[0]

        n_chunk = int(np.floor(n_sample_city / chunk_size))
        for chunk_id in range(0, n_chunk + 1):
            id_start = chunk_id * chunk_size
            n_subsample = min(n_sample_city - id_start, chunk_size)
            id_end = id_start + n_subsample

            # ---------get the average height data
            height_avg = f[city][target_variable][id_start:id_end]
            height.append(height_avg)

            # ---------get the raw VV data
            vv_raw = f[city][s1_prefix + aggregate_suffix + "_" + band_ref["VV"]][id_start:id_end]
            # ---------convert to the raw VV observation into the backscatter coefficients
            s1_vv = get_backscatterCoef(vv_raw[:, scale_level, :, :])
            # ---------calculate the mean backscatter coefficients over our patch
            s1_vv_avg = np.mean(s1_vv, axis=(-1, -2))
            s1_vv_list.append(s1_vv_avg)

            vh_raw = f[city][s1_prefix + aggregate_suffix + "_" + band_ref["VH"]][id_start:id_end]
            s1_vh = get_backscatterCoef(vh_raw[:, scale_level, :, :])
            s1_vh_avg = np.mean(s1_vh, axis=(-1, -2))
            s1_vh_list.append(s1_vh_avg)

    s1_vv = np.concatenate(s1_vv_list)
    s1_vh = np.concatenate(s1_vh_list)
    height = np.concatenate(height)
    if saved:
        np.save("s1_vv" + save_suffix + ".npy", s1_vv)
        np.save("s1_vh" + save_suffix + ".npy", s1_vh)
        np.save(target_variable + save_suffix + ".npy", height)

    f.close()

    return s1_vv, s1_vh, height


# ************************* [2] Feature Preprocessing *************************
def feature_selection(feature, target_variable, num_feature_thresold=50, saved=True, suffix=None):
    variance_thresold = 1e-6

    # ------remove features with low variance
    var_filter = fs.VarianceThreshold(threshold=variance_thresold)
    feature_select = var_filter.fit_transform(X=feature)

    # ------rescale features by removing the mean and scaling to unit variance
    rescaler = preprocessing.StandardScaler()
    feature_rescale = rescaler.fit_transform(X=feature_select)

    # ------transform data into lower dimension
    num_feature_thresold = min(num_feature_thresold, feature_rescale.shape[1])
    transformer = decomposition.PCA(n_components=num_feature_thresold)
    feature_transform = transformer.fit_transform(X=feature_rescale)

    # ------select features using Mutual Information criteria
    #k_best_filter = fs.SelectKBest(score_func=fs.mutual_info_regression, k=num_feature_thresold)
    #feature_select = k_best_filter.fit_transform(X=feature_transform, y=target_variable)

    if saved:
        joblib.dump(var_filter, "filter" + suffix + ".pkl")
        joblib.dump(rescaler, "scaler" + suffix + ".pkl")
        joblib.dump(transformer, "transformer" + suffix + ".pkl")
        #joblib.dump(k_best_filter, "selector" + suffix + ".pkl")

    #return feature_select, target_variable
    return feature_transform, target_variable


# ************************* [3] Model Training *************************
def train_vvh_model(training_dataset_path, test_dataset_path, target_variable, aggregate_suffix, scale_level=0,
                    reported=True, saved=True, selected=True, chunk_size=50000, num_cv=1, **kwargs):
    if "save_suffix" in kwargs.keys():
        save_suffix = kwargs["save_suffix"]
    else:
        save_suffix = None

    # ------get paired data for training
    if "training_feature_file" in kwargs.keys() and kwargs["training_feature_file"] is not None:
        feature_train = np.load(kwargs["training_feature_file"])
        height_train = np.load(kwargs["training_target_file"])
        feature_test = np.load(kwargs["test_feature_file"])
        height_test = np.load(kwargs["test_target_file"])
    else:
        train_suffix = "_train" + save_suffix
        s1_vv_train, s1_vh_train, height_train = GetBackScatterCoefFromDataset(training_dataset_path, target_variable,
                                                                               aggregate_suffix, scale_level, saved,
                                                                               chunk_size, save_suffix=train_suffix)
        test_suffix = "_test" + save_suffix
        s1_vv_test, s1_vh_test, height_test = GetBackScatterCoefFromDataset(test_dataset_path, target_variable,
                                                                            aggregate_suffix, scale_level, saved,
                                                                            chunk_size, save_suffix=test_suffix)

        feature_train = np.concatenate((np.expand_dims(s1_vv_train, 1), np.expand_dims(s1_vh_train, 1)), axis=1)
        feature_test = np.concatenate((np.expand_dims(s1_vv_test, 1), np.expand_dims(s1_vh_test, 1)), axis=1)

    height_train = np.log(height_train)
    height_test = np.log(height_test)

    if selected:
        # ------set the sets of VVH parameters to be tested
        parameters = {"gamma": (3.0, 5.0, 7.0, 9.0)}

        # ------GridSearch with cross-validation
        vvh_search = ms.GridSearchCV(estimator=VVH_model(a=-1.0, b=0.1, c=-1.0), param_grid=parameters,
                                     scoring="r2", cv=10, n_jobs=num_cv)
        vvh_search.fit(X=feature_train, y=height_train)

        vvh_best = vvh_search.best_estimator_
        if reported:
            params_set = vvh_search.cv_results_["params"]
            params = sorted(list(params_set[0].keys()))
            score_mean = vvh_search.cv_results_["mean_test_score"]
            score_std = vvh_search.cv_results_["std_test_score"]
            rank = vvh_search.cv_results_["rank_test_score"]
            rank_id = np.argsort(rank)

            print("*" * 10 + "Results of GridSearch with cross-validation" + "*" * 10)
            print("rank\t%s\tmean_test_score\tstd_test_score" % "\t".join(params))
            for param_id in rank_id:
                param_val = [str(params_set[param_id][p]) for p in params]
                print(rank[param_id], "\t", "\t".join(param_val), "\t", score_mean[param_id], "\t", score_std[param_id])

        # ---------VVH model test
        print("*" * 10 + "Test Results of VVH Regression Model" + "*" * 10)
        r2 = vvh_best.evaluate(feature_test, height_test)
        print("R^2 = %.6f" % r2)
        print("*" * 40)

        model_save_path = "VVH_model_" + target_variable + "_best" + save_suffix + ".txt"
        vvh_best.save_model(model_save_path)
    else:
        # ------use default VVH parameters
        gamma = 5.0
        # ---------VVH model construction
        vvh_model_est = VVH_model(gamma=gamma, a=-1.0, b=0.1, c=-1.0)
        # ---------VVH model fitting
        vvh_model_est.fit(feature_train, height_train)

        # ---------VVH model test
        print("*" * 10 + "Test Results of VVH Regression Model" + "*" * 10)
        r2 = vvh_model_est.evaluate(feature_test, height_test)
        print("R^2 = %.6f" % r2)
        print("*" * 40)

        model_save_path = "VVH_model_" + target_variable + save_suffix + ".txt"
        vvh_model_est.save_model(model_save_path)


def train_baggingSVR_model(training_dataset_path, test_dataset_path, target_variable, aggregate_suffix, scale_level=0,
                           reduction_ratio=0.5, reported=True, saved=True, selected=True, evaluated=True,
                           num_cpu=1, num_cv=1, chunk_size=50000, aux_namelist=None, **kwargs):
    if "save_suffix" in kwargs.keys():
        save_suffix = kwargs["save_suffix"]
    else:
        save_suffix = None

    # ------get paired data for training
    if "training_feature_file" in kwargs.keys() and kwargs["training_feature_file"] is not None:
        feature_train = np.load(kwargs["training_feature_file"])
        height_train = np.load(kwargs["training_target_file"])
        feature_test = np.load(kwargs["test_feature_file"])
        height_test = np.load(kwargs["test_target_file"])
    else:
        train_suffix = "_train" + save_suffix
        feature_train, height_train = GetDataPairFromDataset(training_dataset_path, target_variable, aggregate_suffix,
                                                             scale_level, saved, num_cpu, chunk_size,
                                                             save_suffix=train_suffix, aux_namelist=aux_namelist)
        test_suffix = "_test" + save_suffix
        feature_test, height_test = GetDataPairFromDataset(test_dataset_path, target_variable, aggregate_suffix,
                                                           scale_level, saved, num_cpu, chunk_size,
                                                           save_suffix=test_suffix, aux_namelist=aux_namelist)
    height_train = np.log(height_train)
    height_test = np.log(height_test)

    # ------select features
    n_se = 6
    n_feat = int((36 + 60 * n_se) * reduction_ratio)
    select_suffix = "_baggingSVR_n" + str(n_feat) + save_suffix
    feature_train, height_train = feature_selection(feature_train, height_train, num_feature_thresold=n_feat,
                                                    saved=True, suffix=select_suffix)

    baggingSVR_filter = joblib.load("filter" + select_suffix + ".pkl")
    baggingSVR_scaler = joblib.load("scaler" + select_suffix + ".pkl")
    baggingSVR_transformer = joblib.load("transformer" + select_suffix + ".pkl")
    feature_test = baggingSVR_filter.transform(feature_test)
    feature_test = baggingSVR_scaler.transform(feature_test)
    feature_test = baggingSVR_transformer.transform(feature_test)

    if selected:
        # ------set the sets of BaggingSVR parameters to be searched
        parameters = {"n_svr": (50, 100, 200),
                      "max_samples": (0.05, 0.1, 0.2)}

        # ------GridSearch with cross-validation
        baggingSVR_search = ms.GridSearchCV(estimator=BaggingSupportVectorRegressionModel(kernel="rbf", c=1.0,
                                                                                          epsilon=0.1,
                                                                                          n_jobs=int(num_cpu/num_cv)),
                                            param_grid=parameters, scoring="r2", cv=10, n_jobs=num_cv)
        baggingSVR_search.fit(X=feature_train, y=height_train, evaluated=evaluated)

        baggingSVR_best = baggingSVR_search.best_estimator_
        if reported:
            params_set = baggingSVR_search.cv_results_["params"]
            params = sorted(list(params_set[0].keys()))
            score_mean = baggingSVR_search.cv_results_["mean_test_score"]
            score_std = baggingSVR_search.cv_results_["std_test_score"]
            rank = baggingSVR_search.cv_results_["rank_test_score"]
            rank_id = np.argsort(rank)

            print("*" * 10 + "Results of GridSearch with cross-validation" + "*" * 10)
            print("rank\t%s\tmean_test_score\tstd_test_score" % "\t".join(params))
            for param_id in rank_id:
                param_val = [str(params_set[param_id][p]) for p in params]
                print(rank[param_id], "\t", "\t".join(param_val), "\t", score_mean[param_id], "\t", score_std[param_id])

        # ---------BaggingSVR model saving
        model_save_path = "baggingSVR_model_" + target_variable + "_best" + save_suffix + ".pkl"
        baggingSVR_best.save_model(model_save_path)

        # ---------BaggingSVR model test
        print("*" * 10 + "Test Results of Bagging Support Vector Regression Model" + "*" * 10)
        r2 = baggingSVR_best.evaluate(feature_test, height_test)
        print("R^2 = %.6f" % r2)
        print("*" * 40)
    else:
        # ------use default BaggingSVR parameters
        #n_svr = 40
        n_svr = 50
        #max_samples = 0.05
        max_samples = 0.1
        # ---------BaggingSVR model construction
        baggingSVR_est = BaggingSupportVectorRegressionModel(kernel="rbf", c=1.0, epsilon=0.1, n_jobs=num_cpu,
                                                             n_svr=n_svr, max_samples=max_samples)
        # ---------BaggingSVR model fitting
        baggingSVR_est.fit(feature_train, height_train, evaluated=evaluated)

        # ---------BaggingSVR model saving
        model_save_path = "baggingSVR_model_" + target_variable + save_suffix + ".pkl"
        baggingSVR_est.save_model(model_save_path)

        # ---------BaggingSVR model test
        print("*" * 10 + "Test Results of Bagging Support Vector Regression Model" + "*" * 10)
        r2 = baggingSVR_est.evaluate(feature_test, height_test)
        print("R^2 = %.6f" % r2)
        print("*" * 40)


def train_SVR_model(training_dataset_path, test_dataset_path, target_variable, aggregate_suffix, scale_level=0,
                    reduction_ratio=0.5, reported=True, saved=True, selected=True, evaluated=True,
                    num_cpu=1, num_cv=1, chunk_size=50000, aux_namelist=None, **kwargs):
    if "save_suffix" in kwargs.keys():
        save_suffix = kwargs["save_suffix"]
    else:
        save_suffix = None
    # ------get paired data for training
    if "training_feature_file" in kwargs.keys() and kwargs["training_feature_file"] is not None:
        feature_train = np.load(kwargs["training_feature_file"])
        height_train = np.load(kwargs["training_target_file"])
        feature_test = np.load(kwargs["test_feature_file"])
        height_test = np.load(kwargs["test_target_file"])
    else:
        train_suffix = "_train" + save_suffix
        feature_train, height_train = GetDataPairFromDataset(training_dataset_path, target_variable, aggregate_suffix,
                                                             scale_level, saved, num_cpu, chunk_size,
                                                             save_suffix=train_suffix, aux_namelist=aux_namelist)
        test_suffix = "_test" + save_suffix
        feature_test, height_test = GetDataPairFromDataset(test_dataset_path, target_variable, aggregate_suffix,
                                                           scale_level, saved, num_cpu, chunk_size,
                                                           save_suffix=test_suffix, aux_namelist=aux_namelist)
    height_train = np.log(height_train)
    height_test = np.log(height_test)

    # ------select features
    n_se = 6
    n_feat = int((36 + 60 * n_se) * reduction_ratio)
    select_suffix = "_SVR_n" + str(n_feat) + save_suffix
    feature_train, height_train = feature_selection(feature_train, height_train, num_feature_thresold=n_feat,
                                                    saved=True, suffix=select_suffix)

    SVR_filter = joblib.load("filter" + select_suffix + ".pkl")
    SVR_scaler = joblib.load("scaler" + select_suffix + ".pkl")
    SVR_transformer = joblib.load("transformer" + select_suffix + ".pkl")
    feature_test = SVR_filter.transform(feature_test)
    feature_test = SVR_scaler.transform(feature_test)
    feature_test = SVR_transformer.transform(feature_test)

    if selected:
        # ------set the sets of BaggingSVR parameters to be searched
        parameters = {"c": (1, 10, 100, 1000),
                      "epsilon": (0.1, 1.0, 2.0, 5.0)}

        # ------GridSearch with cross-validation
        SVR_search = ms.GridSearchCV(estimator=SupportVectorRegressionModel(kernel="rbf"),
                                     param_grid=parameters, scoring="r2", cv=10, n_jobs=num_cv)
        SVR_search.fit(X=feature_train, y=height_train, evaluated=evaluated)

        SVR_best = SVR_search.best_estimator_
        if reported:
            params_set = SVR_search.cv_results_["params"]
            params = sorted(list(params_set[0].keys()))
            score_mean = SVR_search.cv_results_["mean_test_score"]
            score_std = SVR_search.cv_results_["std_test_score"]
            rank = SVR_search.cv_results_["rank_test_score"]
            rank_id = np.argsort(rank)

            print("*" * 10 + "Results of GridSearch with cross-validation" + "*" * 10)
            print("rank\t%s\tmean_test_score\tstd_test_score" % "\t".join(params))
            for param_id in rank_id:
                param_val = [str(params_set[param_id][p]) for p in params]
                print(rank[param_id], "\t", "\t".join(param_val), "\t", score_mean[param_id], "\t", score_std[param_id])

        # ---------SVR model saving
        model_save_path = "SVR_model_" + target_variable + "_best" + save_suffix + ".pkl"
        SVR_best.save_model(model_save_path)

        # ---------SVR model test
        print("*" * 10 + "Test Results of Support Vector Regression Model" + "*" * 10)
        r2 = SVR_best.evaluate(feature_test, height_test)
        print("R^2 = %.6f" % r2)
        print("*" * 40)
    else:
        # ------use default SVR parameters
        kernel = "rbf"
        c = 1.0
        epsilon = 0.1
        # ---------SVR model construction
        SVR_est = SupportVectorRegressionModel(kernel=kernel, c=c, epsilon=epsilon)
        # ---------SVR model fitting
        SVR_est.fit(feature_train, height_train, evaluated=evaluated)
        # ---------SVR model saving
        model_save_path = "SVR_model_" + target_variable + save_suffix + ".pkl"
        SVR_est.save_model(model_save_path)
        # ---------SVR model test
        print("*" * 10 + "Test Results of Support Vector Regression Model" + "*" * 10)
        r2 = SVR_est.evaluate(feature_test, height_test)
        print("R^2 = %.6f" % r2)
        print("*" * 40)


def train_RF_model(training_dataset_path, test_dataset_path, target_variable, aggregate_suffix, scale_level=0,
                   reduction_ratio=0.5, reported=True, saved=True, selected=True, evaluated=True,
                   num_cpu=1, num_cv=1, chunk_size=50000, aux_namelist=None, **kwargs):
    if "save_suffix" in kwargs.keys():
        save_suffix = kwargs["save_suffix"]
    else:
        save_suffix = None
    # ------get paired data for training
    if "training_feature_file" in kwargs.keys() and kwargs["training_feature_file"] is not None:
        feature_train = np.load(kwargs["training_feature_file"])
        height_train = np.load(kwargs["training_target_file"])
        feature_test = np.load(kwargs["test_feature_file"])
        height_test = np.load(kwargs["test_target_file"])
    else:
        train_suffix = "_train" + save_suffix
        feature_train, height_train = GetDataPairFromDataset(training_dataset_path, target_variable, aggregate_suffix,
                                                             scale_level, saved, num_cpu, chunk_size,
                                                             save_suffix=train_suffix, aux_namelist=aux_namelist)
        test_suffix = "_test" + save_suffix
        feature_test, height_test = GetDataPairFromDataset(test_dataset_path, target_variable, aggregate_suffix,
                                                           scale_level, saved, num_cpu, chunk_size,
                                                           save_suffix=test_suffix, aux_namelist=aux_namelist)
    height_train = np.log(height_train)
    height_test = np.log(height_test)

    # ------select features
    n_se = 6
    n_feat = int((36 + 60 * n_se) * reduction_ratio)
    select_suffix = "_rf_n" + str(n_feat) + save_suffix
    feature_train, height_train = feature_selection(feature_train, height_train, num_feature_thresold=n_feat,
                                                    saved=True, suffix=select_suffix)

    rf_filter = joblib.load("filter" + select_suffix + ".pkl")
    rf_scaler = joblib.load("scaler" + select_suffix + ".pkl")
    rf_transformer = joblib.load("transformer" + select_suffix + ".pkl")
    feature_test = rf_filter.transform(feature_test)
    feature_test = rf_scaler.transform(feature_test)
    feature_test = rf_transformer.transform(feature_test)

    if selected:
        # ------set the sets of RF parameters to be tested
        parameters = {"n_tree": (100, 250, 500, 1000),
                      "max_depth": (5, 7, 9, 11)}

        # ------GridSearch with cross-validation
        rf_search = ms.GridSearchCV(estimator=RandomForestModel(n_jobs=int(num_cpu/num_cv)), param_grid=parameters,
                                    scoring="r2", cv=10, n_jobs=num_cv)
        rf_search.fit(X=feature_train, y=height_train, evaluated=evaluated)

        rf_best = rf_search.best_estimator_
        if reported:
            params_set = rf_search.cv_results_["params"]
            params = sorted(list(params_set[0].keys()))
            score_mean = rf_search.cv_results_["mean_test_score"]
            score_std = rf_search.cv_results_["std_test_score"]
            rank = rf_search.cv_results_["rank_test_score"]
            rank_id = np.argsort(rank)

            print("*" * 10 + "Results of GridSearch with cross-validation" + "*" * 10)
            print("rank\t%s\tmean_test_score\tstd_test_score" % "\t".join(params))
            for param_id in rank_id:
                param_val = [str(params_set[param_id][p]) for p in params]
                print(rank[param_id], "\t", "\t".join(param_val), "\t", score_mean[param_id], "\t", score_std[param_id])

        # ---------RF model test
        print("*" * 10 + "Test Results of Random Forest Regression Model" + "*" * 10)
        r2 = rf_best.evaluate(feature_test, height_test)
        print("R^2 = %.6f" % r2)
        print("*" * 40)

        model_save_path = "rf_model_" + target_variable + "_best" + save_suffix + ".pkl"
        rf_best.save_model(model_save_path)
    else:
        # ------use default RF parameters
        n_tree = 500
        max_depth = 9
        n_jobs = num_cpu
        # ---------RF model construction
        rf_est = RandomForestModel(n_tree=n_tree, max_depth=max_depth, n_jobs=n_jobs)
        # ---------RF model fitting
        rf_est.fit(feature_train, height_train, evaluated=evaluated)
        # ---------RF model test
        print("*" * 10 + "Test Results of Random Forest Regression Model" + "*" * 10)
        r2 = rf_est.evaluate(feature_test, height_test)
        print("R^2 = %.6f" % r2)
        print("*" * 40)

        model_save_path = "rf_model_" + target_variable + save_suffix + ".pkl"
        rf_est.save_model(model_save_path)


def train_XGBoost_model(training_dataset_path, test_dataset_path, target_variable, aggregate_suffix, scale_level=0,
                        reduction_ratio=0.5, reported=True, saved=True, selected=True, evaluated=True,
                        num_cpu=1, num_cv=1, chunk_size=50000, aux_namelist=None, **kwargs):
    if "save_suffix" in kwargs.keys():
        save_suffix = kwargs["save_suffix"]
    else:
        save_suffix = None
    # ------get paired data for training
    if "training_feature_file" in kwargs.keys() and kwargs["training_feature_file"] is not None:
        feature_train = np.load(kwargs["training_feature_file"])
        height_train = np.load(kwargs["training_target_file"])
        feature_test = np.load(kwargs["test_feature_file"])
        height_test = np.load(kwargs["test_target_file"])
    else:
        train_suffix = "_train" + save_suffix
        feature_train, height_train = GetDataPairFromDataset(training_dataset_path, target_variable, aggregate_suffix,
                                                             scale_level, saved, num_cpu, chunk_size,
                                                             save_suffix=train_suffix, aux_namelist=aux_namelist)
        test_suffix = "_test" + save_suffix
        feature_test, height_test = GetDataPairFromDataset(test_dataset_path, target_variable, aggregate_suffix,
                                                           scale_level, saved, num_cpu, chunk_size,
                                                           save_suffix=test_suffix, aux_namelist=aux_namelist)
    height_train = np.log(height_train)
    height_test = np.log(height_test)

    # ------select features
    n_se = 6
    n_feat = int((36 + 60 * n_se) * reduction_ratio)
    select_suffix = "_xgb_n" + str(n_feat) + save_suffix
    feature_train, height_train = feature_selection(feature_train, height_train, num_feature_thresold=n_feat,
                                                    saved=True, suffix=select_suffix)

    xgb_filter = joblib.load("filter" + select_suffix + ".pkl")
    xgb_scaler = joblib.load("scaler" + select_suffix + ".pkl")
    xgb_transformer = joblib.load("transformer" + select_suffix + ".pkl")
    feature_test = xgb_filter.transform(feature_test)
    feature_test = xgb_scaler.transform(feature_test)
    feature_test = xgb_transformer.transform(feature_test)

    if selected:
        # ------set the sets of XGBoost parameters to be tested
        # ---------at first, we search on parameters_s1 and then using the best n_estimators and max_depth
        # ---------as the model corresponding hyper-parameters when searching on parameters_s2
        parameters_s1 = {"n_estimators": (100, 250, 500, 1000),
                         "max_depth": (5, 7, 9, 11)}

        '''
        parameters_s2 = {"reg_lambda": (0.1, 1.0, 10.0),
                         "gamma": (0, 0.1, 0.2)}
        '''
        # ------GridSearch with cross-validation
        xgb_search = ms.GridSearchCV(estimator=XGBoostRegressionModel(n_jobs=int(num_cpu/num_cv)), param_grid=parameters_s1,
                                     scoring="r2", cv=10, n_jobs=num_cv)
        xgb_search.fit(X=feature_train, y=height_train, evaluated=evaluated)

        xgb_best = xgb_search.best_estimator_
        if reported:
            params_set = xgb_search.cv_results_["params"]
            params = sorted(list(params_set[0].keys()))
            score_mean = xgb_search.cv_results_["mean_test_score"]
            score_std = xgb_search.cv_results_["std_test_score"]
            rank = xgb_search.cv_results_["rank_test_score"]
            rank_id = np.argsort(rank)

            print("*" * 10 + "Results of GridSearch with cross-validation" + "*" * 10)
            print("rank\t%s\tmean_test_score\tstd_test_score" % "\t".join(params))
            for param_id in rank_id:
                param_val = [str(params_set[param_id][p]) for p in params]
                print(rank[param_id], "\t", "\t".join(param_val), "\t", score_mean[param_id], "\t", score_std[param_id])

        # ---------XGBoost model test
        print("*" * 10 + "Test Results of XGBoost Regression Model" + "*" * 10)
        r2 = xgb_best.evaluate(feature_test, height_test)
        print("R^2 = %.6f" % r2)
        print("*" * 40)

        model_save_path = "xgb_model_" + target_variable + "_best" + save_suffix + ".pkl"
        xgb_best.save_model(model_save_path)
    else:
        # ------use default XGBoost parameters
        n_estimators = 500
        max_depth = 9
        gamma = 0.1
        reg_lambda = 1.0
        n_jobs = num_cpu
        # ---------XGBoost model construction
        xgb_est = XGBoostRegressionModel(n_estimators=n_estimators, max_depth=max_depth, gamma=gamma,
                                         reg_lambda=reg_lambda, n_jobs=n_jobs)
        # ---------XGBoost model fitting
        xgb_est.fit(feature_train, height_train, evaluated=evaluated)
        # ---------XGBoost model test
        print("*" * 10 + "Test Results of XGBoost Regression Model" + "*" * 10)
        r2 = xgb_est.evaluate(feature_test, height_test)
        print("R^2 = %.6f" % r2)
        print("*" * 40)

        model_save_path = "xgb_model_" + target_variable + save_suffix + ".pkl"
        xgb_est.save_model(model_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traditional ML Model Training")
    parser.add_argument("--model", type=str, default="RF", choices=["VVH", "baggingSVR", "RF", "XGBoost", "SVR"],
                        help="Traditional ML model for building information extraction (default: XGBoost)")

    # dataset params
    parser.add_argument("--aggregation", type=str, default="50pt", choices=["avg", "25pt", "50pt", "75pt"],
                        help="temporal aggregation operation used for Sentinel-1/Sentinel-2 image reduction in one year (separated with `,`)")
    parser.add_argument("--target", type=str, default="BuildingHeight", choices=["BuildingHeight", "BuildingFootprint"],
                        help="target variable name for model prediction")
    parser.add_argument("--training_dataset", type=str,
                        default="dataset/patch_data_50pt_s15_100m_train_out.h5",
                        help="path of the training dataset")
    parser.add_argument("--test_dataset", type=str,
                        default="dataset/patch_data_50pt_s15_sample_valid.h5",
                        help="path of the test dataset")
    parser.add_argument("--aux_feature", type=str, 
                        default=None, help="comma-separated namelist of auxiliary features (e.g. DEM) for prediction")

    # model training params
    parser.add_argument("--reduction_ratio", type=float, default=0.5,
                        help="reduction ratio used for feature reduction before ML regression (default: 0.5)")
    parser.add_argument("--chunk_size", type=int, default=50000,
                        help="number of samples used for feature computation once (default: 50000)")
    parser.add_argument("--num_cpu", type=int, default=1, help="number of CPUs used for parallel computing (default: 4)")
    parser.add_argument("--num_cv", type=int, default=1,
                        help="number of CPUs used for parallel cross-validation (default: 1)")
    parser.add_argument("--saved", type=str, default="True",
                        help="flag to control whether the intermediate results are saved")
    parser.add_argument("--selected", type=str, default="True",
                        help="flag to control whether the GridSearch is used for optimal hyper-parameters search")
    parser.add_argument("--train_with_evaluation", type=str, default="True",
                        help="flag to control whether evaluation is performed on the training set, which may slows down the training process")
    parser.add_argument("--save_suffix", type=str, default="_s15_100m", help="save suffix for training record files")

    # cache params
    parser.add_argument("--training_feature_file", type=str, default=None,
                        help="pre-computed training feature .npy file")
    parser.add_argument("--training_target_file", type=str, default=None,
                        help="pre-computed training target .npy file corresponding with pre-computed feature .npy file")
    parser.add_argument("--test_feature_file", type=str, default=None,
                        help="pre-computed test feature .npy file")
    parser.add_argument("--test_target_file", type=str, default=None,
                        help="pre-computed test target .npy file corresponding with pre-computed feature .npy file")

    args = parser.parse_args()

    if args.saved in ["True"]:
        saved = True
    else:
        saved = False

    if args.selected in ["True"]:
        selected = True
    else:
        selected = False

    if args.train_with_evaluation in ["True"]:
        train_with_evaluation = True
    else:
        train_with_evaluation = False

    if args.aux_feature is not None:
        args.aux_feature = [s for s in args.aux_feature.split(",")]

    if args.model == "VVH":
        train_vvh_model(training_dataset_path=args.training_dataset, test_dataset_path=args.test_dataset,
                        target_variable=args.target, aggregate_suffix=args.aggregation,
                        saved=saved, selected=selected, save_suffix=args.save_suffix, chunk_size=args.chunk_size,
                        training_feature_file=args.training_feature_file, num_cv=args.num_cv,
                        training_target_file=args.training_target_file,
                        test_feature_file=args.test_feature_file, test_target_file=args.test_target_file)
    elif args.model == "baggingSVR":
        aggregate_list = args.aggregation.split(',')
        train_baggingSVR_model(training_dataset_path=args.training_dataset, test_dataset_path=args.test_dataset,
                               target_variable=args.target, aggregate_suffix=aggregate_list,
                               reduction_ratio=args.reduction_ratio,
                               saved=saved, selected=selected, evaluated=train_with_evaluation,
                               num_cpu=args.num_cpu, chunk_size=args.chunk_size,
                               aux_namelist=args.aux_feature,
                               save_suffix=args.save_suffix, num_cv=args.num_cv,
                               training_feature_file=args.training_feature_file,
                               training_target_file=args.training_target_file,
                               test_feature_file=args.test_feature_file, test_target_file=args.test_target_file)
    elif args.model == "SVR":
        aggregate_list = args.aggregation.split(',')
        train_SVR_model(training_dataset_path=args.training_dataset, test_dataset_path=args.test_dataset,
                        target_variable=args.target, aggregate_suffix=aggregate_list,
                        reduction_ratio=args.reduction_ratio,
                        saved=saved, selected=selected, evaluated=train_with_evaluation,
                        num_cpu=args.num_cpu, chunk_size=args.chunk_size,
                        aux_namelist=args.aux_feature,
                        save_suffix=args.save_suffix, num_cv=args.num_cv,
                        training_feature_file=args.training_feature_file,
                        training_target_file=args.training_target_file,
                        test_feature_file=args.test_feature_file, test_target_file=args.test_target_file)
    elif args.model == "RF":
        aggregate_list = args.aggregation.split(',')
        train_RF_model(training_dataset_path=args.training_dataset, test_dataset_path=args.test_dataset,
                       target_variable=args.target, aggregate_suffix=aggregate_list,
                       reduction_ratio=args.reduction_ratio,
                       saved=saved, selected=selected, evaluated=train_with_evaluation,
                       num_cpu=args.num_cpu, chunk_size=args.chunk_size,
                       aux_namelist=args.aux_feature,
                       save_suffix=args.save_suffix, num_cv=args.num_cv,
                       training_feature_file=args.training_feature_file,
                       training_target_file=args.training_target_file,
                       test_feature_file=args.test_feature_file, test_target_file=args.test_target_file)
    elif args.model == "XGBoost":
        aggregate_list = args.aggregation.split(',')
        train_XGBoost_model(training_dataset_path=args.training_dataset, test_dataset_path=args.test_dataset,
                            target_variable=args.target, aggregate_suffix=aggregate_list,
                            reduction_ratio=args.reduction_ratio,
                            saved=saved, selected=selected, evaluated=train_with_evaluation,
                            num_cpu=args.num_cpu, chunk_size=args.chunk_size,
                            aux_namelist=args.aux_feature,
                            save_suffix=args.save_suffix, num_cv=args.num_cv,
                            training_feature_file=args.training_feature_file,
                            training_target_file=args.training_target_file,
                            test_feature_file=args.test_feature_file, test_target_file=args.test_target_file)
    else:
        raise NotImplementedError
