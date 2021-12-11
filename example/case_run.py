import os
import torch
import shaft
from shaft.inference import pred_height_from_tiff_DL_patch, pred_height_from_tiff_DL_patch_MTL


if __name__ == "__main__":
    var_ref = {
        "height": {
            "min": 2.0,
            "max": 1000.0,
            "activation": "relu"
        },
        "footprint": {
            "min": 0.0,
            "max": 1.0,
            "activation": "sigmoid"
        }
    }

    s1_key = "sentinel_1"
    s2_key = "sentinel_2"
    base_dir = "tmp"

    # ---specify the settings of cases
    case_prefix = "testCase"
    case_loc = {
        # ------case of Glasgow
        "Glasgow": {
            "extent": {
                100: [-4.4786000, 55.6759000, -3.8828000, 56.0197000],
                250: [-4.4786000, 55.6754500, -3.8823500, 56.0197000],
                500: [-4.4786000, 55.6777000, -3.8846000, 56.0197000],
                1000: [-4.4786000, 55.6777000, -3.8846000, 56.0197000]
            },
            "raw_data": {
                "50pt": {
                    s1_key: os.path.join(case_prefix, "infer_test_Glasgow", "raw_data", "Glasgow_2020_sentinel_1.tif"),
                    s2_key: os.path.join(case_prefix, "infer_test_Glasgow", "raw_data", "Glasgow_2020_sentinel_2.tif"),
                }
            },
            "aux_feat": {
                "DEM": {
                    "path": os.path.join(case_prefix, "infer_test_Glasgow", "raw_data", "Glasgow_srtm.tif"),
                    "patch_size_ratio": 1.0,
                } 
            },
            "output_prefix": "Glasgow",
        },
        # ------case of Beijing
        "Beijing": {
            "extent": {
                1000: [116.204, 39.823, 116.575, 40.038]
            },
            "raw_data": {
                "50pt": {
                    s1_key: os.path.join(case_prefix, "infer_test_Beijing", "raw_data", "BeijingC6_2020_sentinel_1_50pt.tif"),
                    s2_key: os.path.join(case_prefix, "infer_test_Beijing", "raw_data", "BeijingC6_2020_sentinel_2_50pt.tif"),
                }
            },
            "aux_feat": {
                "DEM": {
                    "path": os.path.join(case_prefix, "infer_test_Beijing", "raw_data", "Beijing_srtm.tif"),
                    "patch_size_ratio": 1.0,
                } 
            },
            "output_prefix": "Beijing",
        },
        # ------case of Chicago
        "Beijing": {
            "extent": {
                1000: [-87.740, 41.733, -87.545, 41.996]
            },
            "raw_data": {
                "50pt": {
                    s1_key: os.path.join(case_prefix, "infer_test_Chicago", "raw_data", "Chicago_2018_sentinel_1_50pt.tif"),
                    s2_key: os.path.join(case_prefix, "infer_test_Chicago", "raw_data", "Chicago_2018_sentinel_2_50pt.tif"),
                }
            },
            "aux_feat": {
                "DEM": {
                    "path": os.path.join(case_prefix, "infer_test_Chicago", "raw_data", "Chicago_srtm.tif"),
                    "patch_size_ratio": 1.0,
                } 
            },
            "output_prefix": "Chicago",
        },
        # ------case of LosAngeles
        "LosAngeles": {
            "extent": {
                100: [-118.4050, 33.8790, -118.0000, 34.1750],
                250: [-118.4050, 33.8790, -118.0000, 34.1750],
                500: [-118.4050, 33.8790, -118.0000, 34.1750],
                1000: [-118.4050, 33.8790, -118.0000, 34.1750]
            },
            "raw_data": {
                "50pt": {
                    s1_key: os.path.join(case_prefix, "infer_test_LosAngeles", "raw_data", "LosAngeles_2018_sentinel_1.tif"),
                    s2_key: os.path.join(case_prefix, "infer_test_LosAngeles", "raw_data", "LosAngeles_2018_sentinel_2.tif"),
                }
            },
            "aux_feat": {
                "DEM": {
                    "path": os.path.join(case_prefix, "infer_test_LosAngeles", "raw_data", "LosAngeles_srtm.tif"),
                    "patch_size_ratio": 1.0,
                } 
            },
            "output_prefix": "LosAngeles",
        },
    }

    # ---specify the information of pretrained models
    pt_prefix = "dl-models"
    backbone = "senet"
    model = "SEResNet18"

    trained_record = {
        "STL": {
            100: os.path.join("check_pt_{0}_100m".format(backbone), "experiment_0", "checkpoint.pth.tar"), 
            250: os.path.join("check_pt_{0}_250m".format(backbone), "experiment_0", "checkpoint.pth.tar"), 
            500: os.path.join("check_pt_{0}_500m".format(backbone), "experiment_0", "checkpoint.pth.tar"), 
            1000: os.path.join("check_pt_{0}_1000m".format(backbone), "experiment_0", "checkpoint.pth.tar")
        },
        "MTL": {
            100: os.path.join("check_pt_{0}_100m_MTL".format(backbone), "experiment_1", "checkpoint.pth.tar"), 
            250: os.path.join("check_pt_{0}_250m_MTL".format(backbone), "experiment_1", "checkpoint.pth.tar"), 
            500: os.path.join("check_pt_{0}_500m_MTL".format(backbone), "experiment_1", "checkpoint.pth.tar"), 
            1000: os.path.join("check_pt_{0}_1000m_MTL".format(backbone), "experiment_1", "checkpoint.pth.tar")
        }
    }


    input_size = {100: [15], 250: [30], 500: [60], 1000: [120]}
    res = {100: 0.0009, 250: 0.00225, 500: 0.0045, 1000: 0.009}
    padding = 0.0005
    tmp_dir = "tmp"
    cuda_used = torch.cuda.is_available()
    batch_size = 64

    for loc in case_loc.keys():
        input_ref = case_loc[loc]["raw_data"]
        aux_feat_info = case_loc[loc]["aux_feat"]

        for target_res, extent in case_loc[loc]["extent"]:
            input_size = input_size[target_res]
            output_res = res[target_res]

            # ------do inference by STL models
            for target_var in ["height", "footprint"]:
                pt_path = os.path.join(pt_prefix, target_var, trained_record["stl"][target_res])
                output_dir = os.path.join("./infer_test_{0}".format(loc), str(target_res) + "m")
                output_file = "_".join(case_loc[loc]["output_prefix"], target_var, backbone) + ".tif"
                output_path = os.path.join(output_dir, output_file)

                pred_height_from_tiff_DL_patch(extent=extent, out_file=output_path, tif_ref=input_ref, patch_size=input_size,
                                                predictor=model, trained_record=pt_path, resolution=output_res,
                                                s1_key=s1_key, s2_key=s2_key,
                                                aux_feat_info=aux_feat_info, base_dir=tmp_dir, padding=padding, 
                                                batch_size=batch_size, tmp_suffix=None, activation=var_ref[target_var]["activation"], 
                                                log_scale=False, cuda_used=cuda_used, 
                                                v_min=var_ref[target_var]["min"], v_max=var_ref[target_var]["max"])

            # ------do inference by MTL models
            pt_path = os.path.join(pt_prefix, "height", trained_record["mtl"][target_res])
            output_dir = os.path.join("./infer_test_{0}".format(loc), str(target_res) + "m")
            output_footprint_file = "_".join(case_loc[loc]["output_prefix"], "footprint", backbone + "_MTL") + ".tif"
            output_footprint_path = os.path.join(output_dir, output_footprint_file)
            output_height_file = "_".join(case_loc[loc]["output_prefix"], "height", backbone + "_MTL") + ".tif"
            output_height_path = os.path.join(output_dir, output_height_file)

            pred_height_from_tiff_DL_patch_MTL(extent=extent, output_footprint_file=output_footprint_path, out_height_file=output_height_path, 
                                                tif_ref=input_ref, patch_size=input_size,
                                                predictor=model, trained_record=pt_path, resolution=output_res,
                                                s1_key=s1_key, s2_key=s2_key,
                                                aux_feat_info=aux_feat_info, crossed=False, base_dir=tmp_dir, padding=padding, 
                                                batch_size=batch_size, tmp_suffix=None, log_scale=False,
                                                cuda_used=cuda_used, 
                                                h_min=var_ref["height"]["min"], h_max=var_ref["height"]["max"],
                                                f_min=var_ref["footprint"]["min"], f_max=var_ref["footprint"]["max"])
