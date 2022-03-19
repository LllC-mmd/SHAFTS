import os
import torch
from shaft.inference import pred_height_from_tiff_DL_patch, pred_height_from_tiff_DL_patch_MTL


if __name__ == "__main__":
    target_res_mapping = {100: 0.0009, 250: 0.00225, 500: 0.0045, 1000: 0.009}      # map target resolution in meters to degrees

    # ---define some variables for the path look-ups (optional)
    tmp_dir = "tmp"           # path of a directory for temporary results saving during prediction
    pt_prefix = "DL_run"      # path prefix for pretrained models
    case_prefix = "testCase"  # path prefix for input data and results of the case
    backbone = "senet"        # the backbone of prediction models

    # ---specify the lower bound and upper bound of building height prediction by `h_min` and `h_max`, respectively.
    h_min = 2.0
    h_max = 1000.0
    # ---specify the lower bound and upper bound of building footprint prediction by `f_min` and `f_max`, respectively.
    f_min = 0.0
    f_max = 1.0

    # ---specify the settings of cases
    target_resolution = 100                                               # target resolution for building height and footprint mapping 
    target_extent = [-4.4786000, 55.6759000, -3.8828000, 56.0197000]      # target extent for building height and footprint mapping 

    # ------specify the path of input Sentinel data (note: please use the following format for input data specification)
    s1_key = "sentinel_1"       # key which indicates the path of Sentinel-1's files
    s2_key = "sentinel_2"       # key which indicates the path of Sentinel-2's files

    input_img = {
        "50pt": {  # use annual medians as aggregation operation for one year data
            s1_key: os.path.join(case_prefix, "infer_test_Glasgow", "raw_data", "Glasgow_2020_sentinel_1.tif"),      # path of input Sentinel-1 image 
            s2_key: os.path.join(case_prefix, "infer_test_Glasgow", "raw_data", "Glasgow_2020_sentinel_2.tif"),      # path of input Sentinel-2 image
        }
    }
    # ------specify the path of input auxiliary SRTM data (note: please use the following format for input data specification)
    aux_data = {
        "DEM": {
            "path": os.path.join(case_prefix, "infer_test_Glasgow", "raw_data", "Glasgow_srtm.tif"),      # path of input SRTM data
            "patch_size_ratio": 1.0,      # patch size ratio between auxiliary SRTM data and Sentinel data (note: pretrained model offered by SHAFTS uses 1.0 for this parameter)
        }
    }

    # ---specify the information of pretrained models
    model = "SEResNet18"            # name of pretrained models
    pretrained_model_path = os.path.join("check_pt_{0}_100m_MTL".format(backbone), "experiment_1", "checkpoint.pth.tar")  # path of files of pretrained models
    input_patch_size = [20]         # size of input sizes required by pretrained models
    
    # ---specify the common settings of prediction
    padding = 0.03                                  # padding size outside the target region (it is recommended that padding should not be smaller than 0.03)
    cuda_used = torch.cuda.is_available()           # check whether CUDA can be used for prediction
    batch_size = 64                                 # batch size for prediction (default: 64)

    # ---specify the information of output files
    output_prefix = "Glasgow"
    output_dir = os.path.join(case_prefix, "infer_test_Glasgow", "100m")
    output_footprint_file = "_".join([output_prefix, "footprint", backbone + "_MTL"]) + ".tif"
    output_footprint_path = os.path.join(output_dir, output_footprint_file)                     # path of output building footprint files
    output_height_file = "_".join([output_prefix, "height", backbone + "_MTL"]) + ".tif"
    output_height_path = os.path.join(output_dir, output_height_file)                           # path of output building height files

    # ---start our prediction
    pred_height_from_tiff_DL_patch_MTL(extent=target_extent, out_footprint_file=output_footprint_path, out_height_file=output_height_path, 
                                            tif_ref=input_img, patch_size=input_patch_size,
                                            predictor=model, trained_record=pretrained_model_path, resolution=target_res_mapping[target_resolution],
                                            s1_key=s1_key, s2_key=s2_key,
                                            aux_feat_info=aux_data, crossed=False, base_dir=tmp_dir, padding=padding, 
                                            batch_size=batch_size, tmp_suffix=None, log_scale=False,
                                            cuda_used=cuda_used, 
                                            h_min=h_min, h_max=h_max,
                                            f_min=f_min, f_max=f_max)