import os
import shutil
from unittest import TestCase
import warnings
import torch
import shafts


class TestShaft(TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=ImportWarning)

        self.target_var = "height"
        self.v_min = 0.0
        self.v_max = 1000.0
        self.activation = "relu"

        self.s1_key="sentinel_1"
        self.s2_key="sentinel_2"
        self.input_size = {100: [15], 250: [30], 500: [60], 1000: [120]}
        self.res = {100: 0.0009, 250: 0.00225, 500: 0.0045, 1000: 0.009}
        self.padding = 0.03
        
        self.backbone = "senet"
        self.model = "SEResNet18"
        self.batch_size = 2

        self.path_prefix = os.path.join(shafts._path_shaft_module, "dl-models")
        self.stl_trained_record = {100: os.path.join(self.path_prefix, self.target_var, "check_pt_{0}_100m".format(self.backbone), "experiment_0", "checkpoint.pth.tar"), 
                                    250: os.path.join(self.path_prefix, self.target_var, "check_pt_{0}_250m".format(self.backbone), "experiment_0", "checkpoint.pth.tar"), 
                                    500: os.path.join(self.path_prefix, self.target_var, "check_pt_{0}_500m".format(self.backbone), "experiment_0", "checkpoint.pth.tar"), 
                                    1000: os.path.join(self.path_prefix, self.target_var, "check_pt_{0}_1000m".format(self.backbone), "experiment_0", "checkpoint.pth.tar")}
        self.mtl_trained_record = {100: os.path.join(self.path_prefix, self.target_var, "check_pt_{0}_100m_MTL".format(self.backbone), "experiment_1", "checkpoint.pth.tar"), 
                                    250: os.path.join(self.path_prefix, self.target_var, "check_pt_{0}_250m_MTL".format(self.backbone), "experiment_1", "checkpoint.pth.tar"), 
                                    500: os.path.join(self.path_prefix, self.target_var, "check_pt_{0}_500m_MTL".format(self.backbone), "experiment_1", "checkpoint.pth.tar"), 
                                    1000: os.path.join(self.path_prefix, self.target_var, "check_pt_{0}_1000m_MTL".format(self.backbone), "experiment_1", "checkpoint.pth.tar")}

        self.tmp_dir = "tmp"
        self.cuda_used = torch.cuda.is_available()

    # ------test if STL models can run
    def test_inference_STL(self):
        # ------specify the sample case
        sample_loc = {
            "THU": {
                "extent": [116.31916, 40.00171, 116.32078, 40.00189],
                "raw_data": {
                    "50pt": {
                        self.s1_key: os.path.join(shafts._path_shaft_module, "testCase", "infer_test_THU/raw_data/THU_2020_sentinel_1_50pt.tif"),
                        self.s2_key: os.path.join(shafts._path_shaft_module, "testCase", "infer_test_THU/raw_data/THU_2020_sentinel_2_50pt.tif"),
                    }
                },
                "aux_feat": {
                    "DEM": {
                        "path": os.path.join(shafts._path_shaft_module, "testCase", "infer_test_THU/raw_data/THU_srtm.tif"),
                        "patch_size_ratio": 1.0,
                    } 
                },
                "output_prefix": "THU",
            }
        }

        # ------specify the suffix of output files
        output_suffix = self.backbone

        if not os.path.isdir(self.tmp_dir):
            os.mkdir(self.tmp_dir)
        
        test_flag = []

        for loc in sample_loc.keys():
            extent = sample_loc[loc]["extent"]
            input_ref = sample_loc[loc]["raw_data"]
            aux_feat_info = sample_loc[loc]["aux_feat"]
            #for target_res in [100, 250, 500, 1000]:
            for target_res in [100]:
                input_size = self.input_size[target_res]
                output_res = self.res[target_res]
                pt_path = self.stl_trained_record[target_res]

                output_dir = os.path.join(shafts._path_shaft_module, "testCase", "infer_test_{0}".format(loc), str(target_res) + "m")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_file = "_".join([sample_loc[loc]["output_prefix"], self.target_var, output_suffix]) + ".tif"
                output_path = os.path.join(output_dir, output_file)

                shafts.pred_height_from_tiff_DL_patch(extent=extent, out_file=output_path, tif_ref=input_ref, patch_size=input_size,
                                                        predictor=self.model, trained_record=pt_path, resolution=output_res,
                                                        s1_key=self.s1_key, s2_key=self.s2_key,
                                                        aux_feat_info=aux_feat_info, base_dir=self.tmp_dir, padding=self.padding, 
                                                        batch_size=self.batch_size, tmp_suffix="_stl_{0}".format(target_res), activation=self.activation, log_scale=False,
                                                        cuda_used=self.cuda_used, v_min=self.v_min, v_max=self.v_max)
                
                test_flag.append(os.path.exists(output_path))
                os.remove(output_path)
        
        shutil.rmtree(self.tmp_dir)
        
        self.assertTrue(all(test_flag))
    
    # ------test if MTL models can run
    def test_inference_MTL(self):
        # ------specify the sample case
        sample_loc = {
            "THU": {
                "extent": [116.31916, 40.00171, 116.32078, 40.00189],
                "raw_data": {
                    "50pt": {
                        self.s1_key: os.path.join(shafts._path_shaft_module, "testCase", "infer_test_THU/raw_data/THU_2020_sentinel_1_50pt.tif"),
                        self.s2_key: os.path.join(shafts._path_shaft_module, "testCase", "infer_test_THU/raw_data/THU_2020_sentinel_2_50pt.tif"),
                    }
                },
                "aux_feat": {
                    "DEM": {
                        "path": os.path.join(shafts._path_shaft_module, "testCase", "infer_test_THU/raw_data/THU_srtm.tif"),
                        "patch_size_ratio": 1.0,
                    } 
                },
                "output_prefix": "THU",
            }
        }

        # ------specify the suffix of output files
        output_suffix = self.backbone

        if not os.path.isdir(self.tmp_dir):
            os.mkdir(self.tmp_dir)
        
        test_height_flag = []
        test_footprint_flag = []

        for loc in sample_loc.keys():
            extent = sample_loc[loc]["extent"]
            input_ref = sample_loc[loc]["raw_data"]
            aux_feat_info = sample_loc[loc]["aux_feat"]
            #for target_res in [100, 250, 500, 1000]:
            for target_res in [100]:
                input_size = self.input_size[target_res]
                output_res = self.res[target_res]
                pt_path = self.mtl_trained_record[target_res]

                output_dir = os.path.join(shafts._path_shaft_module, "testCase", "infer_test_{0}".format(loc), str(target_res) + "m")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_footprint_file = "_".join([sample_loc[loc]["output_prefix"], "footprint", output_suffix + "_MTL"]) + ".tif"
                output_footprint_path = os.path.join(output_dir, output_footprint_file)
                output_height_file = "_".join([sample_loc[loc]["output_prefix"], "height", output_suffix + "_MTL"]) + ".tif"
                output_height_path = os.path.join(output_dir, output_height_file)
                
                shafts.pred_height_from_tiff_DL_patch_MTL(extent=extent, out_footprint_file=output_footprint_path, out_height_file=output_height_path, 
                                                            tif_ref=input_ref, patch_size=input_size,
                                                            predictor=self.model, trained_record=pt_path, resolution=output_res,
                                                            s1_key=self.s1_key, s2_key=self.s2_key,
                                                            aux_feat_info=aux_feat_info, crossed=False, base_dir=self.tmp_dir, padding=self.padding, 
                                                            batch_size=self.batch_size, tmp_suffix="_mtl_{0}".format(target_res), log_scale=False,
                                                            cuda_used=self.cuda_used, 
                                                            h_min=self.v_min, h_max=self.v_max, f_min=0.0, f_max=1.0)
                
                test_height_flag.append(os.path.exists(output_height_path))
                os.remove(output_height_path)
                test_footprint_flag.append(os.path.exists(output_footprint_path))
                os.remove(output_footprint_path)
        
        shutil.rmtree(self.tmp_dir)
        
        self.assertTrue(all(test_height_flag) and all(test_footprint_flag))
