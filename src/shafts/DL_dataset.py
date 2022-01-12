import os
from re import S
import pyarrow as pa
import h5py
import lmdb

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as album

from .mathexpr import *


rgb_scale_data_transforms = {
    'train': album.Compose([
        # normalize RGB channel
        # transforms.Lambda(lambda rgb_img: rgb_rescale(rgb_img, vmin=0, vmax=255, axis=(0, 1))),
        # add ISO noise
        album.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
        # add Gaussian noise
        album.GaussNoise(var_limit=(10.0, 50.0), mean=0),
        # randomly changes the brightness, contrast, and saturation of an image
        album.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
    ]),
    'valid': album.Compose([
        # transforms.Lambda(lambda rgb_img: rgb_rescale(rgb_img, vmin=0, vmax=255, axis=(0, 1)))
    ]),
}

gray_scale_data_transforms = {
    'train': album.Compose([
        # add Gaussian noise
        album.GaussNoise(var_limit=(5.0, 25.0), mean=0),
        # apply Contrast Limited Adaptive Histogram Equalization to the input image.
        # album.CLAHE(),
    ]),
    'valid': album.Compose([
        #album.CLAHE(),
    ]),
}

flip_transform = album.Flip(p=0.5)
tensor_transform = transforms.ToTensor()
clahe_transform = album.CLAHE(p=1.0)


def get_MAD_fromHDF5(h5_dataset_path, target_variable="BuildingHeight", scale=1.4826, log_scale=False):
    target_list = []
    with h5py.File(h5_dataset_path, "r") as h5_file:
        for group_name in h5_file.keys():
            target_dataset = h5_file[group_name][target_variable][()]
            if log_scale:
                target_dataset = np.log(target_dataset)

            target_list.append(target_dataset)

    target_list = np.concatenate(target_list, axis=0)
    target_median = np.median(target_list)
    target_list = np.abs(target_list - target_median)
    mad = np.median(target_list)

    return scale * mad


def HDF5_to_LMDB(h5_dataset_path, lmdb_dataset_path, target_list, aggregation_list, s1_prefix="sentinel_1", s2_prefix="sentinel_2", scale=0.0001, max_size=1099511627776, aux_namelist=None):
    hf_db = h5py.File(h5_dataset_path, mode="r")
    lmdb_db = lmdb.open(lmdb_dataset_path, subdir=os.path.isdir(lmdb_dataset_path),
                        map_size=max_size, readonly=False, meminit=False, map_async=True)
    print("Dataset conversion from {0} to {1}".format(h5_dataset_path, lmdb_dataset_path))

    txn = lmdb_db.begin(write=True)

    channel_ref = {}
    band_ref = {"VV": "B1", "VH": "B2", "R": "B1", "G": "B2", "B": "B3", "NIR": "B4"}
    for aggregation_ops in aggregation_list:
        channel_ref[aggregation_ops] = {"S1_VV": "_".join([s1_prefix, aggregation_ops, band_ref["VV"]]),
                                        "S1_VH": "_".join([s1_prefix, aggregation_ops, band_ref["VH"]]),
                                        "S2_RGB": ["_".join([s2_prefix, aggregation_ops, band_ref["R"]]),
                                                   "_".join([s2_prefix, aggregation_ops, band_ref["G"]]),
                                                   "_".join([s2_prefix, aggregation_ops, band_ref["B"]])],
                                        "S2_NIR": "_".join([s2_prefix, aggregation_ops, band_ref["NIR"]])}

    num_acc = 0
    for group_name in hf_db.keys():
        num_sample = hf_db[group_name][target_list[0]].shape[0]
        for i in range(0, num_sample):
            # ------get band information
            band_val = []
            for aggregation_ops in channel_ref.keys():
                s1_vv_name = channel_ref[aggregation_ops]["S1_VV"]
                s1_vv_band = hf_db[group_name][s1_vv_name][i]
                s1_vv_band = get_backscatterCoef(s1_vv_band) * 255
                # ------sometimes back-scattering coef would be greater than 1; np.uint8(256) = 0
                s1_vv_band = np.where(s1_vv_band > 255, 255, s1_vv_band)
                s1_vv_band = np.transpose(s1_vv_band, (1, 2, 0))
                band_val.append(s1_vv_band)

                s1_vh_name = channel_ref[aggregation_ops]["S1_VH"]
                s1_vh_band = hf_db[group_name][s1_vh_name][i]
                s1_vh_band = get_backscatterCoef(s1_vh_band) * 255
                # ------sometimes back-scattering coef would be greater than 1; np.uint8(256) = 0
                s1_vh_band = np.where(s1_vh_band > 255, 255, s1_vh_band)
                s1_vh_band = np.transpose(s1_vh_band, (1, 2, 0))
                band_val.append(s1_vh_band)

                s2_rgb_band_tmp = []
                s2_band_namelist = channel_ref[aggregation_ops]["S2_RGB"]
                # ---------shape of s2_patch_tmp: (num_s1_band, num_scale, size_y, size_x)
                for s2_band in s2_band_namelist:
                    s2_rgb_band_tmp.append(hf_db[group_name][s2_band][i])
                s2_rgb_band = np.concatenate(s2_rgb_band_tmp, axis=0) * scale * 255
                s2_rgb_band = np.transpose(s2_rgb_band, (1, 2, 0))
                #s2_rgb_band = rgb_rescale(s2_rgb_band, vmin=0, vmax=255, axis=(0, 1))
                #s2_rgb_band = clahe_transform(image=s2_rgb_band.astype(np.uint8))["image"]
                band_val.append(s2_rgb_band)

                s2_nir_name = channel_ref[aggregation_ops]["S2_NIR"]
                s2_nir_band = hf_db[group_name][s2_nir_name][i] * scale * 255
                s2_nir_band = np.transpose(s2_nir_band, (1, 2, 0))
                #s2_nir_band = rgb_rescale(s2_nir_band, vmin=0, vmax=255, axis=(0, 1))
                #s2_nir_band = clahe_transform(image=s2_nir_band.astype(np.uint8))["image"]
                band_val.append(s2_nir_band)

            band_val = np.concatenate(band_val, axis=-1)

            # ------get auxiliary feature information
            aux_val = None
            if aux_namelist is not None:
                aux_val = []
                aux_namelist = sorted(aux_namelist)
                for k in aux_namelist:
                    aux_band = hf_db[group_name][k][i]
                    aux_band = np.transpose(aux_band, (1, 2, 0))
                    aux_val.append(aux_band)

            # ------get target information
            target_val_list = []
            for target in target_list:
                target_val_list.append(hf_db[group_name][target][i])

            # ------save into LMDB
            if aux_val is None:
                txn.put(u'{}'.format(num_acc).encode('ascii'), dumps_pyarrow((band_val, *target_val_list)))
            else:
                txn.put(u'{}'.format(num_acc).encode('ascii'), dumps_pyarrow((band_val, *aux_val, *target_val_list)))
            num_acc += 1

    hf_db.close()

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(0, num_acc)]
    with lmdb_db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing LMDB database ...")
    lmdb_db.sync()
    lmdb_db.close()


# ---PyTorch dataset based on LMDB (Lightning Memory-Mapped Database) input
# ------ref to: https://github.com/Lyken17/Efficient-PyTorch/blob/master/tools/folder2lmdb.py
def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


class PatchDatasetFromLMDB(Dataset):
    def __init__(self, lmdb_dataset_path, aggregation_list, num_band=6, scale=0.0001, cached=True, mode="train", target_id_shift=1, log_scale=False, aux_namelist=None, aux_id_shift=1):
        self.db_path = lmdb_dataset_path
        self.num_aggregation = len(aggregation_list)
        self.num_band = num_band
        self.scale = scale
        self.env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
                             readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pa.deserialize(txn.get(b'__len__'))
            self.keys = pa.deserialize(txn.get(b'__keys__'))

        self.log_scale = log_scale
        self.target_id_shift = target_id_shift
        self.mode = mode
        self.band_id_ref = {"S1_VV": 0, "S1_VH": 1, "S2_R": 2, "S2_G": 3, "S2_B": 4, "S2_NIR": 5}

        self.aux_id_ref = None
        self.target_id_shift = int(self.target_id_shift + aux_id_shift)
        if aux_namelist is not None:
            self.aux_namelist = sorted(aux_namelist)
            self.aux_id_ref = {self.aux_namelist[i]: int(i+1) for i in range(0, len(self.aux_namelist))}
            self.target_id_shift = int(self.target_id_shift - aux_id_shift + len(self.aux_namelist))

        self.cached = cached
        self.db = {}
        if self.cached:
            self.cache_dateset()

    def __getitem__(self, index):
        if self.cached:
            dta_unpacked = self.db[index]
        else:
            env = self.env
            with env.begin(write=False) as txn:
                byteflow = txn.get(self.keys[index])

            dta_unpacked = pa.deserialize(byteflow)

        # ------get band information
        band_val = dta_unpacked[0]
        patch_multi_band = []
        for idx in range(0, self.num_aggregation):
            id_shift = int(idx * self.num_band)

            s1_vv_id = id_shift + self.band_id_ref["S1_VV"]
            s1_vv_coef_patch = np.expand_dims(band_val[:, :, s1_vv_id], axis=-1)

            s1_vh_id = id_shift + self.band_id_ref["S1_VH"]
            s1_vh_coef_patch = np.expand_dims(band_val[:, :, s1_vh_id], axis=-1)

            s2_r_id = id_shift + self.band_id_ref["S2_R"]
            s2_g_id = id_shift + self.band_id_ref["S2_G"]
            s2_b_id = id_shift + self.band_id_ref["S2_B"]
            s2_rgb_patch_tmp = [np.expand_dims(band_val[:, :, s2_r_id], axis=-1),
                                np.expand_dims(band_val[:, :, s2_g_id], axis=-1),
                                np.expand_dims(band_val[:, :, s2_b_id], axis=-1)]
            s2_rgb_patch = np.concatenate(s2_rgb_patch_tmp, axis=-1)

            s2_nir_id = id_shift + self.band_id_ref["S2_NIR"]
            s2_nir_patch = np.expand_dims(band_val[:, :, s2_nir_id], axis=-1)

            if self.mode == "train":
                s1_vv_patch = gray_scale_data_transforms["train"](image=s1_vv_coef_patch.astype(np.uint8))["image"]
                s1_vh_patch = gray_scale_data_transforms["train"](image=s1_vh_coef_patch.astype(np.uint8))["image"]
                s2_rgb_patch = rgb_scale_data_transforms["train"](image=s2_rgb_patch.astype(np.uint8))["image"]
                s2_nir_patch = gray_scale_data_transforms["train"](image=s2_nir_patch.astype(np.uint8))["image"]
                patch = np.concatenate([s1_vv_patch, s1_vh_patch, s2_rgb_patch, s2_nir_patch], axis=-1)
                # ---------flip the input patch randomly
                patch = flip_transform(image=patch)["image"]
            elif self.mode == "valid":
                s1_vv_patch = gray_scale_data_transforms["valid"](image=s1_vv_coef_patch.astype(np.uint8))["image"]
                s1_vh_patch = gray_scale_data_transforms["valid"](image=s1_vh_coef_patch.astype(np.uint8))["image"]
                s2_rgb_patch = rgb_scale_data_transforms["valid"](image=s2_rgb_patch.astype(np.uint8))["image"]
                s2_nir_patch = gray_scale_data_transforms["valid"](image=s2_nir_patch.astype(np.uint8))["image"]
                patch = np.concatenate([s1_vv_patch, s1_vh_patch, s2_rgb_patch, s2_nir_patch], axis=-1)
            else:
                raise NotImplementedError

            # ---------convert numpy.ndarray of shape [H, W, C] to torch.tensor [C, H, W]
            patch = tensor_transform(patch).type(torch.FloatTensor)

            patch_multi_band.append(patch)

        feat = torch.cat(patch_multi_band, dim=0)

        # ------get target information
        target_val = dta_unpacked[self.target_id_shift]

        if self.log_scale:
            sample = {"feature": feat, "value": np.log(target_val)}
        else:
            sample = {"feature": feat, "value": target_val}

        if self.aux_id_ref is not None:
            aux_feat = []
            for k in self.aux_namelist:
                aux_val =  dta_unpacked[self.aux_id_ref[k]]
                aux_feat.append(tensor_transform(aux_val).type(torch.FloatTensor))
            aux_feat = torch.cat(aux_feat, dim=0)
            sample["aux_feature"] = aux_feat

        return sample

    def __len__(self):
        return self.length

    def cache_dateset(self):
        env = self.env
        counter = 0
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            for k, v in cursor:
                self.db[counter] = pa.deserialize(v)
                counter += 1


class PatchDatasetFromLMDB_MTL(Dataset):
    def __init__(self, lmdb_dataset_path, aggregation_list, num_band=6, scale=0.0001, cached=True, mode="train",
                 footprint_id_shift=1, height_id_shift=2, log_scale=False, aux_namelist=None, aux_id_shift=1):
        self.db_path = lmdb_dataset_path
        self.num_aggregation = len(aggregation_list)
        self.num_band = num_band
        self.scale = scale
        self.env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
                             readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pa.deserialize(txn.get(b'__len__'))
            self.keys = pa.deserialize(txn.get(b'__keys__'))

        self.log_scale = log_scale
        self.footprint_id_shift = footprint_id_shift
        self.height_id_shift = height_id_shift
        self.mode = mode
        self.band_id_ref = {"S1_VV": 0, "S1_VH": 1, "S2_R": 2, "S2_G": 3, "S2_B": 4, "S2_NIR": 5}

        self.aux_id_ref = None
        self.footprint_id_shift = int(self.footprint_id_shift + aux_id_shift)
        self.height_id_shift = int(self.height_id_shift + aux_id_shift)
        if aux_namelist is not None:
            self.aux_namelist = sorted(aux_namelist)
            self.aux_id_ref = {self.aux_namelist[i]: int(i+1) for i in range(0, len(self.aux_namelist))}
            self.footprint_id_shift = int(self.footprint_id_shift - aux_id_shift + len(self.aux_namelist))
            self.height_id_shift = int(self.height_id_shift - aux_id_shift + len(self.aux_namelist))
        
        self.cached = cached
        self.db = {}
        if self.cached:
            self.cache_dateset()

    def __getitem__(self, index):
        if self.cached:
            dta_unpacked = self.db[index]
        else:
            env = self.env
            with env.begin(write=False) as txn:
                byteflow = txn.get(self.keys[index])

            dta_unpacked = pa.deserialize(byteflow)

        # ------get band information
        band_val = dta_unpacked[0]
        patch_multi_band = []
        for idx in range(0, self.num_aggregation):
            id_shift = int(idx * self.num_band)

            s1_vv_id = id_shift + self.band_id_ref["S1_VV"]
            s1_vv_coef_patch = np.expand_dims(band_val[:, :, s1_vv_id], axis=-1)

            s1_vh_id = id_shift + self.band_id_ref["S1_VH"]
            s1_vh_coef_patch = np.expand_dims(band_val[:, :, s1_vh_id], axis=-1)

            s2_r_id = id_shift + self.band_id_ref["S2_R"]
            s2_g_id = id_shift + self.band_id_ref["S2_G"]
            s2_b_id = id_shift + self.band_id_ref["S2_B"]
            s2_rgb_patch_tmp = [np.expand_dims(band_val[:, :, s2_r_id], axis=-1),
                                np.expand_dims(band_val[:, :, s2_g_id], axis=-1),
                                np.expand_dims(band_val[:, :, s2_b_id], axis=-1)]
            s2_rgb_patch = np.concatenate(s2_rgb_patch_tmp, axis=-1)

            s2_nir_id = id_shift + self.band_id_ref["S2_NIR"]
            s2_nir_patch = np.expand_dims(band_val[:, :, s2_nir_id], axis=-1)

            if self.mode == "train":
                s1_vv_patch = gray_scale_data_transforms["train"](image=s1_vv_coef_patch.astype(np.uint8))["image"]
                s1_vh_patch = gray_scale_data_transforms["train"](image=s1_vh_coef_patch.astype(np.uint8))["image"]
                s2_rgb_patch = rgb_scale_data_transforms["train"](image=s2_rgb_patch.astype(np.uint8))["image"]
                s2_nir_patch = gray_scale_data_transforms["train"](image=s2_nir_patch.astype(np.uint8))["image"]
                patch = np.concatenate([s1_vv_patch, s1_vh_patch, s2_rgb_patch, s2_nir_patch], axis=-1)
                # ---------flip the input patch randomly
                patch = flip_transform(image=patch)["image"]
            elif self.mode == "valid":
                s1_vv_patch = gray_scale_data_transforms["valid"](image=s1_vv_coef_patch.astype(np.uint8))["image"]
                s1_vh_patch = gray_scale_data_transforms["valid"](image=s1_vh_coef_patch.astype(np.uint8))["image"]
                s2_rgb_patch = rgb_scale_data_transforms["valid"](image=s2_rgb_patch.astype(np.uint8))["image"]
                s2_nir_patch = gray_scale_data_transforms["valid"](image=s2_nir_patch.astype(np.uint8))["image"]
                patch = np.concatenate([s1_vv_patch, s1_vh_patch, s2_rgb_patch, s2_nir_patch], axis=-1)
            else:
                raise NotImplementedError

            # ---------convert numpy.ndarray of shape [H, W, C] to torch.tensor [C, H, W]
            patch = tensor_transform(patch).type(torch.FloatTensor)

            patch_multi_band.append(patch)

        feat = torch.cat(patch_multi_band, dim=0)

        # ------get target information
        target_footprint = dta_unpacked[self.footprint_id_shift]
        target_height = dta_unpacked[self.height_id_shift]

        if self.log_scale:
            sample = {"feature": feat, "footprint": target_footprint, "height": np.log(target_height)}
        else:
            sample = {"feature": feat, "footprint": target_footprint, "height": target_height}

        aux_feat = None
        if self.aux_id_ref is not None:
            aux_feat = []
            for k in self.aux_namelist:
                aux_val =  dta_unpacked[self.aux_id_ref[k]]
                aux_feat.append(tensor_transform(aux_val).type(torch.FloatTensor))
            aux_feat = torch.cat(aux_feat, dim=0)
            sample["aux_feature"] = aux_feat

        return sample

    def __len__(self):
        return self.length

    def cache_dateset(self):
        env = self.env
        counter = 0
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            for k, v in cursor:
                self.db[counter] = pa.deserialize(v)
                counter += 1


# ---PyTorch dataset based on HDF5 input
class PatchDatasetFromHDF5(Dataset):
    def __init__(self, h5_dataset_path, target_variable, aggregation_list, s1_prefix="sentinel_1", s2_prefix="sentinel_2", scale=0.0001, cached=True, mode="train", log_scale=False, aux_namelist=None):
        self.scale = scale
        self.db = None
        self.ds_path = h5_dataset_path
        # ------define feature and target information
        self.target_variable = target_variable
        self.target_info = {}
        self.feature_info = {}
        self.set_dataset_info(h5_dataset_path, self.target_variable)

        self.group_namelist = sorted(self.target_info.keys())
        self.aggregation_list = aggregation_list
        self.channel_ref = {}
        band_ref = {"VV": "B1", "VH": "B2", "R": "B1", "G": "B2", "B": "B3", "NIR": "B4"}
        for aggregation_ops in aggregation_list:
            self.channel_ref[aggregation_ops] = {"S1_VV": "_".join([s1_prefix, aggregation_ops, band_ref["VV"]]),
                                                 "S1_VH": "_".join([s1_prefix, aggregation_ops, band_ref["VH"]]),
                                                 "S2_RGB": ["_".join([s2_prefix, aggregation_ops, band_ref["R"]]),
                                                            "_".join([s2_prefix, aggregation_ops, band_ref["G"]]),
                                                            "_".join([s2_prefix, aggregation_ops, band_ref["B"]])],
                                                 "S2_NIR": "_".join([s2_prefix, aggregation_ops, band_ref["NIR"]])}

        self.aux_namelist = None
        if aux_namelist is not None:
            self.aux_namelist = sorted(aux_namelist)
        
        self.num_sample_acc = np.cumsum([self.target_info[g]["shape"] for g in self.group_namelist])
        self.num_sample_acc = np.concatenate([[0], self.num_sample_acc], axis=0)

        # ------Note that only part of data can be stored into memory
        # ------data cache for faster data loading
        self.cached = cached
        self.db = {}
        if self.cached:
            self.cache_dateset(h5_dataset_path)
        # ------we use different DataTransform for training phase and validation phase
        self.mode = mode
        self.log_scale = log_scale

    def __len__(self):
        return self.num_sample_acc[-1]

    def __getitem__(self, index):
        # ------avoid opening HDF5 file each time (only available for HDF5 in version >= 1.10)
        # ---------see the discussion: https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
        if self.db is None:
            self.db = h5py.File(self.ds_path, "r")
        # ------By specifying side="right", we have: num_sample_acc[group_id-1] <= index < num_sample_acc[group_id]
        group_id = np.searchsorted(self.num_sample_acc, index, side="right")
        group_name = self.group_namelist[group_id-1]
        shift = index - self.num_sample_acc[group_id-1]
        # ------get feature information from multi-band patches
        # with h5py.File(self.ds_path, "r") as h5_file:
        patch_multi_band = []
        for aggregation_ops in self.channel_ref.keys():
            s1_vv_name = self.channel_ref[aggregation_ops]["S1_VV"]
            s1_vv_coef_patch = get_backscatterCoef(self.db[group_name][s1_vv_name][shift]) * 255
            s1_vv_coef_patch = np.transpose(s1_vv_coef_patch, (1, 2, 0))

            s1_vh_name = self.channel_ref[aggregation_ops]["S1_VH"]
            s1_vh_coef_patch = get_backscatterCoef(self.db[group_name][s1_vh_name][shift]) * 255
            s1_vh_coef_patch = np.transpose(s1_vh_coef_patch, (1, 2, 0))

            s2_rgb_patch_tmp = []
            s2_band_namelist = self.channel_ref[aggregation_ops]["S2_RGB"]
            # ---------shape of s2_patch_tmp: (num_s1_band, num_scale, size_y, size_x)
            for s2_band in s2_band_namelist:
                s2_rgb_patch_tmp.append(self.db[group_name][s2_band][shift])
            s2_rgb_patch = np.concatenate(s2_rgb_patch_tmp, axis=0) * self.scale * 255
            s2_rgb_patch = np.transpose(s2_rgb_patch, (1, 2, 0))
            # s2_rgb_patch = rgb_rescale(s2_rgb_patch, vmin=0, vmax=255, axis=(0, 1))

            s2_nir_name = self.channel_ref[aggregation_ops]["S2_NIR"]
            s2_nir_patch = self.db[group_name][s2_nir_name][shift] * self.scale * 255
            s2_nir_patch = np.transpose(s2_nir_patch, (1, 2, 0))
            # s2_nir_patch = rgb_rescale(s2_nir_patch, vmin=0, vmax=255, axis=(0, 1))

            if self.mode == "train":
                s1_vv_patch = gray_scale_data_transforms["train"](image=s1_vv_coef_patch.astype(np.uint8))["image"]
                s1_vh_patch = gray_scale_data_transforms["train"](image=s1_vh_coef_patch.astype(np.uint8))["image"]
                s2_rgb_patch = rgb_scale_data_transforms["train"](image=s2_rgb_patch.astype(np.uint8))["image"]
                s2_nir_patch = gray_scale_data_transforms["train"](image=s2_nir_patch.astype(np.uint8))["image"]
                patch = np.concatenate([s1_vv_patch, s1_vh_patch, s2_rgb_patch, s2_nir_patch], axis=-1)
                # ---------flip the input patch randomly
                patch = flip_transform(image=patch)["image"]
            elif self.mode == "valid":
                s1_vv_patch = gray_scale_data_transforms["valid"](image=s1_vv_coef_patch.astype(np.uint8))["image"]
                s1_vh_patch = gray_scale_data_transforms["valid"](image=s1_vh_coef_patch.astype(np.uint8))["image"]
                s2_rgb_patch = rgb_scale_data_transforms["valid"](image=s2_rgb_patch.astype(np.uint8))["image"]
                s2_nir_patch = gray_scale_data_transforms["valid"](image=s2_nir_patch.astype(np.uint8))["image"]
                patch = np.concatenate([s1_vv_patch, s1_vh_patch, s2_rgb_patch, s2_nir_patch], axis=-1)
            else:
                raise NotImplementedError

            # ---------convert numpy.ndarray of shape [H, W, C] to torch.tensor [C, H, W]
            patch = tensor_transform(patch).type(torch.FloatTensor)

            patch_multi_band.append(patch)

        feat = torch.cat(patch_multi_band, dim=0)

        # ------get value information
        value = self.db[group_name][self.target_variable][shift]

        if self.log_scale:
            sample = {"feature": feat, "value": np.log(value)}
        else:
            sample = {"feature": feat, "value": value}

        if self.aux_namelist is not None:
            aux_feat = []
            for k in self.aux_namelist:
                aux_patch = self.db[group_name][k][shift]
                aux_patch = np.transpose(aux_patch, (1, 2, 0))
                aux_feat.append(tensor_transform(aux_patch).type(torch.FloatTensor))
            aux_feat = torch.cat(aux_feat, dim=0)
            sample["aux_feature"] = aux_feat

        return sample

    def set_dataset_info(self, h5_dataset_path, target_variable):
        with h5py.File(h5_dataset_path, "r") as h5_file:
            for group_name, group in h5_file.items():
                for dataset_name, dataset in group.items():
                    # ------determine the type of the dataset: feature or value to be predicted
                    if dataset_name == target_variable:
                        self.target_info[group_name] = {"dataset_name": dataset_name, "shape": dataset.shape[0]}
                    else:
                        if group_name not in self.feature_info.keys():
                            self.feature_info[group_name] = {}
                        self.feature_info[group_name][dataset_name] = {"shape": dataset.shape}

    def cache_dateset(self, h5_dataset_path):
        with h5py.File(h5_dataset_path, "r") as h5_file:
            for group_name in self.group_namelist:
                group = h5_file[group_name]
                self.db[group_name] = {}
                for dataset_name, dataset in group.items():
                    self.db[group_name][dataset_name] = dataset


class PatchDatasetFromHDF5_MTL(Dataset):
    def __init__(self, h5_dataset_path, aggregation_list, s1_prefix="sentinel_1", s2_prefix="sentinel_2", scale=0.0001, cached=True, mode="train", log_scale=False, aux_namelist=None):
        self.scale = scale
        self.db = None
        self.ds_path = h5_dataset_path
        # ------define feature and target information
        self.target_info = {}
        self.feature_info = {}
        self.set_dataset_info(h5_dataset_path,)

        self.group_namelist = sorted(self.target_info.keys())
        self.aggregation_list = aggregation_list
        self.channel_ref = {}
        band_ref = {"VV": "B1", "VH": "B2", "R": "B1", "G": "B2", "B": "B3", "NIR": "B4"}
        for aggregation_ops in aggregation_list:
            self.channel_ref[aggregation_ops] = {"S1_VV": "_".join([s1_prefix, aggregation_ops, band_ref["VV"]]),
                                                 "S1_VH": "_".join([s1_prefix, aggregation_ops, band_ref["VH"]]),
                                                 "S2_RGB": ["_".join([s2_prefix, aggregation_ops, band_ref["R"]]),
                                                            "_".join([s2_prefix, aggregation_ops, band_ref["G"]]),
                                                            "_".join([s2_prefix, aggregation_ops, band_ref["B"]])],
                                                 "S2_NIR": "_".join([s2_prefix, aggregation_ops, band_ref["NIR"]])}

        self.aux_namelist = None
        if aux_namelist is not None:
            self.aux_namelist = sorted(aux_namelist)
        
        self.num_sample_acc = np.cumsum([self.target_info[g]["shape"] for g in self.group_namelist])
        self.num_sample_acc = np.concatenate([[0], self.num_sample_acc], axis=0)

        # ------Note that only part of data can be stored into memory
        # ------data cache for faster data loading
        self.cached = cached
        self.db = {}
        if self.cached:
            self.cache_dateset(h5_dataset_path)
        # ------we use different DataTransform for training phase and validation phase
        self.mode = mode
        self.log_scale = log_scale

    def __len__(self):
        return self.num_sample_acc[-1]

    def __getitem__(self, index):
        # ------avoid opening HDF5 file each time (only available for HDF5 in version >= 1.10)
        # ---------see the discussion: https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
        if self.db is None:
            self.db = h5py.File(self.ds_path, "r")
        # ------By specifying side="right", we have: num_sample_acc[group_id-1] <= index < num_sample_acc[group_id]
        group_id = np.searchsorted(self.num_sample_acc, index, side="right")
        group_name = self.group_namelist[group_id-1]
        shift = index - self.num_sample_acc[group_id-1]
        # ------get feature information from multi-band patches
        # with h5py.File(self.ds_path, "r") as h5_file:
        patch_multi_band = []
        for aggregation_ops in self.channel_ref.keys():
            s1_vv_name = self.channel_ref[aggregation_ops]["S1_VV"]
            s1_vv_coef_patch = get_backscatterCoef(self.db[group_name][s1_vv_name][shift]) * 255
            s1_vv_coef_patch = np.transpose(s1_vv_coef_patch, (1, 2, 0))

            s1_vh_name = self.channel_ref[aggregation_ops]["S1_VH"]
            s1_vh_coef_patch = get_backscatterCoef(self.db[group_name][s1_vh_name][shift]) * 255
            s1_vh_coef_patch = np.transpose(s1_vh_coef_patch, (1, 2, 0))

            s2_rgb_patch_tmp = []
            s2_band_namelist = self.channel_ref[aggregation_ops]["S2_RGB"]
            # ---------shape of s2_patch_tmp: (num_s1_band, num_scale, size_y, size_x)
            for s2_band in s2_band_namelist:
                s2_rgb_patch_tmp.append(self.db[group_name][s2_band][shift])
            s2_rgb_patch = np.concatenate(s2_rgb_patch_tmp, axis=0) * self.scale * 255
            s2_rgb_patch = np.transpose(s2_rgb_patch, (1, 2, 0))
            # s2_rgb_patch = rgb_rescale(s2_rgb_patch, vmin=0, vmax=255, axis=(0, 1))

            s2_nir_name = self.channel_ref[aggregation_ops]["S2_NIR"]
            s2_nir_patch = self.db[group_name][s2_nir_name][shift] * self.scale * 255
            s2_nir_patch = np.transpose(s2_nir_patch, (1, 2, 0))
            # s2_nir_patch = rgb_rescale(s2_nir_patch, vmin=0, vmax=255, axis=(0, 1))

            if self.mode == "train":
                s1_vv_patch = gray_scale_data_transforms["train"](image=s1_vv_coef_patch.astype(np.uint8))["image"]
                s1_vh_patch = gray_scale_data_transforms["train"](image=s1_vh_coef_patch.astype(np.uint8))["image"]
                s2_rgb_patch = rgb_scale_data_transforms["train"](image=s2_rgb_patch.astype(np.uint8))["image"]
                s2_nir_patch = gray_scale_data_transforms["train"](image=s2_nir_patch.astype(np.uint8))["image"]
                patch = np.concatenate([s1_vv_patch, s1_vh_patch, s2_rgb_patch, s2_nir_patch], axis=-1)
                # ---------flip the input patch randomly
                patch = flip_transform(image=patch)["image"]
            elif self.mode == "valid":
                s1_vv_patch = gray_scale_data_transforms["valid"](image=s1_vv_coef_patch.astype(np.uint8))["image"]
                s1_vh_patch = gray_scale_data_transforms["valid"](image=s1_vh_coef_patch.astype(np.uint8))["image"]
                s2_rgb_patch = rgb_scale_data_transforms["valid"](image=s2_rgb_patch.astype(np.uint8))["image"]
                s2_nir_patch = gray_scale_data_transforms["valid"](image=s2_nir_patch.astype(np.uint8))["image"]
                patch = np.concatenate([s1_vv_patch, s1_vh_patch, s2_rgb_patch, s2_nir_patch], axis=-1)
            else:
                raise NotImplementedError

            # ---------convert numpy.ndarray of shape [H, W, C] to torch.tensor [C, H, W]
            patch = tensor_transform(patch).type(torch.FloatTensor)

            patch_multi_band.append(patch)

        feat = torch.cat(patch_multi_band, dim=0)

        # ------get value information
        footprint = self.db[group_name]["BuildingFootprint"][shift]
        height = self.db[group_name]["BuildingHeight"][shift]

        if self.log_scale:
            sample = {"feature": feat, "footprint": footprint, "height": np.log(height)}
        else:
            sample = {"feature": feat, "footprint": footprint, "height": height}

        if self.aux_namelist is not None:
            aux_feat = []
            for k in self.aux_namelist:
                aux_patch = self.db[group_name][k][shift]
                aux_patch = np.transpose(aux_patch, (1, 2, 0))
                aux_feat.append(tensor_transform(aux_patch).type(torch.FloatTensor))
            aux_feat = torch.cat(aux_feat, dim=0)
            sample["aux_feature"] = aux_feat

        return sample

    def set_dataset_info(self, h5_dataset_path):
        with h5py.File(h5_dataset_path, "r") as h5_file:
            for group_name, group in h5_file.items():
                for dataset_name, dataset in group.items():
                    # ------determine the type of the dataset: feature or value to be predicted
                    if dataset_name == "BuildingFootprint":
                        self.target_info[group_name] = {"dataset_name_1": dataset_name, "shape": dataset.shape[0]}
                    elif dataset_name == "BuildingHeight":
                        self.target_info[group_name] = {"dataset_name_2": dataset_name, "shape": dataset.shape[0]}
                    else:
                        if group_name not in self.feature_info.keys():
                            self.feature_info[group_name] = {}
                        self.feature_info[group_name][dataset_name] = {"shape": dataset.shape}

    def cache_dateset(self, h5_dataset_path):
        with h5py.File(h5_dataset_path, "r") as h5_file:
            for group_name in self.group_namelist:
                group = h5_file[group_name]
                self.db[group_name] = {}
                for dataset_name, dataset in group.items():
                    self.db[group_name][dataset_name] = dataset


def load_data_hdf5(h5_dataset_path, batch_size, num_workers, target_variable, aggregation_list, mode, cached=True, log_scale=False, aux_namelist=None):
    if mode == "train":
        patch_dataset = PatchDatasetFromHDF5(h5_dataset_path, target_variable, aggregation_list, mode="train", cached=cached, log_scale=log_scale, aux_namelist=aux_namelist)
        data_loader = DataLoader(patch_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    elif mode == "valid":
        patch_dataset = PatchDatasetFromHDF5(h5_dataset_path, target_variable, aggregation_list, mode="valid", cached=cached, log_scale=log_scale, aux_namelist=aux_namelist)
        data_loader = DataLoader(patch_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    else:
        raise NotImplementedError

    return data_loader


def load_data_hdf5_MTL(h5_dataset_path, batch_size, num_workers, aggregation_list, mode, cached=True, log_scale=False, aux_namelist=None):
    if mode == "train":
        patch_dataset = PatchDatasetFromHDF5_MTL(h5_dataset_path, aggregation_list, mode="train", cached=cached, log_scale=log_scale, aux_namelist=aux_namelist)
        data_loader = DataLoader(patch_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    elif mode == "valid":
        patch_dataset = PatchDatasetFromHDF5_MTL(h5_dataset_path, aggregation_list, mode="valid", cached=cached, log_scale=log_scale, aux_namelist=aux_namelist)
        data_loader = DataLoader(patch_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    else:
        raise NotImplementedError

    return data_loader


def load_data_lmdb(lmdb_dataset_path, batch_size, num_workers, aggregation_list, mode, target_id_shift=1, cached=True, log_scale=False, aux_namelist=None, aux_id_shift=1):
    if mode == "train":
        patch_dataset = PatchDatasetFromLMDB(lmdb_dataset_path, aggregation_list, mode="train", target_id_shift=target_id_shift, cached=cached, log_scale=log_scale, aux_namelist=aux_namelist, aux_id_shift=aux_id_shift)
        data_loader = DataLoader(patch_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    elif mode == "valid":
        patch_dataset = PatchDatasetFromLMDB(lmdb_dataset_path, aggregation_list, mode="valid", target_id_shift=target_id_shift, cached=cached, log_scale=log_scale, aux_namelist=aux_namelist, aux_id_shift=aux_id_shift)
        data_loader = DataLoader(patch_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    else:
        raise NotImplementedError

    return data_loader


def load_data_lmdb_MTL(lmdb_dataset_path, batch_size, num_workers, aggregation_list, mode, cached=True, log_scale=False, aux_namelist=None):
    if mode == "train":
        patch_dataset = PatchDatasetFromLMDB_MTL(lmdb_dataset_path, aggregation_list, mode="train", cached=cached,
                                                 footprint_id_shift=1, height_id_shift=2, log_scale=log_scale, aux_namelist=aux_namelist)
        data_loader = DataLoader(patch_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    elif mode == "valid":
        patch_dataset = PatchDatasetFromLMDB_MTL(lmdb_dataset_path, aggregation_list, mode="valid", cached=cached,
                                                 footprint_id_shift=1, height_id_shift=2, log_scale=log_scale, aux_namelist=aux_namelist)
        data_loader = DataLoader(patch_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    else:
        raise NotImplementedError

    return data_loader


if __name__ == "__main__":
    path_prefix = "/data/lyy/BuildingProject/dataset"

    res_scale_mapping = {"100m": 15, "250m": 30, "500m": 60, "1000m": 120}
    aux_feat = ["DEM"]

    for res in ["100m", "250m", "500m", "1000m"]:
        scale = res_scale_mapping[res]
        '''       
        HDF5_to_LMDB(h5_dataset_path=os.path.join(path_prefix, "patch_data_50pt_s{0}_{1}_train.h5".format(scale, res)),
                     lmdb_dataset_path=os.path.join(path_prefix, "patch_data_50pt_s{0}_{1}_train.lmdb".format(scale, res)),
                     target_list=["BuildingFootprint", "BuildingHeight"], aggregation_list=["50pt"],
                     s1_prefix="sentinel_1", s2_prefix="sentinel_2", 
                     scale=1.0, max_size=1099511627776, aux_namelist=aux_feat)
        HDF5_to_LMDB(h5_dataset_path=os.path.join(path_prefix, "patch_data_50pt_s{0}_{1}_valid.h5".format(scale, res)),
                     lmdb_dataset_path=os.path.join(path_prefix, "patch_data_50pt_s{0}_{1}_valid.lmdb".format(scale, res)),
                     target_list=["BuildingFootprint", "BuildingHeight"], aggregation_list=["50pt"],
                     s1_prefix="sentinel_1", s2_prefix="sentinel_2",
                     scale=1.0, max_size=1099511627776, aux_namelist=aux_feat)
        HDF5_to_LMDB(h5_dataset_path=os.path.join(path_prefix, "patch_data_50pt_s{0}_{1}_test.h5".format(scale, res)),
                     lmdb_dataset_path=os.path.join(path_prefix, "patch_data_50pt_s{0}_{1}_test.lmdb".format(scale, res)),
                     target_list=["BuildingFootprint", "BuildingHeight"], aggregation_list=["50pt"],
                     s1_prefix="sentinel_1", s2_prefix="sentinel_2", 
                     scale=1.0, max_size=1099511627776, aux_namelist=aux_feat)
        '''
        h_mad = get_MAD_fromHDF5(h5_dataset_path=os.path.join(path_prefix, "patch_data_50pt_s{0}_{1}_train.h5".format(scale, res)),
                                 target_variable="BuildingHeight", log_scale=False)
        d_mad = get_MAD_fromHDF5(h5_dataset_path=os.path.join(path_prefix, "patch_data_50pt_s{0}_{1}_train.h5".format(scale, res)),
                                 target_variable="BuildingFootprint", log_scale=False)
        print("*" * 30 + res + "*" * 30)
        print("MAD (Median Absolute Deviation) of BuildingHeight is: %.4f" % h_mad)
        print("MAD (Median Absolute Deviation) of BuildingFootprint is: %.4f" % d_mad)
        print("*" * 30 + res + "*" * 30)

    '''
    HDF5_to_LMDB(h5_dataset_path="/data/lyy/BuildingProject/dataset/patch_data_50pt_s15_100m_train.h5",
                 lmdb_dataset_path="/data/lyy/BuildingProject/dataset/patch_data_50pt_s15_100m_train.lmdb",
                 target_list=["BuildingFootprint", "BuildingHeight"], aggregation_list=["50pt"], s1_prefix="sentinel_1",
                 s2_prefix="sentinel_2", max_size=1099511627776)
    HDF5_to_LMDB(h5_dataset_path="/data/lyy/BuildingProject/dataset/patch_data_50pt_s15_100m_valid.h5",
                 lmdb_dataset_path="/data/lyy/BuildingProject/dataset/patch_data_50pt_s15_100m_valid.lmdb",
                 target_list=["BuildingFootprint", "BuildingHeight"], aggregation_list=["50pt"], s1_prefix="sentinel_1",
                 s2_prefix="sentinel_2", max_size=1099511627776)
                 
    mad = get_MAD_fromHDF5(h5_dataset_path="/data/lyy/BuildingProject/dataset/patch_data_50pt_s15_100m.h5", target_variable="BuildingHeight", log_scale=True)
    print("MAD (Median Absolute Deviation) of is: %.4f" % mad)
    '''
