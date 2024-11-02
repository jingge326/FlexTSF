import numpy as np
import pandas as pd
import os
import sklearn
import torch
from torch.utils.data import Dataset
from scipy.io.arff import loadarff

from experiments.utils_exp import StandardScaler


list_unnorm = [
    'AllGestureWiimoteX',
    'AllGestureWiimoteY',
    'AllGestureWiimoteZ',
    'BME',
    'Chinatown',
    'Crop',
    'EOGHorizontalSignal',
    'EOGVerticalSignal',
    'Fungi',
    'GestureMidAirD1',
    'GestureMidAirD2',
    'GestureMidAirD3',
    'GesturePebbleZ1',
    'GesturePebbleZ2',
    'GunPointAgeSpan',
    'GunPointMaleVersusFemale',
    'GunPointOldVersusYoung',
    'HouseTwenty',
    'InsectEPGRegularTrain',
    'InsectEPGSmallTrain',
    'MelbournePedestrian',
    'PickupGestureWiimoteZ',
    'PigAirwayPressure',
    'PigArtPressure',
    'PigCVP',
    'PLAID',
    'PowerCons',
    'Rock',
    'SemgHandGenderCh2',
    'SemgHandMovementCh2',
    'SemgHandSubjectCh2',
    'ShakeGestureWiimoteZ',
    'SmoothSubspace',
    'UMD'
]


class Dataset_Forecast_Irregular_Seperated(Dataset):
    def __init__(self, data, data_util, args):
        self.data = data
        self.args = args

        # int(seq_len * forecast_ratio) should be at least 1
        # Check this condition and remove unqualified samples
        sample_mask = data["mask"].any(
            axis=-1).sum(axis=-1) * self.args.forecast_ratio > 1
        self.data["value"] = data["value"][sample_mask]
        self.data["mask"] = data["mask"][sample_mask]
        self.data["time"] = data["time"][sample_mask]

        self.length = data["value"].shape[0]
        self.data_util = data_util

        if self.data_util.get("scaler") is None:
            d_mean = (data["value"]*data["mask"]).sum(axis=(0, 1)
                                                      ) / data["mask"].sum(axis=(0, 1))
            d_std = np.sqrt(np.sum(((data["value"] - d_mean)*data["mask"])**2, axis=(
                0, 1)) / (data["mask"].sum(axis=(0, 1))) + 1e-8)
            self.data_util["scaler"] = StandardScaler(mean=d_mean, std=d_std)

        self.data["value"] = self.data_util["scaler"].transform(data["value"])
        # Turn the values at missing positions to 0
        self.data["value"] = self.data["value"] * self.data["mask"]
        self.mean = self.data_util["scaler"].mean
        self.std = self.data_util["scaler"].std
        # if args.full_regular == True:
        #     time_max = int(self.data["time"].max()) + 1
        #     # create new arrays whose first dimension is the number of samples,
        #     # the second dimension is the maximum time length, and the third
        #     # dimension is the number of variables
        #     new_value = np.zeros(
        #         (self.length, time_max, self.data["value"].shape[-1]))
        #     new_mask = np.zeros(
        #         (self.length, time_max, self.data["value"].shape[-1]))
        #     new_time = np.repeat(np.arange(time_max)[
        #                          np.newaxis, :], self.length, axis=0)
        #     # Using self.data["time"] as the index to fill the new arrays
        #     new_value[np.arange(self.length)[:, np.newaxis],
        #               self.data["time"].astype(int), :] = self.data["value"]
        #     new_mask[np.arange(self.length)[:, np.newaxis],
        #              self.data["time"].astype(int), :] = self.data["mask"]
        #     # Explain: most sequences except for one in self.data["time"] are padded with 0 at the end,
        #     # meaning the 0 position of new_value and new_mask are assigned with new values more than once
        #     # and the last assignment is not the original value, but corresponding values at the end of the
        #     # self.data["value"] or self.data["mask"]. So we need to correct the values at the 0 position.
        #     new_value[:, 0, :] = self.data["value"][:, 0, :]
        #     new_mask[:, 0, :] = self.data["mask"][:, 0, :]

        #     self.data["value"] = new_value
        #     self.data["mask"] = new_mask
        #     self.data["time"] = new_time

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx_len = self.data["mask"][idx, ...].any(axis=-1).sum()
        idx_for = idx_len - int(idx_len * self.args.forecast_ratio)
        return {"data_in": self.data["value"][idx, :idx_for],
                "mask_in": self.data["mask"][idx, :idx_for],
                "time_in": self.data["time"][idx, :idx_for],
                "data_out": self.data["value"][idx, idx_for:idx_len],
                "mask_out": self.data["mask"][idx, idx_for:idx_len],
                "time_out": self.data["time"][idx, idx_for:idx_len],
                "gmean": self.mean,
                "gstd": self.std}


class Dataset_Forecast_Regular_Seperated(Dataset):
    def __init__(self, data, data_util, args):
        self.data = data
        self.args = args
        self.length = data["value"].shape[0]
        self.data_util = data_util
        if self.data_util.get("scaler") is None:
            d_mean = data["value"].mean(axis=(0, 1))
            d_std = data["value"].std(axis=(0, 1)) + 1e-8
            self.data_util["scaler"] = StandardScaler(mean=d_mean, std=d_std)

        self.data["value"] = self.data_util["scaler"].transform(data["value"])
        self.mean = self.data_util["scaler"].mean
        self.std = self.data_util["scaler"].std

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx_len = self.data["value"].shape[1]
        idx_for = idx_len - int(idx_len * self.args.forecast_ratio)
        return {"data_in": self.data["value"][idx, :idx_for],
                "mask_in": self.data["mask"][idx, :idx_for],
                "time_in": self.data["time"][idx, :idx_for],
                "data_out": self.data["value"][idx, idx_for:idx_len],
                "mask_out": self.data["mask"][idx, idx_for:idx_len],
                "time_out": self.data["time"][idx, idx_for:idx_len],
                "gmean": self.mean,
                "gstd": self.std}


def collate_fn_forecast_general(batch, args, dataset_config):
    dc = dataset_config
    # Mainly process data_in, labels
    value_in_list = []
    mask_in_list = []
    time_in_list = []
    len_in_list = []
    value_out_list = []
    mask_out_list = []
    time_out_list = []
    len_out_list = []
    for b in batch:
        value_in_list.append(b["data_in"])
        mask_in_list.append(b["mask_in"])
        time_in_list.append(b["time_in"])
        len_in_list.append(len(b["data_in"]))
        value_out_list.append(b["data_out"])
        mask_out_list.append(b["mask_out"])
        time_out_list.append(b["time_out"])
        len_out_list.append(len(b["data_out"]))

    max_len_in = max(len_in_list)
    max_len_out = max(len_out_list)
    if args.full_regular == True:
        pred_len = int(dc["seq_len"] * args.forecast_ratio)
        input_len = dc["seq_len"] - pred_len
        if (max_len_in != input_len) or (max_len_out != pred_len):
            max_len_in = input_len
            max_len_out = pred_len

    if args.fixed_patch_len == True:
        patch_len = args.patch_len
    elif dc.get("patch_len") is not None:
        patch_len = dc["patch_len"]
    else:
        if np.array(len_in_list).mean() <= 64:
            patch_len = 4
        elif np.array(len_in_list).mean() <= 256:
            patch_len = 8
        else:
            patch_len = 16

    combined_value_in = torch.from_numpy(np.stack([np.concatenate([values, np.zeros(
        (max_len_in-len_t, dc["var_num"]))], 0) for values, len_t in zip(value_in_list, len_in_list)], 0,)).to(args.device)
    combined_mask_in = torch.from_numpy(np.stack([np.concatenate([mask, np.zeros(
        (max_len_in-len_t, dc["var_num"]))], 0) for mask, len_t in zip(mask_in_list, len_in_list)], 0)).to(args.device)
    combined_time_in = torch.from_numpy(np.stack([np.concatenate([time, np.zeros(
        max_len_in-len_t)], 0) for time, len_t in zip(time_in_list, len_in_list)], 0,)).to(args.device)
    combined_value_out = torch.from_numpy(np.stack([np.concatenate([values, np.zeros(
        (max_len_out-len_t, dc["var_num"]))], 0) for values, len_t in zip(value_out_list, len_out_list)], 0,)).to(args.device)
    combined_mask_out = torch.from_numpy(np.stack([np.concatenate([mask, np.zeros(
        (max_len_out-len_t, dc["var_num"]))], 0) for mask, len_t in zip(mask_out_list, len_out_list)], 0)).to(args.device)
    combined_time_out = torch.from_numpy(np.stack([np.concatenate([time, np.zeros(
        max_len_out-len_t)], 0) for time, len_t in zip(time_out_list, len_out_list)], 0,)).to(args.device)
    # Create a tensor to indicate the existence of the time series
    exist_in = torch.from_numpy(np.stack([np.concatenate(
        [np.ones(len_t), np.zeros(max_len_in-len_t)], 0) for len_t in len_in_list], 0)).to(args.device)

    gmean = torch.from_numpy(batch[0]["gmean"]).to(combined_value_in)
    gstd = torch.from_numpy(batch[0]["gstd"]).to(combined_value_in)

    data_dict = {
        "time_in": combined_time_in.float(),
        "data_in": combined_value_in.float(),
        "mask_in": combined_mask_in.float(),
        "time_out": combined_time_out.float(),
        "data_out": combined_value_out.float(),
        "mask_out": combined_mask_out.float(),
        "gmean": gmean.float(),
        "gstd": gstd.float(),
        "time_unit": torch.tensor(dc["time_unit"]).to(combined_value_in).float(),
        "patch_len": patch_len,
        "exist_in": exist_in
    }

    return data_dict


def trans_ucruea_general(data, dc):
    # data shape (batch_size, seq_len, var_num)
    # A dataframe with shape (batch_size*seq_len, 2+var_num) where the first
    # column is the seq_id, the second column is observations orders,
    # and the rest columns are the values
    df = pd.DataFrame(data.reshape(-1, data.shape[-1]))
    df["seq_id"] = np.repeat(np.arange(data.shape[0]), data.shape[1])
    df["time_id"] = np.tile(np.arange(data.shape[1]), data.shape[0])
    # Set seq_id and time_id as the index
    df = df.set_index(["seq_id", "time_id"])
    # Transform all columns into one column
    df = df.stack()
    # Rename the new index as "var_id"
    df.index = df.index.set_names(["seq_id", "time_id", "var_id"])
    # Reset the index
    df = df.reset_index()
    # Rename the column name of the values
    df = df.rename(columns={0: "value"})
    # Sort the dataframe by "seq_id", "var_id", "time_id"
    df = df.sort_values(by=["seq_id", "var_id", "time_id"])
    # Exchange the order of "var_id" and "time_id"
    df = df[["seq_id", "var_id", "time_id", "value"]]
    # Drop missing values
    df = df.dropna()
    return df


# Randomly assign some elements of a array to zero according to a certain ratio
def mask_random(array_in, ratio):
    array_out = array_in.copy()
    mask = np.random.choice([0, 1], size=array_in.shape, p=[ratio, 1-ratio])
    array_out = array_out * mask
    return array_out


def pack_data_ucruea(data, dc):
    # Transform data into the general form
    df = trans_ucruea_general(data, dc)

    # Transform the dataframe into a numpy array with shape (batch_size, seq_len, var_num)
    # Add zeros to the end if the length of the sequence is less than the maximum length,
    # create "time" to indicate the time of the observations, "value" to indicate the values,
    # and create exist_times to indicate the existence of the time series
    seq_id = df["seq_id"].values
    var_id = df["var_id"].values
    time_id = df["time_id"].values
    seq_len = time_id.max() + 1
    var_num = var_id.max() + 1
    value = np.zeros((seq_id.max() + 1, seq_len, var_num))
    value[seq_id, time_id, var_id] = df["value"].values
    mask = np.zeros_like(value)
    mask[seq_id, time_id, var_id] = 1
    time = np.zeros((seq_id.max() + 1, seq_len))
    time[seq_id, time_id] = df["time_id"].values

    # Mannually create some missing values according to the missing rate
    if dc.get("missing_drop_rate") is not None:
        mask = mask_random(mask, dc["missing_drop_rate"])
        value = value * mask

    data_dict = {"value": value, "mask": mask, "time": time}
    return data_dict


def load_arff_array(path_data):
    p_base, p_tail = str(path_data).split(".")
    if os.path.exists(p_base+".npy"):
        data_npy = np.load(p_base+".npy", allow_pickle=True)
    elif p_tail == "arff":
        data_array = loadarff(path_data)[0]
        res_data = []
        for t_data, _ in data_array:
            t_data = np.array([d.tolist() for d in t_data])
            res_data.append(t_data)
        data_npy = np.array(res_data).swapaxes(1, 2)
        np.save(p_base+".npy", data_npy)
    else:
        raise NotImplementedError
    return data_npy


def read_uea_forecast_tvt(data_config, args, logger):
    path_dataset = args.proj_path/'data/uea_ucr/UEA'/data_config['name']
    path_tv = path_dataset/(data_config['name'] + "_TRAIN.arff")
    path_te = path_dataset/(data_config['name'] + "_TEST.arff")
    data_tv = load_arff_array(path_tv)
    data_te = load_arff_array(path_te)

    data_t, data_v = sklearn.model_selection.train_test_split(data_tv,
                                                              train_size=args.t_v_split,
                                                              random_state=args.random_state,
                                                              shuffle=True)

    data_t = pack_data_ucruea(data_t, data_config)
    data_v = pack_data_ucruea(data_v, data_config)
    data_te = pack_data_ucruea(data_te, data_config)
    return data_t, data_v, data_te
