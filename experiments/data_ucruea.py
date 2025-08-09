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

        # int(seq_len * (forecast_ratio/2)) should be at least 1
        # Check this condition and remove unqualified samples
        sample_mask = data["mask"].any(
            axis=-1).sum(axis=-1) * (self.args.len_forecast / 2) > 1
        self.data["value"] = data["value"][sample_mask]
        self.data["mask"] = data["mask"][sample_mask]
        self.data["time"] = data["time"][sample_mask]

        self.length = data["value"].shape[0]
        self.data_util = data_util

        # find the max and min ignoring masked-out values
        masked_values_max = np.where(data["mask"], data["value"], -np.inf)
        masked_values_min = np.where(data["mask"], data["value"], np.inf)
        self.d_max = np.max(masked_values_max, axis=(0, 1))
        self.d_min = np.min(masked_values_min, axis=(0, 1))
        self.t_max = self.data["time"].max()

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

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx_len = self.data["mask"][idx, ...].any(axis=-1).sum()
        idx_for = int(idx_len * (self.args.len_input / 2))
        idx_end = idx_for + int(idx_len * (self.args.len_forecast / 2))
        if self.args.ar_gen_way == "simul" and self.args.base_model == "flextsf":
            data_in = np.zeros((idx_end, self.data["value"].shape[-1]))
            data_out = data_in.copy()
            data_in[:idx_for] = self.data["value"][idx, :idx_for]
            data_out[idx_for:idx_end] = self.data["value"][idx, idx_for:idx_end]

            time_in = np.zeros((idx_end))
            time_out = time_in.copy()
            time_in[:idx_for] = self.data["time"][idx, :idx_for]
            time_out[idx_for:idx_end] = self.data["time"][idx, idx_for:idx_end]

            mask_in = np.zeros(
                (idx_end, self.data["value"][idx].shape[1]))
            mask_out = mask_in.copy()
            mask_in[:idx_for] = self.data["mask"][idx, :idx_for]
            mask_out[idx_for:idx_end] = self.data["mask"][idx, idx_for:idx_end]

        else:
            data_in = self.data["value"][idx, :idx_for]
            mask_in = self.data["mask"][idx, :idx_for]
            time_in = self.data["time"][idx, :idx_for]
            data_out = self.data["value"][idx, idx_for:idx_end]
            mask_out = self.data["mask"][idx, idx_for:idx_end]
            time_out = self.data["time"][idx, idx_for:idx_end]

        return {"data_in": data_in,
                "mask_in": mask_in,
                "time_in": time_in,
                "data_out": data_out,
                "mask_out": mask_out,
                "time_out": time_out,
                "gmean": self.mean,
                "gstd": self.std,
                "gmax": self.d_max,
                "gmin": self.d_min,
                "tmax": self.t_max}


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
    if args.full_regular == True and dc["regularity"] == "irregular":
        max_len_in = int((args.len_input / 2) * args.irreg_seq_len_max)
        max_len_out = int((args.len_forecast / 2) * args.irreg_seq_len_max)

    if args.patch_seg == "given":
        patch_len = args.patch_len
    elif dc.get("patch_len") is not None:
        patch_len = dc["patch_len"]
    elif args.patch_seg == "deterministic":
        # Automatically determine the patch length
        # Don't change the order of the following if-elif-else
        if np.array(len_in_list).mean() <= 64:
            patch_len = 4
        elif np.array(len_in_list).mean() <= 256:
            patch_len = 8
        else:
            patch_len = 16
    else:
        patch_len = None

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
        "exist_in": exist_in.float()
    }

    if args.test_last and args.len_last > 0:
        # Set the len_last entries as True and the rest as False
        mask_last = torch.from_numpy(np.stack([np.concatenate(
            [np.zeros(len_t - args.len_last), np.ones(args.len_last), np.zeros(max_len_out-len_t)], 0) for len_t in len_out_list], 0)).to(args.device).bool()
        data_dict["mask_last"] = mask_last
        data_dict["mask_out"] = data_dict["mask_out"] * mask_last.unsqueeze(-1)

    # if args.ddr > 0:
    #     # Generate a mask to randomly drop ddr of the data
    #     mask_in_drop = mask_random_tensor(data_dict["mask_in"], args.ddr)
    #     data_dict["mask_in"] = data_dict["mask_in"] * mask_in_drop
    #     data_dict["data_in"] = data_dict["data_in"] * mask_in_drop

    return data_dict


def collate_fn_forecast_irregular(batch, args, dataset_config):
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
    # Create exist_times_in/out based on the length of the time series
    exist_times_in = torch.from_numpy(np.stack(
        [np.concatenate([np.ones(len_t), np.zeros(max_len_in-len_t)], 0) for len_t in len_in_list], 0)).to(args.device).bool()
    exist_times_out = torch.from_numpy(np.stack(
        [np.concatenate([np.ones(len_t), np.zeros(max_len_out-len_t)], 0) for len_t in len_out_list], 0)).to(args.device).bool()

    gmean = torch.from_numpy(batch[0]["gmean"]).to(combined_value_in)
    gstd = torch.from_numpy(batch[0]["gstd"]).to(combined_value_in)

    data_dict = {
        "time_in": combined_time_in.float(),
        "data_in": combined_value_in.float(),
        "mask_in": combined_mask_in.float(),
        "exist_times": exist_times_in,
        "time_out": combined_time_out.float(),
        "data_out": combined_value_out.float(),
        "mask_out": combined_mask_out.float(),
        "exist_times_out": exist_times_out,
        "gmean": gmean.float(),
        "gstd": gstd.float(),
        "time_unit": torch.tensor(dc["time_unit"]).to(combined_value_in).float(),
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


def mask_random_tensor(tensor_in, ratio):
    """
    Randomly sets elements of a tensor to zero according to a specified ratio.

    Parameters:
    tensor_in (torch.Tensor): Input tensor.
    ratio (float): Probability of an element being set to zero (between 0 and 1).

    Returns:
    torch.Tensor: The masked tensor.
    """
    mask = torch.bernoulli(torch.full(
        tensor_in.shape, 1 - ratio)).to(tensor_in.device)
    return tensor_in * mask


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
