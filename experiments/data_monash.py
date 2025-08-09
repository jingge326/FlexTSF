import math
import numpy as np
import torch
from torch.utils.data import Dataset


def read_monash_pretrain_data(args):
    root_path = args.proj_path
    # path_data = root_path/'data/monash/processed/monash_tsc_1024_min18_all_subset'
    path_data = root_path/'data/monash/processed/monash_tsc_1024_min18_all'

    content = np.load(str(path_data)+".npz", allow_pickle=True)
    data_dict = content["data_dict"].item()

    return data_dict


class DatasetGP_Monash(Dataset):
    def __init__(self, data_dict, args):
        self.args = args
        self.data = data_dict
        self.length = len(data_dict)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # The total length is input length + output length
        # The input length is randomly sampled from [seq_len_min, seq_len_max]
        # The output length is 1/8 of the input length
        seq_len_max = min(math.floor(
            self.args.seq_len_max * 2), self.data[idx]["len"])
        seq_len_min = math.floor(self.args.seq_len_min * self.args.tir)
        len_total = np.random.randint(seq_len_min, seq_len_max + 1)
        len_in = math.floor(len_total * (1/self.args.tir))
        item = self.data[idx]
        start_idx = np.random.randint(0, self.data[idx]["len"] - len_total + 1)
        if self.args.ar_gen_way == "simul" and self.args.base_model == "flextsf":
            # data_in and data_out have the same length (len_in + len_fore).
            # data_in has only the beginning part filled with data while the rest is filled with zeros
            # data_out has only the ending part filled with data while the rest is filled with zeros
            data_in = np.zeros(len_total)
            data_out = data_in.copy()
            data_in[:len_in] = item["value"][start_idx:start_idx + len_in]
            data_out[len_in:len_total] = item["value"][start_idx +
                                                       len_in:start_idx + len_total]

            time_in = np.zeros(len_total)
            time_out = time_in.copy()
            time_in[:len_in] = item["time"][start_idx:start_idx +
                                            len_in] - item["time"][start_idx]
            time_out[len_in:len_total] = item["time"][start_idx +
                                                      len_in:start_idx + len_total] - item["time"][start_idx]

        else:
            data_in = item["value"][start_idx:start_idx + len_in]
            data_out = item["value"][start_idx + len_in:start_idx + len_total]
            time_in = item["time"][start_idx:start_idx + len_in]
            time_out = item["time"][start_idx + len_in:start_idx + len_total]
            # Calculate the relative time
            time_out = time_out - time_in[0]
            time_in = time_in - time_in[0]

        gmean = item["gmean"]
        gstd = item["gstd"]
        time_unit = item["time_unit"]

        return {"data_name": item["data_name"],
                "data_in": data_in,
                "data_out": data_out,
                "time_in": time_in,
                "time_out": time_out,
                "gmean": gmean,
                "gstd": gstd,
                "time_unit": time_unit}


def collate_fn_forecast_monash(batch, args):
    # Mainly process data_in, labels
    value_in_list = []
    time_in_list = []
    len_in_list = []
    value_out_list = []
    time_out_list = []
    len_out_list = []
    gmean_list = []
    gstd_list = []
    time_unit_list = []
    data_names = []
    for b in batch:
        value_in_list.append(b["data_in"])
        time_in_list.append(b["time_in"])
        len_in_list.append(len(b["data_in"]))
        value_out_list.append(b["data_out"])
        time_out_list.append(b["time_out"])
        len_out_list.append(len(b["data_out"]))
        gmean_list.append(b["gmean"])
        gstd_list.append(b["gstd"])
        time_unit_list.append(b["time_unit"])
        data_names.append(b["data_name"])

    max_len_in = max(len_in_list)
    max_len_out = max(len_out_list)

    combined_value_in = torch.from_numpy(np.stack([np.concatenate([values, np.zeros(
        max_len_in - len_t)]) for values, len_t in zip(value_in_list, len_in_list)], 0))
    combined_time_in = torch.from_numpy(np.stack([np.concatenate([time, np.zeros(
        max_len_in - len_t)]) for time, len_t in zip(time_in_list, len_in_list)], 0))
    combined_value_out = torch.from_numpy(np.stack([np.concatenate([values, np.zeros(
        max_len_out - len_t)]) for values, len_t in zip(value_out_list, len_out_list)], 0))
    combined_time_out = torch.from_numpy(np.stack([np.concatenate([time, np.zeros(
        max_len_out - len_t)]) for time, len_t in zip(time_out_list, len_out_list)], 0))
    exist_in = torch.from_numpy(np.stack([np.concatenate(
        [np.ones(len_t), np.zeros(max_len_in-len_t)], 0) for len_t in len_in_list], 0))

    combined_mask_in = torch.from_numpy(np.stack([np.concatenate([np.ones(len_t), np.zeros(
        max_len_in - len_t)]) for len_t in len_in_list], 0))
    combined_mask_out = torch.from_numpy(np.stack([np.concatenate([np.ones(len_t), np.zeros(
        max_len_out - len_t)]) for len_t in len_out_list], 0))

    combined_gmean = torch.tensor(gmean_list)
    combined_gstd = torch.tensor(gstd_list)
    combined_time_unit = torch.tensor(time_unit_list)

    if args.ml_task == "uni_pretrain" and args.patch_len_pretrain != 0:
        patch_len = args.patch_len_pretrain
    else:
        if np.array(len_in_list).mean() <= 64:
            patch_len = 4
        elif np.array(len_in_list).mean() <= 256:
            patch_len = 8
        else:
            patch_len = 16

    data_dict = {
        "time_in": combined_time_in.float(),
        "data_in": combined_value_in.float().unsqueeze(-1),
        "mask_in": combined_mask_in.float().unsqueeze(-1),
        "time_out": combined_time_out.float(),
        "data_out": combined_value_out.float().unsqueeze(-1),
        "mask_out": combined_mask_out.float().unsqueeze(-1),
        "gmean": combined_gmean.float().unsqueeze(-1),
        "gstd": combined_gstd.float().unsqueeze(-1),
        "time_unit": combined_time_unit.float().unsqueeze(-1),
        "patch_len": patch_len,
        "data_names": data_names,
        "exist_in": exist_in.float()
    }

    return data_dict
