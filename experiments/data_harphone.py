import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.utils_exp import StandardScaler


class Dataset_HARPhone(Dataset):
    def __init__(self, args, type, logger, data_util={}):

        self.data_util = data_util
        self.logger = logger
        path_har = args.proj_path/"data/har/phone"
        data = torch.load(path_har/"{}.pt".format(type))
        data_x = data["samples"]
        data_y = data["labels"]

        data_x = data_x.permute(0, 2, 1)

        if self.data_util.get("scaler") is None:
            self.data_util["scaler"] = StandardScaler(
                mean=data_x.mean(dim=(0, 1)), std=data_x.std(dim=(0, 1)))
        data_x = self.data_util["scaler"].transform(data_x)

        if args.debug_portion != -1:
            data_x = data_x[:int(data_x.shape[0]*args.debug_portion)]
            data_y = data_y[:int(data_y.shape[0]*args.debug_portion)]

        self.data_x = data_x
        self.data_y = data_y
        self.length = data_x.shape[0]
        self.mean = self.data_util["scaler"].mean
        self.std = self.data_util["scaler"].std
        self.logger.info(type + " data length: " + str(self.length))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {"idx": idx,
                "data_in": self.data_x[idx, ...],
                "labels": self.data_y[idx, ...],
                "gmean": self.mean,
                "gstd": self.std}



def read_harphone_forecast_tvt(dc, args, logger):
    path_har = args.proj_path/"data/har/phone"        
    data1 = torch.load(path_har/"train.pt")
    data2 = torch.load(path_har/"val.pt")
    data3 = torch.load(path_har/"test.pt")

    datas = []
    for data in [data1, data2, data3]:
        value = data["samples"].permute(0, 2, 1).detach().cpu().numpy()
        time = np.tile(np.arange(0, value.shape[1])[
            np.newaxis, :], [value.shape[0], 1])
        datas.append({"value": value, "time": time,
                     "mask": np.ones_like(value)})

    return datas
