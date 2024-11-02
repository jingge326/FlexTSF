import os
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset

import warnings
from experiments.utils_exp import StandardScaler


warnings.filterwarnings('ignore')

fnames = {
    "etth1": "ltf/ETT-small/ETTh1.csv",
    "etth2": "ltf/ETT-small/ETTh2.csv",
    "ettm1": "ltf/ETT-small/ETTm1.csv",
    "ettm2": "ltf/ETT-small/ETTm2.csv",
    "electricity": "ltf/electricity/electricity.csv",
    "exchange_rate": "ltf/exchange_rate/exchange_rate.csv",
    "illness": "ltf/illness/national_illness.csv",
    "traffic": "ltf/traffic/traffic.csv",
    "weather": "ltf/weather/weather.csv",
}


def read_etth_forecast_tvt(dc, args, logger, target='OT'):

    data_path = "data/{}".format(fnames[args.data_name])

    df_raw = pd.read_csv(os.path.join(args.proj_path, data_path))

    border1s = [0, 12 * 30 * 24 - args.ltf_input_len,
                12 * 30 * 24 + 4 * 30 * 24 - args.ltf_input_len]
    border2s = [12 * 30 * 24, 12 * 30 * 24 +
                4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

    data_tvt = []
    for itype in range(3):

        if args.ltf_features == 'M' or args.ltf_features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif args.ltf_features == 'S':
            df_data = df_raw[[target]]
        data = df_data.values

        border1 = border1s[itype]
        border2 = border2s[itype]
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)


        # Calculate the relative time in hours
        # Obtaining the time delta in time_unit intervals
        time_unit = dc["time_unit"]
        time = (df_stamp['date'] - df_stamp['date'].iloc[0]).dt.total_seconds(
        ) / time_unit

        data_tvt.append(
            {"value": data[border1:border2], "time": time.values})

    return data_tvt


def read_ettm_forecast_tvt(dc, args, logger, target='OT'):
    data_path = "data/{}".format(fnames[args.data_name])

    df_raw = pd.read_csv(os.path.join(args.proj_path, data_path))

    border1s = [0, 12 * 30 * 24 * 4 - args.ltf_input_len, 12 *
                30 * 24 * 4 + 4 * 30 * 24 * 4 - args.ltf_input_len]
    border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 *
                30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]

    data_tvt = []
    for itype in range(3):

        if args.ltf_features == 'M' or args.ltf_features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif args.ltf_features == 'S':
            df_data = df_raw[[target]]
        data = df_data.values

        border1 = border1s[itype]
        border2 = border2s[itype]
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        # Calculate the relative time in hours
        # Obtaining the time delta in time_unit intervals
        time_unit = dc["time_unit"]
        time = (df_stamp['date'] - df_stamp['date'].iloc[0]).dt.total_seconds(
        ) / time_unit

        data_tvt.append(
            {"value": data[border1:border2], "time": time.values})

    return data_tvt


def read_ltfcustom_forecast_tvt(dc, args, logger, target='OT'):
    data_path = "data/{}".format(fnames[args.data_name])

    df_raw = pd.read_csv(os.path.join(args.proj_path, data_path))

    cols = list(df_raw.columns)
    cols.remove(target)
    cols.remove('date')
    df_raw = df_raw[['date'] + cols + [target]]
    # print(cols)
    num_train = int(len(df_raw) * 0.7)
    num_test = int(len(df_raw) * 0.2)
    num_vali = len(df_raw) - num_train - num_test
    border1s = [0, num_train - args.ltf_input_len,
                len(df_raw) - num_test - args.ltf_input_len]
    border2s = [num_train, num_train + num_vali, len(df_raw)]

    data_tvt = []
    for itype in range(3):

        if args.ltf_features == 'M' or args.ltf_features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif args.ltf_features == 'S':
            df_data = df_raw[[target]]
        data = df_data.values

        border1 = border1s[itype]
        border2 = border2s[itype]
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        # Calculate the relative time in hours
        # Obtaining the time delta in time_unit intervals
        time_unit = dc["time_unit"]
        time = (df_stamp['date'] - df_stamp['date'].iloc[0]).dt.total_seconds(
        ) / time_unit

        data_tvt.append(
            {"value": data[border1:border2], "time": time.values})

    return data_tvt


class Dataset_Forecast_Regular_Continuous(Dataset):
    def __init__(self, data, data_util, args):
        self.data = data
        self.data_util = data_util
        self.args = args
        self.pred_len = args.ltf_pred_len

        self.length = self.data["value"].shape[0] - \
            self.args.ltf_input_len - self.pred_len + 1

        if self.data_util.get("scaler") is None:
            d_mean = np.mean(self.data["value"], axis=0)
            d_std = np.std(self.data["value"], axis=0)
            self.data_util["scaler"] = StandardScaler(mean=d_mean, std=d_std)

        self.data["value"] = self.data_util["scaler"].transform(data["value"])
        self.mean = self.data_util["scaler"].mean
        self.std = self.data_util["scaler"].std

        self.data_x = self.data["value"]
        self.data_y = self.data["value"].copy()
        self.mask_in = np.ones_like(self.data_x)
        self.mask_out = np.ones_like(self.data_y)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.args.ltf_input_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        time_in = self.data["time"][s_begin:s_end]
        time_out = self.data["time"][r_begin:r_end]
        mask_in = self.mask_in[s_begin:s_end]
        mask_out = self.mask_out[r_begin:r_end]

        # Turn the first timestamp to 0
        time_start = time_in[0]
        time_in = time_in - time_start
        time_out = time_out - time_start

        return {"data_in": seq_x,
                "data_out": seq_y,
                "time_in": time_in,
                "time_out": time_out,
                "mask_in": mask_in,
                "mask_out": mask_out,
                "gmean": self.mean,
                "gstd": self.std}

    def __len__(self):
        return self.length
