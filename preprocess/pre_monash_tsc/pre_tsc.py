from pathlib import Path
import os

import numpy as np


def read_tsc_pretrain_data(root_path):
    path_data = root_path/'data/tsc/processed/tsc_all_1024_10k'
    # path_data = root_path/'data/monash/processed/tsc_all_1024'
    # quick_test = 0

    # Check if the data has been processed before
    if os.path.exists(str(path_data)+".npz"):
        content = np.load(str(path_data)+".npz", allow_pickle=True)
        data_dict = content["data_dict"].item()
    else:
        data_dict = {}
        cnt = 0
        with open(str(path_data)+".csv", "r") as fin:
            next(fin)
            for line in fin:
                items = line.strip().split(";")
                data_name = items[0]
                # split and convert to numpy.array
                value = np.float32(np.array(items[1].split(",")))
                time = np.float32(np.array(items[2].split(",")))
                seq_len = len(value)
                # Calculate the mean and std of the non-missing values
                gmean = np.mean(value)
                gstd = np.std(value)
                # Normalize the data
                value = (value - gmean) / (gstd + 1e-6)

                if gstd > 1e6:
                    print(f"Data {data_name} has too large std: {gstd}")
                    continue

                data_dict[cnt] = {"data_name": data_name,
                                  "value": value,
                                  "time": time,
                                  "gmean": gmean,
                                  "gstd": gstd,
                                  "time_unit": 0,
                                  "len": seq_len}
                cnt += 1
                if cnt % 1000 == 0:
                    print(f"Processed {cnt} rows")

                # quick_test += 1
                # if quick_test == 10000:
                #     break

        # Save the data_dict to a file
        np.savez(str(path_data)+".npz", data_dict=data_dict)

    return data_dict


def extract_series_tsc(root_path):
    seq_len = 1024
    sub_series_overlap = 0.5
    len_min = 17

    datasets_to_ignore = ["SpokenArabicDigits", "CharacterTrajectories"]

    path_data = root_path/'data/tsc'
    path_raw = path_data/'raw'

    # iterate over all files in path_raw
    for data_folder in os.listdir(path_raw):
        print(data_folder)
        if data_folder in datasets_to_ignore:
            continue
        with open(path_raw/data_folder/f"seq_{data_folder}.csv", "w") as fout:
            fout.write("source;series;times\n")
            for data_file in os.listdir(path_raw/data_folder):
                if data_file.endswith(".ts"):
                    with open(path_raw/data_folder/data_file, "r") as fin:
                        for line in fin:
                            # Strip white space from start/end of line
                            line = line.strip()
                            if line[0] == "@" or line[0] == "#":
                                continue
                            series_list = line.split(":")[:-1]
                            for series in series_list:
                                # series = series.replace("?", "")
                                series = series.split(",")
                                times = list(range(0, len(series)))
                                # eliminate missing values
                                # get all the positions of not "?"
                                idx_good = [i for i, x in enumerate(
                                    series) if x != "?"]
                                if len(idx_good) < len_min:
                                    continue
                                # get the values of not "?"
                                times = [times[i] for i in idx_good]
                                series = [series[i] for i in idx_good]

                                sub_series_step = int(
                                    seq_len * (1 - sub_series_overlap))

                                for i in range(0, len(series), sub_series_step):
                                    if len(series) - i < len_min:
                                        break
                                    idx_end = i + seq_len
                                    if idx_end > len(series):
                                        idx_end = len(series)
                                    str_line = data_file[:-4] + ";" + ",".join(series[i:idx_end]) + ";" + ",".join(
                                        [str(x) for x in times[i:idx_end]])
                                    fout.write(str_line + "\n")


if __name__ == "__main__":

    root_path = Path(__file__).parents[2]

    extract_series_tsc(root_path)

    # read_tsc_pretrain_data(root_path)
