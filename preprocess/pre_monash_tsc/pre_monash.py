from pathlib import Path
import os

import numpy as np


def draw_subset(root_path):
    path_data = root_path/'data/monash/processed/monash_tsc_1024_min18_all'
    content = np.load(str(path_data)+".npz", allow_pickle=True)
    data_dict = content["data_dict"].item()
    # randomly draw 20k samples
    keys = np.random.choice(list(data_dict.keys()), 20000, replace=False)
    data_dict_subset = {}
    for i, key in enumerate(keys):
        data_dict_subset[i] = data_dict[key]
    np.savez(str(path_data)+"_subset.npz", data_dict=data_dict_subset)


def transform_to_npz(root_path):
    # path_data = root_path/'data/monash/processed/monash_tsc_1024_min18_20k'
    # Added FordA and FordB
    path_data = root_path/'data/monash/processed/monash_tsc_1024_min18_all'

    # Check if the data has been processed before
    if os.path.exists(str(path_data)+".npz"):
        content = np.load(str(path_data)+".npz", allow_pickle=True)
        data_dict = content["data_dict"].item()
    else:
        path_time_unit = root_path/"experiments/monash_tsc_time_unit.csv"
        info_dict = {}
        with open(path_time_unit, "r") as fin:
            next(fin)
            for line in fin:
                items = line.strip().split(",")
                data_name = items[0]
                time_unit = items[1]
                info_dict[data_name] = float(time_unit)
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

                if data_name in info_dict:
                    time_unit = info_dict[data_name]
                else:
                    time_unit = -1.0

                data_dict[cnt] = {"data_name": data_name,
                                  "value": value,
                                  "time": time,
                                  "gmean": gmean,
                                  "gstd": gstd,
                                  "time_unit": time_unit,
                                  "len": seq_len}
                cnt += 1
                if cnt % 10000 == 0:
                    print(f"Processed {cnt} rows")

        # Save the data_dict to a file
        np.savez(str(path_data)+".npz", data_dict=data_dict)


def list_files_walk(start_path='.', ext='ts'):
    list_files = []
    for root, dirs, files in os.walk(start_path):
        for file in files:
            if file.endswith(ext):
                list_files.append(os.path.join(root, file))
    return list_files


def count_lines(root_path):
    folder_tsc = root_path/'data/tsc/raw'
    folder_monash = root_path/'data/monash/raw'
    list_files_tsc = list_files_walk(folder_tsc, ext='csv')
    list_files_monash = list_files_walk(folder_monash, ext='csv')
    list_files_all = list_files_tsc + list_files_monash
    # count the number of lines in all files and save to a file
    with open(root_path/'preprocess/pre_monash_tsc/line_count.csv', 'w') as f:
        for file in list_files_all:
            with open(file, 'r') as fin:
                lines = fin.readlines()
                f.write(f"{file},{len(lines)}\n")


def merge_files(root_path):
    folder_tsc = root_path/'data/tsc/raw'
    folder_monash = root_path/'data/monash/raw'
    list_files_tsc = list_files_walk(folder_tsc, ext='csv')
    list_files_monash = list_files_walk(folder_monash, ext='csv')
    list_files_all = list_files_tsc + list_files_monash
    # count the number of lines in all files and save to a file
    with open(root_path/'data/monash/processed/monash_tsc_1024_min18_all.csv', 'w') as f:
        print(f"Total files: {len(list_files_all)}")
        cnt = 0
        for file in list_files_all:
            # Count the number of lines in the file
            # If the number of lines is less than 200k, then write the file to the merged file
            # Otherwise, randomly sample 200k lines and write to the merged file
            with open(file, 'r') as fin:
                # skip the first line
                next(fin)
                lines = fin.readlines()
                if len(lines) < 200000:
                    f.writelines(lines)
                else:
                    lines_sample = np.random.choice(
                        lines, 200000, replace=False)
                    f.writelines(lines_sample)
                cnt += 1
                print(f"Processed {cnt}th file: {file}")


def extract_series_tsc(root_path):
    seq_len = 1024
    sub_series_overlap = 0.5
    len_min = 18

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


def extract_series_monash(root_path):
    seq_len = 1024
    sub_series_overlap = 0.5
    len_min = 18

    path_data = root_path/'data/monash'
    path_raw = path_data/'raw'

    # iterate over all files in path_raw
    for data_file in os.listdir(path_raw):
        if data_file.endswith(".tsf"):
            # skip files whose names contain "without_missing"
            if "without_missing" in data_file:
                continue
            col_names = []
            col_types = []
            all_data = {}
            line_count = 0
            found_data_tag = False
            started_reading_data_section = False

            with open(path_raw/data_file, "r", encoding="cp1252") as fin, open(path_raw/f"seq_{data_file[:-4]}.csv", "w") as fout:
                fout.write("source;series;times\n")
                for line in fin:
                    # Strip white space from start/end of line
                    line = line.strip()

                    if line:
                        if line.startswith("@"):  # Read meta-data
                            if not line.startswith("@data"):
                                line_content = line.split(" ")
                                if line.startswith("@attribute"):
                                    # Attributes have both name and type
                                    if len(line_content) != 3:
                                        raise Exception(
                                            "Invalid meta-data specification.")
                                    col_names.append(line_content[1])
                                    col_types.append(line_content[2])
                                else:
                                    # Other meta-data have only values
                                    if len(line_content) != 2:
                                        raise Exception(
                                            "Invalid meta-data specification.")
                            else:
                                if len(col_names) == 0:
                                    raise Exception(
                                        "Missing attribute section. Attribute section must come before data.")

                                found_data_tag = True
                        elif not line.startswith("#"):
                            if len(col_names) == 0:
                                raise Exception(
                                    "Missing attribute section. Attribute section must come before data.")
                            elif not found_data_tag:
                                raise Exception("Missing @data tag.")
                            else:
                                if not started_reading_data_section:
                                    started_reading_data_section = True
                                    for col in col_names:
                                        all_data[col] = []

                                full_info = line.split(":")

                                if len(full_info) != (len(col_names) + 1):
                                    raise Exception(
                                        "Missing attributes/values in series.")

                                series = full_info[len(full_info) - 1]
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

                        line_count = line_count + 1

                if line_count == 0:
                    raise Exception("Empty file.")


if __name__ == "__main__":

    root_path = Path(__file__).parents[2]

    # extract_series_monash(root_path)

    extract_series_tsc(root_path)

    count_lines(root_path)

    merge_files(root_path)

    transform_to_npz(root_path)

    draw_subset(root_path)
