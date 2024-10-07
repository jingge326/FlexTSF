import os
import shutil
import tarfile
import pandas as pd
import numpy as np
import sklearn.model_selection
from torchvision.datasets.utils import download_url


def load_eicu_tvt(args, path_eicu, logger):
    path_processed = path_eicu/'processed'
    data_eicu = pd.read_csv(path_processed/'eicu_data.csv', index_col=0)
    return data_eicu


def download_and_process_p12(path_p12):
    urls = [
        'https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download',
        'https://physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz?download',
        'https://physionet.org/files/challenge-2012/1.0.0/set-c.tar.gz?download',
    ]
    outcome_urls = [
        'https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt',
        'https://physionet.org/files/challenge-2012/1.0.0/Outcomes-b.txt',
        'https://physionet.org/files/challenge-2012/1.0.0/Outcomes-c.txt',
    ]
    params = [
        'Weight', 'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN',
        'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg',
        'MAP', 'MechVent', 'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate',
        'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC']
    params_dict = {k: i for i, k in enumerate(params)}

    raw_folder = path_p12/"raw"
    processed_folder = path_p12/"processed"
    os.makedirs(raw_folder, exist_ok=True)

    # Download outcome data
    list_lab_df = []
    for url in outcome_urls:
        filename = url.rpartition('/')[2]
        download_url(url, raw_folder, filename, None)
        list_lab_df.append(pd.read_csv(raw_folder/filename, header=0).rename(
            columns={"RecordID": "ID", "In-hospital_death": "labels"})[["ID", "labels"]])

    labels_df = pd.concat(list_lab_df)

    os.makedirs(processed_folder, exist_ok=True)
    labels_df.to_csv(processed_folder/"p12_labels_abc.csv", index=False)

    list_data_df = []
    for url in urls:
        filename = url.rpartition('/')[2]
        download_url(url, raw_folder, filename, None)
        tar = tarfile.open(os.path.join(raw_folder, filename), 'r:gz')
        tar.extractall(raw_folder)
        tar.close()
        print('Processing {}...'.format(filename))

        dirname = os.path.join(raw_folder, filename.split('.')[0])
        files_all = [fname.split('.')[0] for fname in os.listdir(dirname)]
        files_selected = list(set(files_all) & set(map(str, labels_df["ID"])))

        list_ids_dup = []
        list_vals = []
        list_masks = []
        list_times = []

        if len(files_selected) == 0:
            continue

        for record_id in files_selected:
            prev_time = -1
            num_obs = []
            with open(os.path.join(dirname, record_id + ".txt")) as f:
                for l in f.readlines()[1:]:
                    time, param, val = l.split(',')
                    # Time in minutes
                    time = float(time.split(':')[
                        0])*60 + float(time.split(':')[1])

                    if time != prev_time:
                        list_times.append(time)
                        list_vals.append(np.zeros(len(params)))
                        list_masks.append(np.zeros(len(params)))
                        num_obs.append(np.zeros(len(params)))
                        list_ids_dup.append(record_id)
                        prev_time = time

                    if param in params_dict:
                        n_observations = num_obs[-1][params_dict[param]]
                        # integration by average
                        if n_observations > 0:
                            prev_val = list_vals[-1][params_dict[param]]
                            new_val = (prev_val * n_observations +
                                       float(val)) / (n_observations + 1)
                            list_vals[-1][params_dict[param]] = new_val
                        else:
                            list_vals[-1][params_dict[param]] = float(val)
                        list_masks[-1][params_dict[param]] = 1
                        num_obs[-1][params_dict[param]] += 1
                    else:
                        print("Omitting param {}".format(param))
                        # assert param == 'RecordID', 'Unexpected param {}'.format(
                        #     param)

        arr_values = np.stack(list_vals, axis=0)
        arr_masks = np.stack(list_masks, axis=0)
        df_times = pd.DataFrame(list_times, columns=['Time'])

        df_values = pd.DataFrame(arr_values, columns=[
            'Value_'+str(i) for i in params_dict.values()])
        df_mask = pd.DataFrame(
            arr_masks, columns=['Mask_'+str(i) for i in params_dict.values()])

        df_p12 = pd.concat([pd.DataFrame(list_ids_dup, columns=[
            'ID']), df_times, df_values, df_mask], axis=1)
        list_data_df.append(df_p12)

    df_p12_data = pd.concat(list_data_df)
    df_p12_data.to_csv(processed_folder/'p12_data_abc.csv', index=False)


def load_p12_tvt(args, path_p12, logger):
    path_processed = path_p12/"processed"
    path_raw = path_p12/"raw"
    if os.path.exists(path_processed/'p12_data_abc.csv') and os.path.exists(path_processed/'p12_labels_abc.csv'):
        pass
    else:
        if os.path.exists(path_raw):
            shutil.rmtree(path_raw)
        if os.path.exists(path_processed):
            shutil.rmtree(path_processed)
        download_and_process_p12(path_p12)

    data_tvt = pd.read_csv(path_processed/'p12_data_abc.csv', index_col=0)

    return data_tvt


def read_ehr_forecast_tvt(data_config, args, logger):
    root_path = args.proj_path
    if data_config["name"] == "PhysioNet2012":
        path_p12 = root_path/'data/ehr/PhysioNet12'
        path_raw = path_p12/'raw'
        path_processed = path_p12/'processed'
        if os.path.exists(path_processed/'p12_data_abc.csv'):
            pass
        else:
            if os.path.exists(path_raw):
                shutil.rmtree(path_raw)
            if os.path.exists(path_processed):
                shutil.rmtree(path_processed)
            download_and_process_p12(path_p12)

        data_tvt = pd.read_csv(path_processed/'p12_data_abc.csv', index_col=0)

    elif data_config["name"] == "eICU":
        data_tvt = pd.read_csv(
            root_path/'data/ehr/eicu/processed/eicu_data.csv', index_col=0)
        # Take the timestamp of the first observation as the starting point
        # Reassign the time to be the time difference from the first observation
        data_tvt["Time"] = data_tvt.groupby(data_tvt.index)["Time"].transform(
            lambda x: x - x.min())
        
    else:
        raise NotImplementedError

    # Apply some filstering
    # For Extrap: there are values before and after 24h
    # For Classf: the patient didn't die within the first 24h
    ids_before = data_tvt.loc[data_tvt['Time']
                              < args.next_start].index.unique()
    ids_after = data_tvt.loc[data_tvt['Time']
                             > args.next_start].index.unique()
    ids_selected = set(ids_before) & set(ids_after)
    data_tvt = data_tvt.loc[list(ids_selected)]
    # drop admissions which have less than 20 time steps
    len_counts = data_tvt.index.value_counts()
    ids_good = len_counts.index[len_counts >= 20]
    data_tvt = data_tvt.loc[data_tvt.index.isin(ids_good)]
    # Only extract the data within the max time range
    if args.time_max < data_tvt['Time'].max():
        data_tvt = data_tvt.loc[data_tvt['Time'] <= args.time_max]

    value_cols = []
    mask_cols = []
    for col in data_tvt.columns:
        value_cols.append(col.startswith("Value"))
        mask_cols.append(col.startswith("Mask"))

    len_max = len_counts.max()
    data_in = []
    mask_in = []
    time_in = []
    df_tmp = pd.DataFrame(0, index=range(len_max), columns=data_tvt.columns)
    for idx, group in data_tvt.groupby(data_tvt.index):
        group = pd.concat([group, df_tmp.iloc[group.shape[0]:, :]], axis=0)
        data_in.append(group.loc[:, value_cols].values)
        mask_in.append(group.loc[:, mask_cols].values)
        time_in.append(group["Time"].values)

    data_in = np.stack(data_in, axis=0)
    mask_in = np.stack(mask_in, axis=0)
    time_in = np.stack(time_in, axis=0)
    exist_times = mask_in.sum(axis=-1) > 0

    data_ts_train = {}
    data_ts_val = {}
    data_ts_test = {}

    train_idx, v_t_idx = sklearn.model_selection.train_test_split(
        np.arange(data_in.shape[0]),
        train_size=0.8,
        random_state=args.random_state,
        shuffle=True)
    val_idx, test_idx = sklearn.model_selection.train_test_split(
        v_t_idx,
        train_size=0.5,
        random_state=args.random_state,
        shuffle=True)

    data_ts_train["value"] = data_in[train_idx, ...]
    data_ts_train["mask"] = mask_in[train_idx, ...]
    data_ts_train["time"] = time_in[train_idx, ...]
    data_ts_train["exist_times"] = exist_times[train_idx, ...]
    data_ts_val["value"] = data_in[val_idx, ...]
    data_ts_val["mask"] = mask_in[val_idx, ...]
    data_ts_val["time"] = time_in[val_idx, ...]
    data_ts_val["exist_times"] = exist_times[val_idx, ...]
    data_ts_test["value"] = data_in[test_idx, ...]
    data_ts_test["mask"] = mask_in[test_idx, ...]
    data_ts_test["time"] = time_in[test_idx, ...]
    data_ts_test["exist_times"] = exist_times[test_idx, ...]

    return data_ts_train, data_ts_val, data_ts_test
