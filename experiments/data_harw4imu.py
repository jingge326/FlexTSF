import os
import shutil
import numpy as np
import sklearn

from torchvision.datasets.utils import download_url


def download_and_process_harw4imu(path_harw4imu, reduce='average', max_seq_length=50, n_samples=None):

    urls = [
        'https://archive.ics.uci.edu/ml/machine-learning-databases/00196/ConfLongDemo_JSI.txt',
    ]

    tag_ids = [
        '010-000-024-033',  # 'ANKLE_LEFT',
        '010-000-030-096',  # 'ANKLE_RIGHT',
        '020-000-033-111',  # 'CHEST',
        '020-000-032-221'  # 'BELT'
    ]

    tag_dict = {k: i for i, k in enumerate(tag_ids)}

    label_names = [
        'walking',
        'falling',
        'lying down',
        'lying',
        'sitting down',
        'sitting',
        'standing up from lying',
        'on all fours',
        'sitting on the ground',
        'standing up from sitting',
        'standing up from sitting on the ground'
    ]

    # label_dict = {k: i for i, k in enumerate(label_names)}

    # Merge similar labels into one class
    label_dict = {
        'walking': 0,
        'falling': 1,
        'lying': 2,
        'lying down': 2,
        'sitting': 3,
        'sitting down': 3,
        'standing up from lying': 4,
        'standing up from sitting': 4,
        'standing up from sitting on the ground': 4,
        'on all fours': 5,
        'sitting on the ground': 6
    }

    raw_folder = path_harw4imu/"raw"
    processed_folder = path_harw4imu/"processed"

    # # check_exists
    # for url in urls:
    #     filename = url.rpartition('/')[2]
    #     if os.path.exists(os.path.join(processed_folder, 'harw4imu_data.csv')):
    #         return

    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(processed_folder, exist_ok=True)

    def save_record(records, record_id, tt, vals, mask, labels):
        assert min(labels) >= 0

        tt = np.array(tt).astype('int')

        vals = np.stack(vals)
        mask = np.stack(mask)
        labels = np.stack(labels)

        # flatten the measurements for different tags
        vals = vals.reshape(vals.shape[0], -1)
        mask = mask.reshape(mask.shape[0], -1)
        assert (len(tt) == vals.shape[0])
        assert (mask.shape[0] == vals.shape[0])
        assert (labels.shape[0] == vals.shape[0])

        # records.append((record_id, tt, vals, mask, labels))

        seq_length = len(tt)
        # split the long time series into smaller ones
        offset = 0
        slide = max_seq_length // 2

        while (offset + max_seq_length < seq_length):
            idx = range(offset, offset + max_seq_length)

            first_tp = tt[idx][0]
            records.append(
                (record_id, tt[idx] - first_tp, vals[idx], mask[idx], labels[idx]))
            offset += slide

    for url in urls:
        filename = url.rpartition('/')[2]
        download_url(url, raw_folder, filename, None)

        print('Processing {}...'.format(filename))

        dirname = os.path.join(raw_folder)
        records = []
        first_tp = None

        for txtfile in os.listdir(dirname):
            with open(os.path.join(dirname, txtfile)) as f:
                lines = f.readlines()
                prev_time = -1
                tt = []

                record_id = None
                for l in lines:
                    cur_record_id, tag_id, time, date, val1, val2, val3, label = l.strip().split(',')
                    value_vec = np.array(
                        (float(val1), float(val2), float(val3)))
                    time = float(time)

                    if cur_record_id != record_id:
                        if record_id is not None:
                            save_record(records, record_id,
                                        tt, vals, mask, labels)
                        tt, vals, mask, nobs, labels = [], [], [], [], []
                        record_id = cur_record_id

                        tt = [np.zeros(1)]
                        vals = [np.zeros((len(tag_ids), 3))]
                        mask = [np.zeros((len(tag_ids), 3))]
                        nobs = [np.zeros(len(tag_ids))]
                        labels = [-1]

                        first_tp = time
                        time = round((time - first_tp) / 10**5)
                        prev_time = time
                    else:
                        # for speed -- we actually don't need to quantize it in Latent ODE
                        # quatizing by 100 ms. 1,000 is one millisecond, 1,000,000 is one second
                        time = round((time - first_tp) / 10**5)

                    if time != prev_time:
                        tt.append(time)
                        vals.append(np.zeros((len(tag_ids), 3)))
                        mask.append(np.zeros((len(tag_ids), 3)))
                        nobs.append(np.zeros(len(tag_ids)))
                        if labels[-1] != -1:
                            labels.append(-1)
                        prev_time = time

                    if tag_id in tag_ids:
                        n_observations = nobs[-1][tag_dict[tag_id]]
                        if (reduce == 'average') and (n_observations > 0):
                            prev_val = vals[-1][tag_dict[tag_id]]
                            new_val = (prev_val * n_observations +
                                       value_vec) / (n_observations + 1)
                            vals[-1][tag_dict[tag_id]] = new_val
                        else:
                            vals[-1][tag_dict[tag_id]] = value_vec

                        mask[-1][tag_dict[tag_id]] = 1
                        nobs[-1][tag_dict[tag_id]] += 1

                        if label in label_names:
                            if labels[-1] == -1:
                                labels[-1] = label_dict[label]
                        else:
                            print('Read unexpected label {}'.format(label))
                    else:
                        assert tag_id == 'RecordID', 'Read unexpected tag id {}'.format(
                            tag_id)
                save_record(records, record_id, tt, vals, mask, labels)

        np.save(
            os.path.join(processed_folder, 'harw4imu_data.npy'),
            records
        )

    print('Done!')


def read_harw4imu_forecast_tvt(dc, args, logger):
    path_data = args.proj_path/'data/har/4imu'
    path_processed = path_data/"processed"
    path_raw = path_data/"raw"
    if os.path.exists(path_processed/'harw4imu_data.npy'):
        pass
    else:
        if os.path.exists(path_raw):
            shutil.rmtree(path_raw)
        if os.path.exists(path_processed):
            shutil.rmtree(path_processed)
        download_and_process_harw4imu(path_data)

    data_tvt = np.load(
        path_processed/'harw4imu_data.npy', allow_pickle=True)

    data_t, data_vt = sklearn.model_selection.train_test_split(
        data_tvt,
        train_size=0.8,
        random_state=args.random_state,
        shuffle=True)

    data_v, data_t = sklearn.model_selection.train_test_split(
        data_vt,
        train_size=0.5,
        random_state=args.random_state,
        shuffle=True)

    datas = []
    for dd in [data_t, data_v, data_t]:
        data = {"value": [], "time": [], "mask": []}
        for arr in dd:
            # check if the time_in is in increasing order, if not, print time_in and skip this record
            if not np.all(np.diff(arr[1]) > 0):
                # logger.info(f"record_id: {arr[0]}, time_in: {arr[1]}")
                continue
            data["value"].append(arr[2][np.newaxis, ...])
            data["time"].append(arr[1][np.newaxis, ...])
            data["mask"].append(arr[3][np.newaxis, ...])

        data["value"] = np.concatenate(data["value"], axis=0)
        data["time"] = np.concatenate(data["time"], axis=0)
        data["mask"] = np.concatenate(data["mask"], axis=0)

        datas.append(data)

    return datas
