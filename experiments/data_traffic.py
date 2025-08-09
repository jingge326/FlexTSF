import numpy as np


def read_traffic_forecast_tvt(dc, args, logger):
    if args.data_name == "metrla":
        path_traffic = args.proj_path/'data/traffic/metr-la'
    elif args.data_name == "pems_bay":
        path_traffic = args.proj_path/'data/traffic/pems-bay'
    else:
        raise ValueError("Unknown dataset: {}".format(args.data))

    datas = []
    for type in ["train", "val", "test"]:
        dd = np.load(path_traffic/'{}.npz'.format(type))
        value = np.concatenate([dd["x"][..., 0], dd["y"][..., 0]], axis=1)
        # [...,0]： just take timestamp of one variable is sufficient
        time = np.concatenate(
            [dd["x"][..., 1][..., 0], dd["y"][..., 1][..., 0]], axis=1)

        # select series whose timestamps are in increasing order
        series_good_idx = np.where((time[:, 1:] - time[:, :-1] > 0).all(1))[0]
        value = value[series_good_idx]
        time = time[series_good_idx]

        # The starting time of each sequence should be 0
        time = time - time[:, 0:1]

        mask = (value != 0).astype(np.float32)

        # select series who have at least four valid timestamps so that we have at least two timestamps for the input
        valid_samples = mask.any(-1).sum(-1) >= 4
        value = value[valid_samples]
        time = time[valid_samples]
        mask = mask[valid_samples]

        # There some samples that all values are 0, we take them as invalid samples and remove them
        exist_times = mask.any(-1)
        exist_samples = exist_times.any(-1)
        value = value[exist_samples]
        time = time[exist_samples]
        mask = mask[exist_samples]
        exist_times = exist_times[exist_samples]
        datas.append({"value": value, "time": time, "mask": mask,
                     "exist_times": exist_times})
    return datas
