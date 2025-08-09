import numpy as np


def read_satsm_forecast_tvt(dc, args, logger):
    path_satsm = args.proj_path/"data/satsm/processed"

    datasets = []
    for source in ["train", "val", "test"]:
        data = np.load(
            path_satsm/f"{source}_data.npy", allow_pickle=True)
        mask = np.load(
            path_satsm/f"{source}_mask.npy", allow_pickle=True)
        time = np.load(
            path_satsm/f"{source}_time.npy", allow_pickle=True)
        # realign time to make the first time point 0
        time = (time - time[:, 0][:, np.newaxis]) * mask[:, :, 0]
        datasets.append({"value": data, "time": time, "mask": mask})

    return datasets
