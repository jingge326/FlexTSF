import torch
from torch.utils.data import DataLoader, Subset
from experiments.data_harw4imu import read_harw4imu_forecast_tvt
from experiments.data_ltf import Dataset_Forecast_Regular_Continuous, Dataset_Forecast_Regular_Continuous, read_etth_forecast_tvt, read_ettm_forecast_tvt, read_ltfcustom_forecast_tvt
from experiments.data_ehr import read_ehr_forecast_tvt
from experiments.data_monash import DatasetGP_Monash, collate_fn_forecast_monash, read_monash_pretrain_data
from experiments.data_satsm import read_satsm_forecast_tvt
from experiments.data_traffic import read_traffic_forecast_tvt
from experiments.data_ucruea import Dataset_Forecast_Irregular_Seperated, collate_fn_forecast_general, read_uea_forecast_tvt
from models.flextsf import FlexTSF_General_Forecast


def get_general_tvt(args, logger):

    dataset_class = dataset_info[args.data_name]["modules"][args.ml_task]["dataclass"]

    read_data_fn = dataset_info[args.data_name]["modules"][args.ml_task]["data_reader"]
    dc = dataset_info[args.data_name]["configs"]

    if args.ltf_features == "S" and args.data_name in ["etth2", "ettm2", "exrate", "illness", "weather"]:
        dc["var_num"] = 1

    data_train, data_val, data_test = read_data_fn(dc, args, logger)

    dataset_train = dataset_class(data_train, data_util={}, args=args)
    dataset_val = dataset_class(
        data_val, data_util=dataset_train.data_util, args=args)
    dataset_test = dataset_class(
        data_test, data_util=dataset_train.data_util, args=args)

    others = {"scaler": dataset_train.data_util.get("scaler", None)}

    # Make a subset of the training data
    if args.few_shot_config < 1:
        len_train = len(dataset_train)
        subset_indices = torch.randperm(
            len_train)[:int(len_train * args.few_shot_config)]
        dataset_train = Subset(dataset_train, subset_indices)
    elif args.few_shot_config > 1:
        len_train = len(dataset_train)
        subset_indices = torch.randperm(
            len_train)[:int(args.few_shot_config)]
        dataset_train = Subset(dataset_train, subset_indices)

    try:
        collate_fn = dict_models[args.base_model][args.ml_task]["collate_fn"]
    except KeyError:
        raise NotImplementedError(
            f'No collate_fn available for {args.base_model} and {args.ml_task}')

    dl_train = DataLoader(
        dataset=dataset_train,
        collate_fn=lambda batch: collate_fn(
            batch, args, dc),
        shuffle=True,
        batch_size=args.batch_size)
    dl_val = DataLoader(
        dataset=dataset_val,
        collate_fn=lambda batch: collate_fn(
            batch, args, dc),
        shuffle=False,
        batch_size=args.batch_size)
    dl_test = DataLoader(
        dataset=dataset_test,
        collate_fn=lambda batch: collate_fn(
            batch, args, dc),
        shuffle=False,
        batch_size=args.batch_size)

    return dl_train, dl_val, dl_test, others


dict_models = {
    "flextsf": {
        "forecast": {
            "class_obj": FlexTSF_General_Forecast,
            "collate_fn": collate_fn_forecast_general,
        },
        "uni_pretrain": {
            "class_obj": FlexTSF_General_Forecast,
            "collate_fn": collate_fn_forecast_monash,
        },
    },
}


# Time unit in seconds
# minutely: 60
# half hourly: 1800
# hourly: 3600
# daily: 86400
# weekly: 604800
# monthly: 2592000
# quarterly: 7776000
# yearly: 31536000

dataset_info = {
    "monash": {
        "modules": {
            "uni_pretrain": {
                "dataclass": DatasetGP_Monash,
                "data_reader": read_monash_pretrain_data},
        }
    },
    "etth1": {
        "modules": {
            "forecast": {
                "dataclass": Dataset_Forecast_Regular_Continuous,
                "data_reader": read_etth_forecast_tvt},
        },
        "configs": {
            "name": "etth1",
            "var_num": 7,
            "data_type": "ltf",
            "frequency": 0.0002778,
            "time_unit": 3600,
            "regularity": "regular",
        },
    },
    "etth2": {
        "modules": {
            "forecast": {
                "dataclass": Dataset_Forecast_Regular_Continuous,
                "data_reader": read_etth_forecast_tvt},
        },
        "configs": {
            "name": "etth2",
            "var_num": 7,
            "data_type": "ltf",
            "frequency": 0.0002778,
            "time_unit": 3600,
            "regularity": "regular",
            "patch_len": 4,
        },
    },
    "ettm1": {
        "modules": {
            "forecast": {
                "dataclass": Dataset_Forecast_Regular_Continuous,
                "data_reader": read_ettm_forecast_tvt},
        },
        "configs": {
            "name": "ettm1",
            "var_num": 7,
            "data_type": "ltf",
            "frequency": 0.001111,
            "time_unit": 900,
            "regularity": "regular",
        },
    },
    "ettm2": {
        "modules": {
            "forecast": {
                "dataclass": Dataset_Forecast_Regular_Continuous,
                "data_reader": read_ettm_forecast_tvt},
        },
        "configs": {
            "name": "ettm2",
            "var_num": 7,
            "data_type": "ltf",
            "frequency": 0.001111,
            "time_unit": 900,
            "regularity": "regular",
        },
    },
    "exrate": {
        "modules": {
            "forecast": {
                "dataclass": Dataset_Forecast_Regular_Continuous,
                "data_reader": read_ltfcustom_forecast_tvt},
        },
        "configs": {
            "name": "exrate",
            "var_num": 8,
            "data_type": "ltf",
            "frequency": 1.15741e-5,
            "time_unit": 86400,
            "regularity": "regular",
        },
    },
    "illness": {
        "modules": {
            "forecast": {
                "dataclass": Dataset_Forecast_Regular_Continuous,
                "data_reader": read_ltfcustom_forecast_tvt},
        },
        "configs": {
            "name": "illness",
            "var_num": 7,
            "data_type": "ltf",
            "frequency": 1.653439e-6,
            "time_unit": 604800,
            "regularity": "regular",
        },
    },
    "weather": {
        "modules": {
            "forecast": {
                "dataclass": Dataset_Forecast_Regular_Continuous,
                "data_reader": read_ltfcustom_forecast_tvt},
        },
        "configs": {
            "name": "weather",
            "var_num": 21,
            "data_type": "ltf",
            "frequency": 0.001667,
            "time_unit": 600,
            "regularity": "regular",
        },
    },
    "electricity": {
        "modules": {
            "forecast": {
                "dataclass": Dataset_Forecast_Regular_Continuous,
                "data_reader": read_ltfcustom_forecast_tvt},
        },
        "configs": {
            "name": "electricity",
            "var_num": 321,
            "data_type": "ltf",
            "frequency": 0.0002778,
            "time_unit": 3600,
            "regularity": "regular",
        },
        "hyperparameters": {
            "flextsf01": {
                "batch_size": 8,
            },
            "flextsf": {
                "batch_size": 8,
            },
        },
    },
    "traffic": {
        "modules": {
            "forecast": {
                "dataclass": Dataset_Forecast_Regular_Continuous,
                "data_reader": read_ltfcustom_forecast_tvt},
        },
        "configs": {
            "name": "traffic",
            "var_num": 862,
            "data_type": "ltf",
            "frequency": 0.0002778,
            "time_unit": 3600,
            "regularity": "regular",
        },
        "hyperparameters": {
            "flextsf01": {
                "batch_size": 4,
            },
            "flextsf": {
                "batch_size": 4,
            },
        },
    },
    "metrla": {
        "modules": {
            "forecast": {
                "dataclass": Dataset_Forecast_Irregular_Seperated,
                "data_reader": read_traffic_forecast_tvt, },
        },
        "configs": {
            "name": "metrla",
            "var_num": 207,
            "data_type": "traffic",
            "frequency": 1.1574e-5,
            "time_unit": 86400,
            "regularity": "irregular",
            "irreg_seq_len_max": 24,
        },
        "hyperparameters": {
            "contiformer": {
                "batch_size": 32,
            },
            "flextsf": {
                "batch_size": 32,
            },
        },
    },
    "SpokenArabicDigits": {
        "modules": {
            "forecast": {
                "dataclass": Dataset_Forecast_Irregular_Seperated,
                "data_reader": read_uea_forecast_tvt},
        },
        "configs": {
            "name": "SpokenArabicDigits",
            "var_num": 13,
            "data_type": "uea",
            "frequency": 11025,
            "time_unit": 9.0703e-5,
            "intro": "https://archive.ics.uci.edu/dataset/195/spoken+arabic+digit",
            "regularity": "irregular",
            "irreg_seq_len_max": 93,
            "missing_drop_rate": 0.1,
        },
    },
    "CharacterTrajectories": {
        "modules": {
            "forecast": {
                "dataclass": Dataset_Forecast_Irregular_Seperated,
                "data_reader": read_uea_forecast_tvt},
        },
        "configs": {
            "name": "CharacterTrajectories",
            "var_num": 3,
            "data_type": "uea",
            "frequency": 200,
            "time_unit": 0.005,
            "intro": "https://archive.ics.uci.edu/dataset/175/character+trajectories",
            "regularity": "irregular",
            "irreg_seq_len_max": 182,
            "missing_drop_rate": 0.1,
        },
    },
    "HARw4IMU": {
        "modules": {
            "forecast": {
                "dataclass": Dataset_Forecast_Irregular_Seperated,
                "data_reader": read_harw4imu_forecast_tvt},
        },
        "configs": {
            "name": "HARw4IMU",
            "var_num": 12,
            "data_type": "harw4imu",
            "time_unit": 0.1,
            "intro": "https://archive.ics.uci.edu/dataset/196/localization+data+for+person+activity",
            "irreg_seq_len_max": 50,
            "regularity": "irregular",
        },
    },
    "PhysioNet2012": {
        "modules": {
            "forecast": {
                "dataclass": Dataset_Forecast_Irregular_Seperated,
                "data_reader": read_ehr_forecast_tvt},
        },
        "configs": {
            "name": "PhysioNet2012",
            "var_num": 37,
            "data_type": "ehr",
            "time_unit": 60,
            "intro": "https://physionet.org/content/challenge-2012/1.0.0/",
            "irreg_seq_len_max": 216,
            "regularity": "irregular",
        },
    },
    "eICU": {
        "modules": {
            "forecast": {
                "dataclass": Dataset_Forecast_Irregular_Seperated,
                "data_reader": read_ehr_forecast_tvt},
        },
        "configs": {
            "name": "eICU",
            "var_num": 14,
            "data_type": "ehr",
            "time_unit": 60,
            "intro": "https://physionet.org/content/eicu-crd/2.0/",
            "irreg_seq_len_max": 446,
            "regularity": "irregular",
        },
    },
    "MIMIC4": {
        "modules": {
            "forecast": {
                "dataclass": Dataset_Forecast_Irregular_Seperated,
                "data_reader": read_ehr_forecast_tvt},
        },
        "configs": {
            "name": "MIMIC4",
            "var_num": 71,
            "data_type": "ehr",
            "time_unit": 60,
            "intro": "https://physionet.org/content/mimiciv/2.2/",
            "irreg_seq_len_max": 786,
            "regularity": "irregular",
        },
        "hyperparameters": {
            "flextsf01": {
                "batch_size": 16,
            },
            "flextsf": {
                "batch_size": 16,
            },
            "contiformer": {
                "batch_size": 16,
            },
        },
    },
    "satsm": {
        "modules": {
            "forecast": {
                "dataclass": Dataset_Forecast_Irregular_Seperated,
                "data_reader": read_satsm_forecast_tvt},
        },
        "configs": {
            "name": "satsm",
            "var_num": 13,
            "data_type": "satellite",
            "time_unit": 86400,
            "regularity": "irregular",
            "irreg_seq_len_max": 96,
        },
    },
}
