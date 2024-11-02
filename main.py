from pathlib import Path
import traceback
import argparse

from experiments.exp_forecast_dh_tvt import Exp_Forecast_Dh
from experiments.exp_uni_pretrain import Exp_Uni_Pretrain
from experiments.utils_exp import check_and_create_folders


parser = argparse.ArgumentParser(
    description="Run all experiments.")

###### Args for the experiment ######
# Args for the experiment setup
parser.add_argument("--random_state", type=int, default=1, help="Random seed")
parser.add_argument("--proj_path", type=str, default=Path(__file__).parents[0])
parser.add_argument("--test_info", default="test")
parser.add_argument("--version", default="v1")
parser.add_argument("--base_model", default="flextsf",
                    choices=["flextsf", "flextsfab"])
parser.add_argument("--model_type", default="initialize",
                    choices=["initialize", "reconstruct"])
parser.add_argument("--exp_name", type=str, default="")
parser.add_argument("--ml_task", default="uni_pretrain",
                    choices=["uni_pretrain", "forecast"])
parser.add_argument("--train_setting", default="full",
                    choices=["full", "few", "zero"],)
parser.add_argument("--patch_len", type=int, default=16)
parser.add_argument("--patch_len_pretrain", type=int, default=0)
parser.add_argument("--pre_total_steps", type=int, default=400000)
parser.add_argument("--pre_random_seed", type=int, default=1)
parser.add_argument("--few_shot_config", type=float, default=1.0)
parser.add_argument("--ckpt_path", type=str, default="")
parser.add_argument("--zeroshot_epoch", type=int, default=1,
                    help="Ranging from 0 to the epochs_max of pretraining")

# Args for the training process
parser.add_argument("--dev_mode", default="debug",
                    choices=["debug", "run", "resume", "spec_gpu"])
parser.add_argument("--num_dl_workers", type=int, default=2)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--epochs_min", type=int, default=1)
parser.add_argument("--epochs_max", type=int, default=1000,
                    help="Max training epochs")
parser.add_argument("--patience", type=int, default=5,
                    help="Early stopping patience")
parser.add_argument("--weight_decay", type=float,
                    default=0.0001, help="Weight decay (regularization)")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--lr_scheduler_step", type=int, default=20,
                    help="Every how many steps to perform lr decay")
parser.add_argument("--lr_decay", type=float, default=0.5,
                    help="Multiplicative lr decay factor")
parser.add_argument("--clip_gradient", action='store_false')
parser.add_argument("--clip", type=float, default=1)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.95)
parser.add_argument("--warmup_steps", type=int, default=1000)
parser.add_argument("--log_tool", default="wandb",
                    choices=["logging", "wandb", "all"])
parser.add_argument("--pre_model", default="")
parser.add_argument("--test_info_pt", default="")

# Args for datasets
parser.add_argument("--data_group", default="all",
                    choices=["all", "regular", "irregular"])
parser.add_argument("--data_name", default="",
                    help="Dataset name. Options are in model_task_data_config. If empty, use data_group")
parser.add_argument("--var_num", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--debug_portion", type=float, default=-1)
parser.add_argument("--max_seq_length", type=int, default=1000)
parser.add_argument("--t_v_split", type=float, default=0.8)
parser.add_argument("--forecast_ratio", default=0.2, type=float)
parser.add_argument("--ltf_input_len", type=int, default=0)
parser.add_argument("--ltf_pred_len", type=int, default=0)
parser.add_argument("--ltf_features", default="M", choices=["M", "S", "MS"])
parser.add_argument("--timeenc", type=int, default=1, choices=[0, 1])
parser.add_argument('--ltf_freq', type=str, default='h',
                    help='freq for time features encoding')
parser.add_argument("--full_regular", action='store_true')
parser.add_argument("--next-start", type=float, default=1440)
parser.add_argument("--time-max", type=int, default=2880)

###### Args for the FlexTSF model ######
# Args for IVP Patcher
parser.add_argument("--ivp_solver", default="resnetflow",
                    choices=["resnetflow", "couplingflow", "gruflow", "ode"])
parser.add_argument("--flow_layers", type=int, default=2,
                    help="Number of flow layers")
parser.add_argument("--hidden_layers", type=int, default=2,
                    help="Number of hidden layers")
parser.add_argument("--dim_ivp_hidden", type=int, default=128,
                    help="Size of ivp hidden layer")
parser.add_argument("--activation", type=str, default="ELU",
                    help="Hidden layer activation")
parser.add_argument("--final_activation", type=str,
                    default="Tanh", help="Last layer activation")
parser.add_argument("--odenet", type=str, default="concat",
                    help="Type of ODE network", choices=["concat", "gru"])  # gru only in GOB
parser.add_argument("--ode_solver", type=str, default="dopri5",
                    help="ODE solver", choices=["dopri5", "rk4", "euler"])
parser.add_argument("--solver_step", type=float,
                    default=0.05, help="Fixed solver step")
parser.add_argument("--atol", type=float, default=1e-4,
                    help="Absolute tolerance")
parser.add_argument("--rtol", type=float, default=1e-3,
                    help="Relative tolerance")
parser.add_argument("--time_net", type=str, default="TimeTanh", help="Name of time net",
                    choices=["TimeFourier", "TimeFourierBounded", "TimeLinear", "TimeTanh"])
parser.add_argument("--time_hidden_dim", type=int, default=8,
                    help="Number of time features (only for Fourier)")
parser.add_argument("--patch_overlap_rate", default=0,
                    type=float, choices=[0.5, 0.25, 0])

# Args for the VAE framework
parser.add_argument("--k_iwae", type=int, default=3)
parser.add_argument("--kl_coef", type=float, default=1.0)
parser.add_argument("--prior_mu", type=float, default=0.0)
parser.add_argument("--prior_std", type=float, default=1.0)
parser.add_argument("--obsrv_std", type=float, default=0.01)
parser.add_argument("--combine_methods", default="kl_weighted",
                    choices=["average", "kl_weighted"])
parser.add_argument("--kl_alles", action='store_false')

# Args for the Attention mechanism
parser.add_argument("--nhead", type=int, default=4,
                    help="number of heads in multihead-attention")
parser.add_argument("--dropout", type=float, default=0.2,
                    help="Specifies a layer-wise dropout factor")
parser.add_argument("--embed_time", type=int, default=128,
                    help="Time embedding size")
parser.add_argument("--attn_layers", type=int, default=2)
parser.add_argument("--dim_attn_internal", type=int, default=64)
parser.add_argument("--seq_len_min", type=int, default=16)
parser.add_argument("--seq_len_max", type=int, default=512)
parser.add_argument("--dim_patch_ts", type=int, default=64)
parser.add_argument("--max_batch_size", type=int, default=64)
parser.add_argument("--max_seq_len", type=int, default=512)
parser.add_argument("--norm_eps", type=float, default=1e-5)
parser.add_argument("--freqs_theta", type=float, default=10000.0)
parser.add_argument("--multiple_of", type=int, default=256)
parser.add_argument("--dummy_type", choices=["zero", "detach", "clone"],
                    default="clone")

# Args for the Abalation Study
parser.add_argument("--patch_module", default="ivp", choices=["ivp", "none"])
parser.add_argument("--fixed_patch_len", action='store_true')
parser.add_argument("--leader_node", action='store_false')
parser.add_argument("--lyr_time_embed", action='store_false')
parser.add_argument("--dummy_patch", action='store_false')
parser.add_argument("--vt_norm", action='store_false')
parser.add_argument("--time_norm_affine", action='store_true')


if __name__ == "__main__":
    args = parser.parse_args()
    check_and_create_folders(args.proj_path)
    if args.ml_task == 'uni_pretrain':
        experiment = Exp_Uni_Pretrain(args)
    else:
        experiment = Exp_Forecast_Dh(args)

    try:
        experiment.run()
    except Exception:
        with open(experiment.proj_path/"log"/"err_{}.log".format(experiment.args.exp_name), "w") as fout:
            print(traceback.format_exc(), file=fout)

    # experiment.run()
