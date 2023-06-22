import os
import argparse
import torch
import random
import numpy as np

from models.model_utils import load_last_checkpoint, get_decoder
from data.crosstask import CrossTaskModule
from data.procel import ProceLModule
from data.coin import COINModule

from eval.unsupervised_step_segmentation import evaluate as unsup_eval
from eval.zeroshot_step_localization import evaluate as zeroshot_eval
from paths import PROJECT_PATH
from config import CONFIG


# Enabling reproducibility
random.seed(2021)
np.random.seed(2021)
torch.manual_seed(2021)

# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="long", help="name of the model (experiment) we are testing")
parser.add_argument("--dataset", type=str, default="CrossTask", choices=["ProceL", "CrossTask", "COIN"], help="name of the dataset")
parser.add_argument("--bg_ratio", type=float, default=0.4, help="background threshold for the global clustering in unsupervised segmentation")
parser.add_argument("--verbose", type=int, default=0, help="print everything")
parser.add_argument("--override", nargs=argparse.REMAINDER, help='override train/val keys in CONFIG')
args = parser.parse_args()
# fmt: on


def main():
    # setup from config
    output_folder = os.path.join(PROJECT_PATH, "outputs", args.name)
    CONFIG.load(os.path.join(output_folder, "config.yml"))
    CONFIG.override_with_args(args.override)

    # set optimal segmentation parameters here
    CONFIG.EVAL.FIXED_DROP_SIM = 0.35

    # prep data
    if args.dataset == "CrossTask":
        data = CrossTaskModule()
    elif args.dataset == "ProceL":
        data = ProceLModule()
    elif args.dataset == "COIN":
        data = COINModule()
    else:
        print("UNKNOWN DATASET")

    # prep model
    model = get_decoder()
    load_last_checkpoint(args.name, model, remove_name_preffix="model.")
    model.eval()

    # evaluate
    zeroshot_eval(data, model)
    unsup_eval(args, data, model)


if __name__ == "__main__":
    main()
