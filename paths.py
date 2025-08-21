import sys
import os
from os.path import join

if os.path.isdir("/user/n.dvornik/"):
    # office machine of n.dvornik
    PROJECT_PATH = "/user/n.dvornik/Git/unsup-step-pred/"
    S3D_PATH = "/user/n.dvornik/Git/S3D_HowTo100M/"
    COIN_PATH = "/user/n.dvornik/Datasets/COIN/"
    CT_PATH = "/user/n.dvornik/Datasets/crosstask/"
    YC_PATH = "/user/n.dvornik/Datasets/YouCook2/"
    HOWTO100M = "/user/n.dvornik/Datasets/HowTo100M/"
    PROCEL_PATH = "/user/n.dvornik/Datasets/ProceL/"
    ANNOT_PATH = os.path.join(PROJECT_PATH, "annotations")
else:
    # AWS
    PROJECT_PATH = "/home/ubuntu/unsup-step-pred-vid-seg/"
    CT_PATH = "/data2/crosstask_release/"
    PROCEL_PATH = ""
    BREAKFAST_PATH = "/data/Breakfast/"
    PROCEL_PATH = "/data/ProceL_dataset/"
    COIN_PATH = None
    YC_PATH = None
    HOWTO100M = "/data/HowTo100M/"

DATASET_PATHS_DICT = {"CrossTask": CT_PATH, "COIN": COIN_PATH, "YouCook2": YC_PATH, "HowTo100M": HOWTO100M}
