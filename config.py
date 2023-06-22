import os
import logging
from omegaconf import OmegaConf, DictConfig

from paths import PROJECT_PATH


logger = logging.getLogger("logger")


class ConfigManager(DictConfig):
    """DictConfig with extended functionality"""

    def __init__(self):
        super().__init__({})

        default_config_path = os.path.join(PROJECT_PATH, "conf", "default.yml")
        if os.path.exists(default_config_path):
            default_config = OmegaConf.load(default_config_path)
            self.merge(default_config)
            logger.info(f"CONFIG: Initialized from default config at {default_config_path} \n")
        else:
            logger.info("CONFIG: Initialized from empty config \n")

    def setup(self, name, configdir, model_name, override_args=None):
        """Load stuff from the dataset"""

        self.NAME = name

        # load dataset config
        config_path = os.path.join(configdir, (model_name + ".yml"))
        self.load(config_path, override_args=override_args)

    def load(self, config_path, override_args=None):
        """Load from config_path into CONFIG"""

        if os.path.exists(config_path):
            loaded_config = OmegaConf.load(config_path)
            self.merge(loaded_config)
        else:
            assert False, f"CONFIG: Could not find config at {config_path}"

        # override fields in CONFIG with override_args
        if override_args is not None:
            self.override_with_args(override_args)

    def dump(self, config_path):
        "Dumps CONFIG at config_path"

        base_folder = os.path.dirname(config_path)
        if not os.path.isdir(base_folder):
            logger.info(f"DUMP CONFIG: creating directory {base_folder}")
            os.makedirs(base_folder)
        OmegaConf.save(self, config_path)

    def override_with_args(self, override_args):
        """Overrides or adds entries from override_args into CONFIG"""

        override_dict = OmegaConf.from_cli(override_args)
        self.merge(override_dict)

    def merge(self, new_config):
        """Merges new_config into CONFIG"""

        merged_config = OmegaConf.merge(self, new_config)
        [OmegaConf.update(self, k, merged_config[k]) for k in merged_config.keys()]

    def yaml_format(self):
        """Makes config content pretty - in yaml format"""

        return OmegaConf.to_yaml(self)

    def dotdict_format(self):
        """Transforms the hierarchical config dict into a flat dict with dod-separated keys"""

        def dot_merge(d, key, result):
            if isinstance(d, dict):
                for k in d:
                    new_key = key + "." + k if key else k
                    dot_merge(d[k], new_key, result)
            else:
                result[key] = d
            return result

        dotdict = dot_merge(OmegaConf.to_container(self), "", dict())
        return dotdict


try:
    CONFIG
except:
    CONFIG = ConfigManager()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parse arguments of desired setup
    parser.add_argument("--name", type=str, default="noname", help="experiment name")
    parser.add_argument("--config", type=str, default="./conf/datasets/", help="path to config file")
    parser.add_argument("--dataset", type=str, default="COIN", help="dataset type")
    parser.add_argument("--model", type=str, default="VideoTransformerEncDec", help="dataset type")
    parser.add_argument("--override", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    # args, unknownargs = parser.parse_known_args()

    # automatically setup config either from config.py or from DATASET.yml
    CONFIG.setup(args.config, args.dataset)
    # setup_config(args.config, args.dataset, override_args=args.override)
    print(OmegaConf.to_yaml(CONFIG))
    import ipdb

    ipdb.set_trace()
