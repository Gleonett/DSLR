"""
Class for loading and storing configurations
"""

import yaml
import numpy as np


class Config(object):

    def __init__(self, filepath: str):

        with open(filepath) as f:
            config = yaml.safe_load(f)

        for key in config.keys():
            setattr(self, key, config[key])

    def choosed_features(self) -> np.ndarray:
        features = np.array(list(self.features.keys()))
        mask = np.array(list(self.features.values()))
        return features[mask]
