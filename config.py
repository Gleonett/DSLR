"""
Class for loading and storing configurations
"""

import yaml


class Config(object):

    def __init__(self, filepath: str):

        with open(filepath) as f:
            config = yaml.safe_load(f)

        for key in config.keys():
            setattr(self, key, config[key])
