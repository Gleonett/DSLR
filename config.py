import yaml


class Config(object):
    def __init__(self, filepath=None):
        with open(filepath) as fin:
            config = yaml.safe_load(fin)
        for key in config.keys():
            setattr(self, key, config[key])
