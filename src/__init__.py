from collections import namedtuple
import json


def _dict2obj(d):
    return namedtuple("Config", d.keys())(*d.values())


def load_config(path):
    with open(path) as config_file:
        return json.loads(config_file.read(), object_hook=_dict2obj)