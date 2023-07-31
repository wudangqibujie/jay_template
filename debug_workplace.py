import os
import logging
import logging.config
import yaml
import glob
import json


class FeatureInfo:
    def __init__(self):
        pass


def load_model_config(config_dir, experiment_id):
    model_configs = glob.glob(os.path.join(config_dir, "model_config.yaml"))
    if not model_configs:
        model_configs = glob.glob(os.path.join(config_dir, "model_config/*.yaml"))
    if not model_configs:
        raise RuntimeError('config_dir={} is not valid!'.format(config_dir))
    found_params = dict()
    for config in model_configs:
        with open(config, 'r') as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            if 'Base' in config_dict:
                found_params['Base'] = config_dict['Base']
            if experiment_id in config_dict:
                found_params[experiment_id] = config_dict[experiment_id]
        if len(found_params) == 2:
            break
    # Update base and exp_id settings consectively to allow overwritting when conflicts exist
    params = found_params.get('Base', {})
    params.update(found_params.get(experiment_id, {}))
    assert "dataset_id" in params, f'expid={experiment_id} is not valid in config.'
    params["model_id"] = experiment_id
    return params


def load_dataset_config(config_dir, dataset_id):
    params = {"dataset_id": dataset_id}
    dataset_configs = glob.glob(os.path.join(config_dir, "dataset_config.yaml"))
    if not dataset_configs:
        dataset_configs = glob.glob(os.path.join(config_dir, "dataset_config/*.yaml"))
    for config in dataset_configs:
        with open(config, "r") as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            if dataset_id in config_dict:
                params.update(config_dict[dataset_id])
                return params
    raise RuntimeError(f'dataset_id={dataset_id} is not found in config.')


def load_config(config_dir, experiment_id):
    params = load_model_config(config_dir, experiment_id)
    data_params = load_dataset_config(config_dir, params['dataset_id'])
    params.update(data_params)
    return params


config_dir = 'tmp/example3_config'
experiment_id = 'DeepFM_test_csv'
params = load_config(config_dir, experiment_id)

print(params)