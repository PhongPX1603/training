import yaml
import torch
from importlib import import_module


def load_yaml(yaml_file):
    with open(yaml_file) as f:
        settings = yaml.safe_load(f)
    return settings


def eval_config(config):

    def _eval_config(config):
        if isinstance(config, dict):
            for key, value in config.items():
                if key not in ['module', 'class']:
                    config[key] = _eval_config(value)

            if 'module' in config and 'class' in config:
                module = config['module']
                class_ = config['class']
                config_kwargs = config.get(class_, {})
                return getattr(import_module(module), class_)(**config_kwargs)

            return config
        elif isinstance(config, list):
            return [_eval_config(ele) for ele in config]
        elif isinstance(config, str):
            return eval(config, __extralibs__)
        else:
            return config

    __extralibs__ = {name: import_module(lib) for (name, lib) in config.pop('extralibs', {}).items()}
    __extralibs__['config'] = config

    return _eval_config(config)


def prepare_device(n_gpu_use: int = 0) -> Tuple[str, List[int]]:
    n_gpu = torch.cuda.device_count()  # get all GPU indices of system.

    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0

    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are available on this machine.")
        n_gpu_use = n_gpu

    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    gpu_indices = list(range(n_gpu_use))

    return device, gpu_indices
