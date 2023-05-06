import logging
from pathlib import Path
from datetime import datetime
from logger import setup_logging
from utils import read_json, write_json


class BaseConfig:
    def __init__(self, base_config, task_id=None, resume=None):
        self._config = base_config

        save_dir = Path(self._config['trainer']['save_dir'])
        task_name = self._config['name']
        if task_id is None:
            task_id = datetime.now().strftime(r'%m%d_%H%M%S')

        self.resume = resume
        self.task_id = task_id
        self._save_dir = save_dir / 'models' / task_name / task_id
        self._log_dir = save_dir / 'log' / task_name / task_id

        exist_ok = task_id == ''
        self._save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self._log_dir.mkdir(parents=True, exist_ok=exist_ok)

        setup_logging(self._log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }


    def get_logger(self, name, verbosity=2):
        assert verbosity in self.log_levels
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    def __getitem__(self, name):
        return self._config[name]

    def update_config(self, add_config):
        self._config.update(add_config)

    def write_config(self):
        write_json(self._config, self._save_dir / f'{self.task_id}_config.json')

    @classmethod
    def from_json(cls, config_file, task_id=None):
        config = read_json(Path(config_file))
        return cls(config, task_id=task_id)

    @classmethod
    def from_yaml(cls):
        # TODO 有空再写
        pass

    @property
    def get_config(self):
        return self._config

    @property
    def get_save_dir(self):
        return self._save_dir

    @property
    def get_log_dir(self):
        return self._log_dir
