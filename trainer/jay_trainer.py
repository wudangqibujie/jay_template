import numpy as np
import pandas as pd
import torch
from utils import inf_loop
from model.metric import MetricCollect
from numpy import inf
from abc import abstractmethod
from tqdm import tqdm
import time
import logging
import logging.config as logging_config
from utils import read_json
config = read_json("logger/logger_config.json")
logging_config.dictConfig(config)
logger = logging.getLogger("train")
logger.setLevel(logging.INFO)

class BaseTrainer:
    def __init__(self,
                 model,
                 criterion,
                 metric_funcs,
                 optimzer,
                 config,
                 ):
        self.config = config
        self.model = model
        self.criterion = criterion
        self.metric_funcs = metric_funcs
        self.optimzer = optimzer

        self.trainer_config = config['trainer']
        self.epochs = self.trainer_config['epochs']
        self.save_period = self.trainer_config['save_period']
        self.log_step = self.trainer_config['log_step']

        self.start_epoch = 1
        self.checkpoint_dir = config["save_dir"]
        if config["resume"] is not None:
            self._resume_checkpoint(config["resume"])

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            rslt = self._train_epoch(epoch)
            time.sleep(1)
            logger.info(str(rslt))
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch):
        model_name = type(self.model).__name__
        model_info = {
            'model_name': model_name,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimzer': self.optimzer.state_dict(),
            'config': self.config
        }
        filename = f"{self.checkpoint_dir}/checkpoint-epoch{epoch}.pth"
        torch.save(model_info, filename)
        logger.info("save checkpoint: {}".format(filename))

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimzer.load_state_dict(checkpoint['optimzer'])
        print("resume checkpoint from: {}".format(resume_path))


class Trainer(BaseTrainer):
    def __init__(self,
                 model,
                 criterion,
                 metric_funcs,
                 optimzer,
                 config,
                 device,
                 train_data_loader,
                 valid_data_loader=None,
                 lr_scheduler=None,
                 len_epoch=None,
                 max_cum_num=10000,
                 log_step=inf
                 ):
        super().__init__(model, criterion, metric_funcs, optimzer, config)
        self.config = config
        self.device = device
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        if len_epoch is None:
            self.len_epoch = len(self.train_data_loader)
        else:
            self.train_data_loader = inf_loop(train_data_loader)
            self.len_epoch = len_epoch
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = log_step

        self.train_metric_collect = MetricCollect(self.metric_funcs, max_cum_num=max_cum_num)
        self.train_monitor_metric = MetricCollect(self.metric_funcs, max_cum_num=max_cum_num)
        self.valid_metric_collect = MetricCollect(self.metric_funcs)

    def _train_epoch(self, epoch):
        self.model.train()
        tqpm_loader = tqdm(self.train_data_loader)
        self.train_metric_collect.reset()
        self.train_monitor_metric.reset()
        total_loss = 0
        total_sample = 0
        for batch_idx, (data, target) in enumerate(tqpm_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimzer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimzer.step()
            log_loss = total_loss / total_sample if total_sample > 0 else 0
            tqpm_loader.set_description("[TRAIN] epoch: {}, batch: {} of {}, loss: {}".format(epoch, batch_idx, self.len_epoch, log_loss))

            total_sample += data.size(0)
            total_loss += loss.item() * data.size(0)

            self.train_metric_collect.update(output, target)
            self.train_monitor_metric.update(output, target)

            if batch_idx > 0 and batch_idx % self.log_step == 0:
                # print("train: epoch: {}, batch_idx: {}, rslt: {}".format(epoch, batch_idx, self.train_monitor_metric.rslt))
                self.train_monitor_metric.reset()

            if batch_idx == self.len_epoch:
                break
        log = dict()
        train_rslt = self.train_metric_collect.rslt
        train_rslt['epoch'] = epoch
        train_rslt['loss'] = total_loss / total_sample
        log["train"] = train_rslt

        if self.do_validation:
            valid_rslt = self._valid_epoch(epoch)
            log["valid"] = valid_rslt
            # print("valid: epoch: {}, rslt: {}".format(epoch, valid_rslt))

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metric_collect.reset()
        total_loss = 0
        total_sample = 0
        with torch.no_grad():
            tqpm_loader = tqdm(self.valid_data_loader)
            for batch_idx, (data, target) in enumerate(tqpm_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                total_sample += data.size(0)
                total_loss += loss.item() * data.size(0)

                tqpm_loader.set_description("[VALId] epoch: {}, batch: {} of {}, loss: {}".format(epoch, batch_idx, self.len_epoch,  loss.item()))
                self.valid_metric_collect.update(output, target)
        rslt = self.valid_metric_collect.rslt
        rslt['epoch'] = epoch
        rslt['loss'] = total_loss / total_sample
        return rslt
