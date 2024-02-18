'''
Description: Base FedAvg Trainer
Author: Jechin jechinyu@163.com
Date: 2024-02-16 16:14:16
LastEditors: Jechin jechinyu@163.com
LastEditTime: 2024-02-18 22:26:01
'''
import sys, os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import copy
import time
import random
import math
import logging
import pandas as pd
from sklearn import metrics
from utils.loss import DiceLoss
from utils.util import _eval_haus, _eval_iou
from dataset.dataset import DatasetSplit
from utils.nova_utils import SimpleFedNova4Adam

from fed.local_trainer import LocalUpdateDP

def FedWeightAvg(w, size):
    totalSize = sum(size)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w[0][k]*size[0]
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * size[i]
        # print(w_avg[k])
        w_avg[k] = torch.div(w_avg[k], totalSize)
    return w_avg

class FedTrainner(object):
    def __init__(
        self,
        args,
        logging,
        device,
        server_model,
        train_sites,
        val_sites,
        client_weights=None,
        **kwargs,
    ) -> None:
        self.args = args
        self.logging = logging
        self.device = device
        self.lr_decay = args.lr_decay > 0
        self.server_model = server_model
        self.train_sites = train_sites
        self.val_sites = val_sites
        self.client_num = len(train_sites)
        self.client_num_val = len(val_sites)
        self.sample_rate = args.sample_rate
        assert self.sample_rate > 0 and self.sample_rate <= 1
        self.aggregation_idxs = None
        self.aggregation_client_num = max(int(self.client_num * self.sample_rate), 1)
        self.client_weights = (
            [1 / self.aggregation_client_num for i in range(self.aggregation_client_num)]
            if client_weights is None
            else client_weights
        )
        self.client_models = [copy.deepcopy(server_model) for idx in range(self.client_num)]
        self.client_grads = [None for i in range(self.client_num)]
        (
            self.train_loss,
            self.train_acc,
            self.val_loss,
            self.val_acc,
            self.test_loss,
            self.test_acc,
        ) = ({}, {}, {}, {}, {}, {})

        self.generalize_sites = (
            kwargs["generalize_sites"] if "generalize_sites" in kwargs.keys() else None
        )

        self.train_loss["mean"] = []
        self.val_loss["mean"] = []
        self.test_loss["mean"] = []

        self.sigma = []
        self.used_sigma_reciprocal = 0

    def start(
        self,
        train_loaders,
        val_loaders,
        test_loaders,
        loss_fun,
        SAVE_PATH,
    ):
        # clients = [LocalUpdateDP(args=self.args, dataset=dataset_train, idxs=dict_users[i]) for i in range(args.num_users)]
        train_clients = [
            LocalUpdateDP(
                args=self.args, loader=train_loaders[idx], loss_fun=loss_fun, device=self.device, logging=self.logging, idx=idx
            ) for idx in range(len(train_loaders))
        ]
        self.logging.info("=====================Training Start=====================") # TODO 添加====
        for iter in range(self.args.rounds):
            self.logging.info("------------ Round({:^5d}/{:^5d}) ------------".format(iter, self.args.rounds))
            t_start = time.time()
            sigma = self._calculate_sigma(
                epsilon=self.args.epsilon, 
                delta=self.args.delta, 
                iter=iter, 
                rounds=self.args.rounds
            )
            w_locals, loss_locals = [], []
            if self.sample_rate < 1:
                self.aggregation_idxs = random.sample(
                    list(range(self.client_num)), self.aggregation_client_num
                )
            else:
                self.aggregation_client_num = len(self.client_models)
                self.aggregation_idxs = list(range(len(self.client_models)))
            for idx in self.aggregation_idxs:
                local = train_clients[idx]
                w, loss = local.train(model=copy.deepcopy(self.server_model).to(self.device), sigma=sigma)
                w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
            
            # update global weights
            w_glob = FedWeightAvg(w_locals, self.client_weights)
            # copy weight to net_glob
            self.server_model.load_state_dict(w_glob)

            t_end = time.time()
            self.logging.info("Round {:3d},Testing accuracy: {:.2f},Time:  {:.2f}s".format(iter, 0.0, t_end - t_start))
            self.adaptive_rounds(acc=None)
            if iter == 3:
                break

        self.logging.info("=====================Training completed=====================")

    def _calculate_sigma(self, epsilon, delta, iter, rounds):
        # \sigma^2=\frac{T-t}{\frac{\epsilon^2}{2\ln(1/\delta)}-\sum_{i=1}^{t}\frac{1}{\sigma_i^2}}
        sigma_sqr = (rounds - iter) / (epsilon**2 / (2 * np.log(1 / delta)) - self.used_sigma_reciprocal)
        self.sigma.append(sigma_sqr**0.5)
        self.used_sigma_reciprocal += 1 / sigma_sqr
        return sigma_sqr**0.5
    
    def adaptive_rounds(self, acc=None):
        if acc is None:
            return False
        else:
            # TODO adaptive rounds
            raise NotImplementedError
    