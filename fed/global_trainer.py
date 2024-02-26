'''
Description: Base FedAvg Trainer
Author: Jechin jechinyu@163.com
Date: 2024-02-16 16:14:16
LastEditors: Jechin jechinyu@163.com
LastEditTime: 2024-02-22 14:34:25
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
from utils.util import _eval_haus, _eval_iou, dict_append, metric_log_print
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
                args=self.args, 
                train_loader=train_loaders[idx], 
                val_loader=val_loaders[idx],
                test_loader=test_loaders[idx],
                loss_fun=loss_fun, 
                device=self.device, 
                logging=self.logging, 
                idx=idx
            ) for idx in range(len(train_loaders))
        ]
        self.logging.info("=====================FL Start=====================") 
        for iter in range(self.args.rounds):
            self.logging.info("------------ Round({:^5d}/{:^5d}) Train ------------".format(iter, self.args.rounds))
            t_start = time.time()
            sigma = self._calculate_sigma(
                epsilon=self.args.epsilon, 
                delta=self.args.delta, 
                iter=iter, 
                rounds=self.args.rounds
            )
            self.logging.info(f"sigma: {sigma}")
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

            # validation
            self.logging.info("------------ Validation ------------")
            with torch.no_grad():
                assert len(self.val_sites) == len(val_loaders)
                for client_idx in self.aggregation_idxs:
                    local = train_clients[client_idx]
                    val_loss, val_acc = local.test(model=self.server_model.to(self.device), mode="val")
                    self.val_loss = dict_append(
                        f"client_{self.val_sites[client_idx]}", val_loss, self.val_loss
                    )
                    self.args.writer.add_scalar(
                        f"Loss/val_{self.val_sites[client_idx]}", val_loss, iter
                    )
                    if isinstance(val_acc, dict):
                        out_str = ""
                        for k, v in val_acc.items():
                            out_str += " | Val {}: {:.4f}".format(k, v)
                            self.val_acc = dict_append(
                                f"client{self.val_sites[client_idx]}_" + k, v, self.val_acc
                            )
                            self.args.writer.add_scalar(
                                f"Performance/val_client{self.val_sites[client_idx]}_{k}", v, iter,
                            )

                        self.logging.info(
                            " Site-{:<10s}| Val Loss: {:.4f}{}".format(
                                str(self.val_sites[client_idx]), val_loss, out_str
                            )
                        )
                    else:
                        self.val_acc = dict_append(
                            f"client_{self.val_sites[client_idx]}",
                            round(val_acc, 4),
                            self.val_acc,
                        )
                        self.logging.info(
                            " Site-{:<10s}| Val Loss: {:.4f} | Val Acc: {:.4f}".format(
                                str(self.val_sites[client_idx]), val_loss, val_acc
                            )
                        )
                        self.args.writer.add_scalar(
                            f"Accuracy/val_{self.val_sites[client_idx]}", val_acc, iter
                        )

                clients_loss_avg = np.mean(
                    [v[-1] for k, v in self.val_loss.items() if "mean" not in k]
                )
                self.val_loss["mean"].append(clients_loss_avg)
                self.val_acc, out_str = metric_log_print(self.val_acc, val_acc)

                self.args.writer.add_scalar(f"Loss/val", clients_loss_avg, iter)

                mean_val_acc_ = (
                    self.val_acc["mean_Acc"][-1]
                    if "mean_Acc" in list(self.val_acc.keys())
                    else self.val_acc["mean_Dice"][-1]
                )
                self.logging.info(
                    " Site-Average | Val Loss: {:.4f}{}".format(
                        clients_loss_avg, out_str
                    )
                )

                if mean_val_acc_ > self.best_acc:
                    self.best_acc = mean_val_acc_
                    self.best_epoch = iter
                    self.best_changed = True
                    self.logging.info(
                        " Best Epoch:{} | Avg Val Acc: {:.4f}".format(
                            self.best_epoch, np.mean(mean_val_acc_)
                        )
                    )

            t_end = time.time()
            self.logging.info("Round {:3d}, Time:  {:.2f}s".format(iter, t_end - t_start))
            if self.args.adp_round and iter > 1:
                self.adaptive_rounds(iter)
            if iter == 3 and self.args.debug:
                break

        self.logging.info("=====================FL completed=====================")

    def _calculate_sigma(self, epsilon, delta, iter, rounds):
        # \sigma^2=\frac{T-t}{\frac{\epsilon^2}{2\ln(1/\delta)}-\sum_{i=1}^{t}\frac{1}{\sigma_i^2}}
        sigma_sqr = (rounds - iter) / (epsilon**2 / (2 * np.log(1 / delta)) - self.used_sigma_reciprocal)
        self.sigma.append(sigma_sqr**0.5)
        self.used_sigma_reciprocal += 1 / sigma_sqr
        if iter == 1:
            self.logging.info(f"sigma_0: {self.sigma[0]} , sigma_1: {sigma_sqr**0.5}")
        return sigma_sqr**0.5
    
    def adaptive_rounds(self, iter):
        # TODO adaptive rounds
        factor = 0.99
        threshold = 0.0001
        if self.val_loss["mean"][iter-1] - self.val_loss["mean"][iter] < threshold:
            self.args.rounds = iter + int((self.args.rounds - iter) * factor)
    