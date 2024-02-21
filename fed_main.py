"""
Federated training main logic
"""
import sys, os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import time
import copy
import random
import math
import logging
import pandas as pd
import pickle as pkl
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import monai.transforms as monai_transforms

from nets.models import (
    DenseNet,
    UNet,
)

from fed.global_trainer import FedTrainner
from utils.util import setup_logger, get_timestamp, setup_parser
from utils.datasets import split_df, split_dataset, balance_split_dataset
from utils.workflow import prepare_workflow



if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    if args.save_path != "":
        os.path.join(args.save_path, "./experiments/checkpoint/{}/seed{}".format(
            args.data, args.seed
        ))
    else: 
        args.save_path = "./experiments/checkpoint/{}/seed{}".format(
            args.data, args.seed
        )
    exp_folder = "{}_rounds{}_lr{}_batch{}_N{}_eps{}_delta{}".format(
        args.mode,
        args.rounds,
        args.lr,
        args.batch,
        args.clients,
        args.epsilon,
        args.delta
    )
    if args.debug:
        exp_folder = exp_folder + "_debug"
    if args.test:
        exp_folder = exp_folder + "_test"
    if args.adp_noise:
        exp_folder = exp_folder + "_adpnoise"

    args.save_path = os.path.join(args.save_path, exp_folder)
    if not args.test:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
    SAVE_PATH = args.save_path

    # Set up logging
    args.log_path = args.save_path.replace("/checkpoint/", "/log/")
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    lg = setup_logger(
        f"{args.mode}-{get_timestamp()}",
        args.log_path,
        level=logging.INFO,
        screen=False,
        tofile=True,
    )
    lg = logging.getLogger(f"{args.mode}-{get_timestamp()}")
    
    lg.info(args)

    generalize_sites = None
    (
        server_model,
        loss_fun,
        train_sites,
        val_sites,
        train_sets,
        val_sets,
        test_sets,
        train_loaders,
        val_loaders,
        test_loaders,
    ) = prepare_workflow(args, lg)

    assert (
        int(args.clients) == len(train_loaders) == len(train_sites)
    ), f"Client num {args.clients}, train loader num {len(train_loaders)},\
         train site num {len(train_sites)} do not match."
    assert len(val_loaders) == len(val_sites)  # == int(args.clients)
    train_total_len = sum([len(tr_set) for tr_set in train_sets])
    client_weights = (
        [len(tr_set) / train_total_len for tr_set in train_sets]
        if args.weighted_avg
        else [
            1.0 / float(int(args.clients * args.sample_rate) * args.virtual_clients)
            for i in range(int(args.clients * args.virtual_clients))
        ]
    )
    lg.info("Client Weights: " + str(client_weights))

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.cuda.manual_seed(123)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    lg.info(f"Device: {device}")

    
    from torch.utils.tensorboard import SummaryWriter

    args.writer = SummaryWriter(args.log_path)

    # setup trainer
    trainer_dict = {
        "fedavg": FedTrainner,
        "dpsgd": FedTrainner,
        "no_dp": FedTrainner,
    }
    TrainerClass = trainer_dict[args.mode]

    original_server_model  = copy.deepcopy(server_model)

    trainer = TrainerClass(
        args,
        lg,
        device,
        server_model,
        train_sites,
        val_sites,
        client_weights=client_weights,
        generalize_sites=generalize_sites,
    )

    trainer.best_changed = False
    trainer.early_stop = 20

    trainer.client_steps = [torch.tensor(len(train_loader)) for train_loader in train_loaders]
    print("Client steps:", trainer.client_steps)

    if args.resume:
        checkpoint = torch.load(SAVE_PATH)
        trainer.server_model.load_state_dict(checkpoint["server_model"])
        if args.local_bn:
            for client_idx in range(trainer.client_num):
                trainer.client_models[client_idx].load_state_dict(
                    checkpoint["model_{}".format(client_idx)]
                )
        else:
            for client_idx in range(trainer.client_num):
                trainer.client_models[client_idx].load_state_dict(checkpoint["server_model"])
        trainer.best_epoch, trainer.best_acc = checkpoint["best_epoch"], checkpoint["best_acc"]
        trainer.start_iter = int(checkpoint["a_iter"]) + 1

        print("Resume training from epoch {}".format(trainer.start_iter))
    else:
        # log the best for each model on all datasets
        trainer.best_epoch = 0
        trainer.best_acc = 0.0
        trainer.start_iter = 0

    if args.test:
        trainer.inference(args.ckpt, test_loaders, loss_fun, val_sites, process=True)
    else:
        try:
            trainer.start(
                train_loaders, val_loaders, test_loaders, loss_fun, SAVE_PATH
            )
        except NotImplementedError:
            print("private finish")


    logging.shutdown()