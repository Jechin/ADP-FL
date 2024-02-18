import sys, os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from nets.models import DenseNet, UNet
from utils.datasets import split_df, split_dataset, balance_split_dataset
from dataset.dataset import (
    ProstateDataset,
    DFDataset,
    DatasetSplit,
)

def prepare_workflow(args, logging):
    assert args.data in [
        "prostate",
        "RSNA-ICH",
    ]
    train_loaders, val_loaders, test_loaders = [], [], []
    trainsets, valsets, testsets = [], [], []
    if args.data == "prostate":
        return None
    elif args.data == "RSNA-ICH":
        N_total_client = 20
        assert args.clients <= N_total_client

        model = DenseNet(num_classes=2)
        loss_fun = nn.CrossEntropyLoss()
        train_sites = list(range(args.clients))
        val_sites = list(range(N_total_client))  # original clients
        train_data_sizes = []
        ich_folder = "binary_25k"

        train_dfs = split_df(
            args, pd.read_csv(f"./dataset/RSNA-ICH/{ich_folder}/train.csv"), N_total_client
        )
        val_dfs = split_df(
            args, pd.read_csv(f"./dataset/RSNA-ICH/{ich_folder}/validate.csv"), N_total_client
        )
        test_dfs = split_df(
            args, pd.read_csv(f"./dataset/RSNA-ICH/{ich_folder}/test.csv"), N_total_client
        )

        transform_list = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        real_trainsets = []
        for idx in range(N_total_client):
            trainset = DFDataset(
                root_dir=args.data_path,    # TODO: change the path here
                data_frame=train_dfs[idx],
                transform=transform_list,
                site_idx=idx,
            )
            valset = DFDataset(
                root_dir=args.data_path,
                data_frame=val_dfs[idx],
                transform=transform_test,
                site_idx=idx,
            )
            testset = DFDataset(
                root_dir=args.data_path,
                data_frame=test_dfs[idx],
                transform=transform_test,
                site_idx=idx,
            )
            logging.info(
                f"[Client {idx}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}"
            )
            train_data_sizes.append(len(trainset))
            real_trainsets.append(trainset)
            valsets.append(valset)
            testsets.append(testset)

        if args.merge:
            valset = torch.utils.data.ConcatDataset(valsets)
            testset = torch.utils.data.ConcatDataset(testsets)

        if args.clients < N_total_client:
            idx = np.argsort(np.array(train_data_sizes))[::-1][: args.clients]
            trainsets = [real_trainsets[i] for i in idx]
        
        # for c_idx, client_trainset in enumerate(real_trainsets):
        #     dict_users = split_dataset(client_trainset, args.virtual_clients)
        #     for v_idx in range(args.virtual_clients):
        #         virtual_trainset = DatasetSplit(
        #             client_trainset, dict_users[v_idx], c_idx, v_idx
        #         )
        #         trainsets.append(virtual_trainset)
        #         logging.info(f"[Virtual Client {c_idx}-{v_idx}] Train={len(virtual_trainset)}")
    else:
        raise NotImplementedError

    if args.debug:
        trainsets = [
            torch.utils.data.Subset(trset, list(range(args.batch * 4))) for trset in trainsets
        ]
        if args.merge:
            valset = torch.utils.data.Subset(valset, list(range(args.batch * 2)))
            testset = torch.utils.data.Subset(testset, list(range(args.batch * 2)))
        else:
            valsets = [
                torch.utils.data.Subset(trset, list(range(args.batch * 4)))
                for trset in valsets[: len(valsets)]
            ]
            testsets = [
                torch.utils.data.Subset(trset, list(range(args.batch * 4)))
                for trset in testsets[: len(testsets)]
            ]

    if args.balance:
        assert args.split == "FeatureNonIID"
        min_data_len = min([len(s) for s in trainsets])
        print(f"Balance training set, using {args.percent*100}% training data")
        for idx in range(len(trainsets)):
            trainset = torch.utils.data.Subset(
                trainsets[idx], list(range(int(min_data_len * args.percent)))
            )
            print(f"[Client {trainsets[idx]}] Train={len(trainset)}")

            train_loaders.append(
                torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True)
            )
        if args.merge:
            val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch, shuffle=False)
            test_loader = torch.utils.data.DataLoader(
                testset, batch_size=args.batch, shuffle=False
            )
        else:
            for idx in range(len(valsets)):
                valset = valsets[idx]
                testset = testsets[idx]
                val_loaders.append(
                    torch.utils.data.DataLoader(valset, batch_size=args.batch, shuffle=False)
                )
                test_loaders.append(
                    torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=False)
                )
    else:
        for idx in range(len(trainsets)):
            if args.debug:
                train_loaders.append(
                    torch.utils.data.DataLoader(
                        trainsets[idx], batch_size=args.batch, shuffle=False, drop_last=False
                    )
                )
            else:
                train_loaders.append(
                    torch.utils.data.DataLoader(
                        trainsets[idx], batch_size=args.batch, shuffle=True, drop_last=True
                    )
                )
        if args.merge:
            val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch, shuffle=False)
            test_loader = torch.utils.data.DataLoader(
                testset, batch_size=args.batch, shuffle=False
            )
        else:
            for idx in range(len(valsets)):
                valset = valsets[idx]
                val_loaders.append(
                    torch.utils.data.DataLoader(valset, batch_size=args.batch, shuffle=False)
                )
            for idx in range(len(testsets)):
                testset = testsets[idx]
                test_loaders.append(
                    torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=False)
                )
    print(f"Train loaders: {len(train_loaders)}")
    print(f"Val loaders: {len(val_loaders)}")
    print(f"Test loaders: {len(test_loaders)}")
    if args.merge:
        return (
            model,
            loss_fun,
            train_sites,
            val_sites,
            trainsets,
            valsets,
            testsets,
            train_loaders,
            val_loader,
            test_loader,
        )
    else:
        return (
            model,
            loss_fun,
            train_sites,
            val_sites,
            trainsets,
            valsets,
            testsets,
            train_loaders,
            val_loaders,
            test_loaders,
        )
