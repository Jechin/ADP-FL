import sys, os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
import monai.transforms as monai_transforms
from nets.models import DenseNet, UNet
from utils.datasets import split_df, split_dataset, balance_split_dataset
from utils.loss import DiceLoss
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
        args.batch = 8
        args.lr = 0.001
        args.C = 0.0002 if args.clip == 0 else args.clip
        assert args.clients <= 6
        model = UNet(out_channels=1)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        loss_fun = DiceLoss()
        # sites = ['BIDMC', 'HK',  'ISBI', 'ISBI_1.5', 'UCL']
        sites = [1, 2, 3, 4, 5, 6]
        train_sites = list(range(args.clients))
        val_sites = [1, 2, 3, 4, 5, 6]
        keys = ["Image", "Mask"]
        data_splits = [0.6, 0.2, 0.2]
        train_data_sizes = []
        transform_list = [
            monai_transforms.Resized(keys, [256, 256]),
            monai_transforms.ToTensord(keys),
        ]

        transform = monai_transforms.Compose(transform_list)

        real_trainsets = []
        if args.generalize:
            if int(args.leave) in sites:
                leave_idx = sites.index(int(args.leave))
                sites.pop(leave_idx)
                generalize_sites = [int(args.leave)]
                logging.info("Source sites:" + str(sites))
                logging.info("Unseen sites:" + str(generalize_sites))
            else:
                raise ValueError(f"Unkown leave dataset{args.leave}")

            for site in sites:
                trainset = ProstateDataset(
                    transform=transform, site=site, split=0, splits=data_splits, seed=args.seed, path=args.data_path
                )
                valset = ProstateDataset(
                    transform=transform, site=site, split=1, splits=data_splits, seed=args.seed, path=args.data_path
                )
                logging.info(f"[Client {site}] Train={len(trainset)}, Val={len(valset)}")
                trainsets.append(trainset)
                valsets.append(valset)
            for site in generalize_sites:
                trainset = ProstateDataset(
                    transform=transform, site=site, split=0, splits=data_splits, seed=args.seed, path=args.data_path
                )
                valset = ProstateDataset(
                    transform=transform, site=site, split=1, splits=data_splits, seed=args.seed, path=args.data_path
                )
                testset = ProstateDataset(
                    transform=transform, site=site, split=2, splits=data_splits, seed=args.seed, path=args.data_path
                )
                wholeset = torch.utils.data.ConcatDataset([trainset, valset, testset])
                logging.info(f"[Unseen Client {site}] Test={len(wholeset)}")
                testsets.append(wholeset)
        else:
            for site in sites:
                if site == args.free:
                    trainset = ProstateDataset(
                        transform=transform,
                        site=site,
                        split=0,
                        splits=data_splits,
                        seed=args.seed,
                        freerider=True,
                    )
                    valset = ProstateDataset(
                        transform=transform, site=site, split=1, splits=data_splits, seed=args.seed, path=args.data_path
                    )
                    testset = ProstateDataset(
                        transform=transform, site=site, split=2, splits=data_splits, seed=args.seed, path=args.data_path
                    )
                    logging.info(
                        f"[Free Rider Client {site}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}"
                    )
                elif site == args.noisy:
                    trainset = ProstateDataset(
                        transform=transform,
                        site=site,
                        split=0,
                        splits=data_splits,
                        seed=args.seed,
                        randrot=transforms.RandomRotation(degrees=(1, 179)),
                    )
                    valset = ProstateDataset(
                        transform=transform, site=site, split=1, splits=data_splits, seed=args.seed, path=args.data_path
                    )
                    testset = ProstateDataset(
                        transform=transform, site=site, split=2, splits=data_splits, seed=args.seed, path=args.data_path
                    )
                    logging.info(
                        f"[Noisy Client {site}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}"
                    )
                else:
                    trainset = ProstateDataset(
                        transform=transform, site=site, split=0, splits=data_splits, seed=args.seed, path=args.data_path
                    )
                    valset = ProstateDataset(
                        transform=transform, site=site, split=1, splits=data_splits, seed=args.seed, path=args.data_path
                    )
                    testset = ProstateDataset(
                        transform=transform, site=site, split=2, splits=data_splits, seed=args.seed, path=args.data_path
                    )

                    logging.info(
                        f"[Client {site}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}"
                    )
                train_data_sizes.append(len(trainset))
                real_trainsets.append(trainset)
                valsets.append(valset)
                testsets.append(testset)

            if args.merge:
                valset = torch.utils.data.ConcatDataset(valsets)
                testset = torch.utils.data.ConcatDataset(testsets)

            if not "no" in args.leave:
                if int(args.leave) in sites:
                    leave_idx = sites.index(int(args.leave))
                    sites.pop(leave_idx)
                    trainsets.pop(leave_idx)
                    logging.info("New sites:" + str(sites))
                    generalize_sites = [int(args.leave)]
                else:
                    raise ValueError(f"Unkown leave dataset{args.leave}")

            if args.clients < 6:
                idx = np.argsort(np.array(train_data_sizes))[::-1][: args.clients]
                trainsets = [real_trainsets[i] for i in idx]
                sites = [sites[i] for i in idx]
            else:
                trainsets = real_trainsets

    elif args.data == "RSNA-ICH":
        args.batch = 16
        args.lr = 0.0003
        args.C = 0.002 if args.clip == 0 else args.clip
        N_total_client = 1 if args.center else 20
        if args.center: 
            args.clients = 1
        assert args.clients <= N_total_client
        model = DenseNet(num_classes=2)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
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
        root_dir = "./dataset/RSNA-ICH/research/dept8/qdou/data/RSNA-ICH/organized/stage_2_train" \
            if args.data_path is "" else args.data_path
        for idx in range(N_total_client):
            trainset = DFDataset(
                root_dir=root_dir,
                data_frame=train_dfs[idx],
                transform=transform_list,
                site_idx=idx,
            )
            valset = DFDataset(
                root_dir=root_dir,
                data_frame=val_dfs[idx],
                transform=transform_test,
                site_idx=idx,
            )
            testset = DFDataset(
                root_dir=root_dir,
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
        else:
            trainsets = real_trainsets
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
