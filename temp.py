import argparse

import torch

import models
from config import get_config
from data import build_loader
from parse_args import parse_args

if __name__ == '__main__':
    args = parse_args()
    args.cfg = 'configs/MetaFG_meta_2_inat21_384.yaml'
    config = get_config(args)

    model = models.build_model(config)
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load('modelzoo/metafg_2_inat21_384.pth', map_location=device))
    num_ftrs = model.head.in_features
    model.head = torch.nn.Linear(num_ftrs, 533)
    torch.save(model.state_dict(), 'metafg_533.pth')

