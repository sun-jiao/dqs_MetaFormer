import argparse

import torch

import models
from config import get_config
from data import build_loader
from parse_args import parse_args

if __name__ == '__main__':
    args = parse_args()
    args.cfg = 'configs/MetaFG_2_384.yaml'
    config = get_config(args)

    model = models.build_model(config)
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load('modelzoo/metafg-2-384_0.pth', map_location=device))
    old_weights = model.head.weight
    new_head = torch.nn.Linear(model.head.in_features, 13500)

    for i in range(11238):
        for j in range(1024):
            new_head.weight[i][j] = old_weights[i][j].clone().detach()

    model.head = new_head

    torch.save(model.state_dict(), 'metafg_533.pth')

