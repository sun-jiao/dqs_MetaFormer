import datetime
import multiprocessing

import torch.multiprocessing
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import models
from config import get_config
from data import build_loader
from parse_args import parse_args
from simple_train import max_index_file, models_dir, model_name

# import pickle
# from collections import Counter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataloader = None


def evaluate(model):
    model = model.to(device)
    model.eval()

    nn.CrossEntropyLoss()

    correct = 0
    total = 0
    top3_correct = 0

    start = datetime.datetime.now().timestamp()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            _, top3 = torch.topk(outputs, k=3, dim=1)
            top3 = top3.squeeze().tolist()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            top3_correct += sum(labels.item() in top3_row for labels, top3_row in zip(labels, top3))

    end = datetime.datetime.now().timestamp()
    time = end - start

    accuracy = correct / total
    top3_accuracy = top3_correct / total
    print("Accuracy on the validation set: {:.2f}".format(accuracy))
    print("Top 3 accuracy on the validation set: {:.2f}".format(top3_accuracy))
    print(f"Time: {time}")


args = parse_args()
args.cfg = 'configs/MetaFG_2_384.yaml'
if args.num_workers is None:
    args.num_workers = multiprocessing.cpu_count()

config = get_config(args)

# 使用模型
model = models.build_model(config)
_, max_file = max_index_file(models_dir, model_name, 'pth')
if max_file is not None:
    model.load_state_dict(torch.load(max_file, map_location=device))

model = model.to(device)

dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

dataloader = data_loader_train

evaluate(model)
