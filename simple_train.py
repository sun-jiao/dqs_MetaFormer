import copy
import multiprocessing
import os
# import pickle
import time

import torch
import torch.multiprocessing
import torch.nn as nn
# from collections import Counter
from PIL import ImageFile
from torch.optim import lr_scheduler

import models
from config import get_config
from data import build_loader
from optimizer import build_optimizer
from parse_args import parse_args

if torch.cuda.is_available():
    from torch.cuda.amp import autocast as autocast, GradScaler
    scaler = GradScaler()

ImageFile.LOAD_TRUNCATED_IMAGES = True
# Assuming you want to sample 10% of the dataset, the ratio should be 0.1
sampling_ratio = 1

data_dir = './dataset'
models_dir = './modelzoo'
model_name = 'metafg-2-384'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

freeze = False


def max_index_file(directory, prefix, suffix):
    max_index = -1
    max_file = None

    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith(suffix):
            # 提取索引部分
            index_str = filename[len(prefix) + 1: -len(suffix) - 1]
            try:
                index = int(index_str)
                if index > max_index:
                    max_index = index
                    max_file = os.path.join(directory, filename)
            except ValueError:
                continue

    return max_index, max_file


def train_model(_model, _dataloaders, _criterion, _optimizer, _scheduler, _num_epochs=25):
    since = time.time()

    best_acc = 0.0
    best_model_wts = copy.deepcopy(_model.state_dict())

    for epoch in range(_num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, _num_epochs - 1))

        # load best model weights
        # _model.load_state_dict(best_model_wts)

        for phase in ['train', 'val']:
            if phase == 'train':
                _model.train()  # Set model to training mode
            else:
                _model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in _dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                _optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # compute output
                    if torch.cuda.is_available():
                        with autocast():
                            outputs = _model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = _criterion(outputs, labels)
                        if phase == 'train':
                            # Scales loss. 为了梯度放大.
                            scaler.scale(loss).backward()

                            # scaler.step() 首先把梯度的值unscale回来.
                            # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
                            # 否则，忽略step调用，从而保证权重不更新（不被破坏）
                            scaler.step(_optimizer)

                            # 准备着，看是否要增大scaler
                            scaler.update()
                    else:
                        outputs = _model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = _criterion(outputs, labels)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            _optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                epoch_loss = epoch_loss / sampling_ratio
                epoch_acc = epoch_acc / sampling_ratio
                _scheduler.step(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(_model.state_dict())
                torch.save(_model.state_dict(), os.path.join(models_dir, f'{model_name}_temp.pth'))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    _model.load_state_dict(best_model_wts)
    return _model


def save_model(_model: nn.Module, _models_dir: str, name: str):
    max_index, _ = max_index_file(_models_dir, name, 'pth')
    torch.save(_model.state_dict(), os.path.join(_models_dir, f'{name}_{max_index + 1}.pth'))


if __name__ == '__main__':
    args = parse_args()
    args.cfg = 'configs/MetaFG_2_384.yaml'
    if args.num_workers is None:
        args.num_workers = multiprocessing.cpu_count()

    config = get_config(args)
    sampling_ratio = config.SAMPLING_RATIO

    torch.multiprocessing.set_sharing_strategy('file_system')

    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_datasets = {'train': dataset_train, 'val': dataset_val}

    dataloaders = {'train': data_loader_train, 'val': data_loader_val}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    # 使用模型
    model = models.build_model(config)
    _, max_file = max_index_file(models_dir, model_name, 'pth')
    if max_file is not None:
        model.load_state_dict(torch.load(max_file, map_location=device))

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # 优化器
    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer = build_optimizer(config, model)

    # 学习率调整策略
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", factor=0.1, patience=3, verbose=True, threshold=5e-3, threshold_mode="abs")

    for i in range(100):  # uncomment本行时下面两行都应该缩进，否则会连训100轮不保存。
        # 训练模型
        model = train_model(model, dataloaders, criterion, optimizer, scheduler, _num_epochs=25)
        save_model(model, models_dir, model_name)
