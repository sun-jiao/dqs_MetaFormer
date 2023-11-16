# import re

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch import multiprocessing
from torchvision.transforms import transforms

import models
from config import get_config
from parse_args import parse_args

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
filename = 'modelzoo/metafg-2-384_6.pth'
imgs_path = "test_image/blujay.jpg"


def image_proprecess(img_path):
    img = Image.open(img_path).convert('RGB')
    data = data_transforms(img)
    data = torch.unsqueeze(data, 0)
    img_resize = img.resize((384, 384))
    return img_resize, data


args = parse_args()
args.cfg = 'configs/MetaFG_2_384.yaml'
if args.num_workers is None:
    args.num_workers = multiprocessing.cpu_count()

config = get_config(args)

data_transforms = transforms.Compose([
    transforms.Resize(384),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 使用模型
model = models.build_model(config)
model.load_state_dict(torch.load(filename, map_location=device))

model = model.to(device).eval()

img, data = image_proprecess(imgs_path)
data = data.to(device)

# for name, param in model.named_parameters():
#     if name.endswith('weight'):
#         target_layers指的是需要可视化的层，这里可视化最后一层
#         target_layers = [ ]
#         name = name.replace('.weight', '')
#         name = re.sub(r'\.([0-9]+)', r'[\1]', name)
#         code = f"target_layers.append(model.{name})"
#         exec(code)

target_layers = [
    model.stage_0[0],
    model.stage_0[1],
    model.stage_0[3],
    model.stage_0[4],
    model.stage_0[6],
    model.bn1,
    model.stage_1[0]._expand_conv,
    model.stage_1[0]._bn0,
    model.stage_1[0]._depthwise_conv,
    model.stage_1[0]._bn1,
    model.stage_1[0]._se_reduce,
    model.stage_1[0]._se_expand,
    model.stage_1[0]._project_conv,
    model.stage_1[0]._bn2,
    model.stage_1[1]._expand_conv,
    model.stage_1[1]._bn0,
    model.stage_1[1]._depthwise_conv,
    model.stage_1[1]._bn1,
    model.stage_1[1]._se_reduce,
    model.stage_1[1]._se_expand,
    model.stage_1[1]._project_conv,
    model.stage_1[1]._bn2,
    model.stage_2[0]._expand_conv,
    model.stage_2[0]._bn0,
    model.stage_2[0]._depthwise_conv,
    model.stage_2[0]._bn1,
    model.stage_2[0]._se_reduce,
    model.stage_2[0]._se_expand,
    model.stage_2[0]._project_conv,
    model.stage_2[0]._bn2,
    model.stage_2[1]._expand_conv,
    model.stage_2[1]._bn0,
    model.stage_2[1]._depthwise_conv,
    model.stage_2[1]._bn1,
    model.stage_2[1]._se_reduce,
    model.stage_2[1]._se_expand,
    model.stage_2[1]._project_conv,
    model.stage_2[1]._bn2,
    model.stage_2[2]._expand_conv,
    model.stage_2[2]._bn0,
    model.stage_2[2]._depthwise_conv,
    model.stage_2[2]._bn1,
    model.stage_2[2]._se_reduce,
    model.stage_2[2]._se_expand,
    model.stage_2[2]._project_conv,
    model.stage_2[2]._bn2,
    model.stage_2[3]._expand_conv,
    model.stage_2[3]._bn0,
    model.stage_2[3]._depthwise_conv,
    model.stage_2[3]._bn1,
    model.stage_2[3]._se_reduce,
    model.stage_2[3]._se_expand,
    model.stage_2[3]._project_conv,
    model.stage_2[3]._bn2,
    model.stage_2[4]._expand_conv,
    model.stage_2[4]._bn0,
    model.stage_2[4]._depthwise_conv,
    model.stage_2[4]._bn1,
    model.stage_2[4]._se_reduce,
    model.stage_2[4]._se_expand,
    model.stage_2[4]._project_conv,
    model.stage_2[4]._bn2,
    model.stage_2[5]._expand_conv,
    model.stage_2[5]._bn0,
    model.stage_2[5]._depthwise_conv,
    model.stage_2[5]._bn1,
    model.stage_2[5]._se_reduce,
    model.stage_2[5]._se_expand,
    model.stage_2[5]._project_conv,
    model.stage_2[5]._bn2,
    model.stage_3[0].patch_embed.proj,
    model.stage_4[0].patch_embed.proj,
]

try:
    cam = GradCAM(model=model, target_layers=target_layers)
    # 指定可视化的类别，指定为None，则按照当前预测的最大概率的类作为可视化类。
    target_category = None

    grayscale_cam = cam(
        input_tensor=data,
        targets=target_category,
    )

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(np.array(img) / 255., grayscale_cam)
    plt.imshow(visualization)
    plt.xticks()
    plt.yticks()
    plt.axis('off')
    plt.savefig(f"heat/A.g.jpg")
    print(f'successfully generated')
except:
    print(f'error occurred')
