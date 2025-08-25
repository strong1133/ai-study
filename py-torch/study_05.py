## 5. 단순 뉴런 부터 깊은 모델 만들기

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader

import matplotlib.pyplot as plt
import matplotlib.image as image
import numpy as np

# transform_train, transform_val, transform_test 는
# PyTorch의 torchvision.transforms를 사용해서 이미지에 전처리/증강을 적용하는 파이프라인

transform_train = transforms.Compose([
    transforms.Resize((256, 256)),  # 이미지를 256x256 크기로 강제로 리사이즈,, 모델 학습 시 입력 이미지 크기를 통일하는 목적
    transforms.RandomHorizontalFlip(),  # 이미지를 50% 확률로 좌우 반전. 학습용 데이터(train)에만 사용되는 데이터 증강(Augmentation) 기법,, 같은 이미지를 다양하게 변형시켜 모델이 방향에 덜 민감하도록 학습
    transforms.ToTensor(),  # 이미지를 Tensor(텐서)로 변환 [H, W, C] 형탸
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )  # 이미지의 각 채널(R, G, B)을 정규화,, RGB 각 채널별 평균(mean)과 표준편차(std) 값
])

transform_val = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

transform_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

train_dataset = datasets.ImageFolder(
    root='./data/weather_dataset/train/',
    transform=transform_train
)
dataset_size = len(train_dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
test_dataset = datasets.ImageFolder(
    root='./data/weather_dataset/test/',
    transform=transform_test
)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


## 시각화

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 60
plt.rcParams.update({'font.size': 20})


def imshow(input):
    # torch.Tensor => numpy
    input = input.numpy().transpose((1, 2, 0))
    # undo image normalization
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    # display images
    plt.imshow(input)
    plt.show()


class_names = {
    0: "Cloudy",
    1: "Rain",
    2: "Shine",
    3: "Sunrise"
}

# load a batch of train image
iterator = iter(train_dataloader)

# visualize a batch of train image
imgs, labels = next(iterator)
out = torchvision.utils.make_grid(imgs[:4])
imshow(out)
print([class_names[labels[i].item()] for i in range(4)])





