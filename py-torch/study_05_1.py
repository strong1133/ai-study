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

# PyTorch에서 데이터셋을 불러오고, 학습/검증/테스트용 DataLoader를 만드는 과정.

train_dataset = datasets.ImageFolder(
    root='./data/weather_dataset/train/',  # 이미지를 불러와 라벨링.
    transform=transform_train
)  # transform=transform_train: 불러온 이미지에 Resize + Flip + ToTensor + Normalize 같은 변환 적용

# 데이터셋 크기 계산 및 분할
dataset_size = len(train_dataset)  # 전체 train 데이터 개수
train_size = int(0.8 * dataset_size)  # 80%를 학습(train)용으로
val_size = dataset_size - train_size  # 나머지 20%를 검증(validation)용

train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])  # 학습용과 검증용을 랜덤하게 분할
test_dataset = datasets.ImageFolder(
    root='./data/weather_dataset/test/',
    transform=transform_test
)

# DataLoader 생성
# batch_size=64: 데이터를 64개씩 묶어서 한 번에 학습/평가.
# shuffle=True (train만): 학습용 데이터는 매 epoch 마다 섞어서 사용 → 과적합 방지 & 일반화 성능 향상.
# shuffle=False (val, test): 검증/테스트는 성능 평가가 목적이므로 순서를 섞지 않음.
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
# → 학습 데이터는 매번 epoch 마다 데이터 순서를 섞기(shuffle=True) 때문에 일반적으로 모델 학습에 사용

val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
# → 검증 데이터는 모델 평가 시 사용하므로 순서를 섞지 않음

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
# → 테스트 데이터도 최종 평가용이라 순서를 유지

## 시각화
plt.rcParams['figure.figsize'] = (12, 8)  # 그래프 기본 크기를 가로 12, 세로 8로 설정
plt.rcParams['figure.dpi'] = 60  # 해상도(DPI) 60으로 설정
plt.rcParams.update({'font.size': 20})  # 폰트 크기를 20으로 설정

# 이미지 시각화 함수 정의
# 모델 학습 때 편하게 쓰려고 정규화(Normalize) 했던 걸, 사람이 보기 쉽게 다시 되돌리는 과정
# 원래 이미지 픽셀 값은 0~255 범위.
# 보통 신경망 학습에서는 이런 큰 숫자보다 -1 ~ 1 또는 0 ~ 1 범위로 바꿔 주는 게 안정적이에요.
# → 학습이 더 빨라지고, 숫자가 균일하니 모델이 잘 수렴함.

# 왜 다시 역정규화(복원) 하는가?
# 사람이 보기에는 픽셀 값이 -1 ~ 1이면 이상하게 보임 (검은 화면, 색감 깨짐).
# 그래서 시각화할 때는 다시 0~1 범위로 돌려놔야 정상적인 이미지처럼 보임.

# 왜 clip(0,1) 하는가?
# 정규화 → 역정규화 과정에서 계산 오차 때문에
# 일부 값이 0보다 작거나 1보다 클 수도 있음.
# 예: -0.05, 1.02 이런 값,,
# 이런 값이 matplotlib에서 색으로 표현되면 깨질 수 있어서
# np.clip(input, 0, 1) 으로 안전하게 잘라줌.

def imshow(input):
    input = input.numpy().transpose((1, 2, 0))  # → Tensor(C,H,W)를 numpy 배열(H,W,C)로 변환하여 matplotlib이 인식 가능한 형식으로 변경
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5]) # → 데이터 전처리 시 Normalize 했던 mean/std 값 (여기선 [-1,1] 범위로 정규화한 것)
    input = input * std + mean  # → 역정규화: normalize 했던 이미지를 다시 원래 범위(0~1)로 복원
    input = np.clip(input, 0, 1) # → 값이 0~1 범위를 벗어나지 않도록 클리핑
    plt.imshow(input)
    plt.show() # → 이미지 출력

# 클래스 인덱스와 실제 라벨 이름 매핑
class_names = {
    0: "Cloudy",
    1: "Rain",
    2: "Shine",
    3: "Sunrise"
}

# 학습 데이터에서 배치 하나 불러오기
iterator = iter(train_dataloader)  # DataLoader는 반복자(iterator) 형태 → iter()로 변환
imgs, labels = next(iterator) # next()로 배치 하나 꺼냄 → imgs: 이미지 텐서, labels: 정답 라벨

# 불러온 이미지 중 4개를 하나의 grid(격자)로 합치기
out = torchvision.utils.make_grid(imgs[:4])

# 합친 이미지 grid를 시각화
imshow(out)

# 라벨을 숫자에서 실제 클래스 이름으로 변환 후 출력
print([class_names[labels[i].item()] for i in range(4)])
