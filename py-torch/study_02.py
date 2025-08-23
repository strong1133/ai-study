## 2. 파이토치 텐서
## 텐서는 고차원 데이터를 처리하기 위한 자료구조
## 텐서는 shape, data_type, device 3가지의 속성을 가진다.

import torch

device = torch.device('mps')

# randn : "random normal"의 약자 → 평균 0, 표준편차 1 인 표준 정규분포에서 샘플을 뽑습니다.
# (3, 4) : 생성할 텐서의 shape → 3행 4열짜리 2D 텐서를 만듭니다.
tensorSample01 = torch.randn(3, 4)  # 정규분포 샘플 3행 4열짜리 2D 텐서
print(tensorSample01)
print(f"Shape: {tensorSample01.shape}")  # shape 출력
print(f"Data Type: {tensorSample01.dtype}")  # data_type 출력
print(f"Device: {tensorSample01.device}")  # device 출력 >> CPU

# CPU -> GPU
tensorSample01 = tensorSample01.to(device)
print(f"Device: {tensorSample01.device}")  # device 출력 >> GPU

# 직접 텐서 초기화
arr = [
    [1, 2],
    [3, 4]
]
tensorSample01 = torch.tensor(arr, device=device)  # device= 옵션이 없으면 cpu로 기본할당

print(tensorSample01)
print(f"Shape: {tensorSample01.shape}")
print(f"Data Type: {tensorSample01.dtype}")
print(f"Device: {tensorSample01.device}")

# numpy 예제

a = torch.tensor([5])
b = torch.tensor([7])
c = (a + b).numpy()  # Numpy 로 형변환

print(a)
print(b)
print(c)
print(f"Type A: {type(a)}")
print(f"Type C: {type(c)}")

result1 = b * 10
print(f"Result1: {result1}")

result2 = c * 10
print(f"Resu lt2: {result2}")

tensor = torch.from_numpy(result2)
print(f"Type tensor: {type(tensor)}")

## 텐서로 부터 텐서 초기화

x= torch.tensor(
    [
        [5,7],
        [1,2]
    ]
)

print(x)

x_ones = torch.ones_like(x)
print(x_ones)

x_rand = torch.randn_like(x, dtype=torch.float, device=device)
print(x_rand)