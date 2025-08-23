## 1. 파이토치 개요
# 파이토치의 텐서는 Numpy 배열과 매우 유사하다.
# 파이토치에서도 GPU 사용이 가능하다.


import torch

# 메트릭스
arrData = [
    [1, 2],
    [3, 4]
]
print(arrData)

# 파이토치에서 배열 데이터를 '텐서'형으로 변환
tensorSample = torch.tensor(arrData)
print(tensorSample) # 텐서 프린트

# 현재 텐서 데이터가 cpu에 할당인지 gpu 할당인지 확인
# 아직 gpu 이동 명령이 없었으니 기본적으로 cpu
print(tensorSample.is_cuda) # false
print(tensorSample.is_mps) # false
print(tensorSample.is_cpu) # true

# CPU -> GPU 이동
# tensorSample = tensorSample.cuda()
# .cuda()는 nvidia 기반에서만 사용 가능

# 애플 실리콘에서는 mps 를 사용해야함.
device = torch.device("mps")
tensorSample = tensorSample.to(device)

print(tensorSample.is_mps) # true
print(tensorSample.is_cpu) # false

# GPU -> CPU
tensorSample = tensorSample.cpu()
print(tensorSample.is_mps) # false
print(tensorSample.is_cpu) # true

