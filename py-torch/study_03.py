## 3. 파이토치 차원 접근, 형변환

import torch

device = torch.device('mps')

# 텐서 차원접근
tensor = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], device=device)
print(tensor)
print(tensor[0])  # first row >> tensor([1, 2, 3, 4], device='mps:0')
print(tensor[:, 0])  # first column >> tensor([1, 5, 9], device='mps:0')
print(tensor[..., -1])  # last column >> tensor([ 4,  8, 12], device='mps:0')

# 텐서 붙이기
tensor = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], device=device)

# dim 은 텐서를 이어 붙이기 위한 축
# 0번 축(행)을 기준으로 이어 붙이기
result = torch.cat((tensor, tensor, tensor), dim=0)
print(result)

# 1번 축(행)을 기준으로 이어 붙이기
result = torch.cat((tensor, tensor, tensor), dim=1)
print(result)

# 텐서 형변환
a = torch.tensor([2], dtype=torch.int64)
b = torch.tensor([5.0])

print(a + b)  # >> 7.
print(a + b.type(torch.int64))  # >> 7

# 텐서 모양 변경
a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
print(a)
b = a.view(4, 2)
print(b)

a[0] = 7  # a를 변경 하면 b도 변경 -> 원치 않고 값만 복사를 원하면 copy 해야함 .clone
print(b)

c = a.clone().view(4, 2)
a[0] = 123
print(b)
print(c)

# 텐서의 차원 교환
a=torch.rand(64, 32, 3)
print(a.shape)
print(a)

b = a.permute(2,1,0) # 차원 교환 -> (2번째 축, 1번째 축, 0번째 축)의 형태
print(b.shape)
print(b)


a=torch.rand(2, 3, 4, 5, 6)
print(a)