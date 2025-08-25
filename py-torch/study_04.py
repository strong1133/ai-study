## 4. 자동 미분, 기울기

import torch

device = torch.device('mps')

# requires_grad true 면 기울기를 갖는다
x = torch.tensor([3.0, 4.0], requires_grad=True)
y = torch.tensor([1.0, 2.0], requires_grad=True)
z = x + y
print(z)
print(z.grad_fn)

out = z.mean()
print(out)
print(out.grad_fn)
