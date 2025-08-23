import torch
import math
dtype = torch.float
device = torch.device("mps")

# 맥 M1, M2, M3 등과 같은 Apple Silicon 칩은 NVIDIA가 아닌 Apple이 자체적으로 개발한 ARM 기반의 프로세서입니다.
# CUDA는 NVIDIA의 GPU 전용 프로그래밍 인터페이스로, NVIDIA의 GPU 하드웨어에서만 작동합니다.
# 반면에, Apple은 자체 GPU 아키텍처를 개발했고, 프로그래밍 인터페이스인 Metal을 사용합니다.
# Metal은 Apple의 하드웨어에 최적화된 저수준의, 고효율의 그래픽스 및 컴퓨트 프로그래밍 인터페이스입니다.
# Apple Silicon 칩에서는 이 Metal을 기반으로 하는 Metal Performance Shaders (MPS)를 사용하여 GPU 가속을 제공합니다.
# MPS는 특히 머신러닝과 같은 작업을 위해 최적화된 라이브러리 세트를 포함합니다.
# 따라서, 맥 M1 실리콘과 같은 Apple Silicon에서는 CUDA 대신 MPS를 사용하는 것입니다.


# Create random input and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Randomly initialize weights
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(5000):
    # Forward pass: compute predicted y
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

# Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights using gradient descent
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d


print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')