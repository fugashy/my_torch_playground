# -*- coding: utf-8 -*-
from sys import version
import torch
import numpy as np

print(f'python:\n{version}')
print(f'torch: {torch.__version__}')

if torch.backends.mps.is_available():
    print('mps is available')
    torch.device('mps')


# listからデータ作れるよ
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(f'generate tensor from list:\n{x_data}')


# numpy からも作れるよ
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f'generate tensor from np:\n{x_np}')


# npみたいに，1だけのtensorとかrandomな値をもつtensorも作れるよ
x_ones = torch.ones_like(x_data)
print(f'ones tensor:\n{x_ones}')
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f'rand tensor:\n{x_rand}')


# shape指定
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)


# attributes
tensor = torch.rand(3, 4)
print(f'shape: {tensor.shape}')  # ただのtupleではない
print(f'dtype: {tensor.dtype}')
print(f'device tensor is stored on: {tensor.device}')

# 大きいtensorをdevice間でやり取りすると時間とメモリがかかるとのこと
if torch.backends.mps.is_available():
    tensor = tensor.to('mps')
print(f'device tensor is stored on: {tensor.device}')

# 要素アクセスとスライシング
tensor = torch.ones(4, 4)
tensor[:, 1] = 0  # 1列目を0とする
print(f'first row: {tensor[0]}')
print(f'first col: {tensor[:, 0]}')
print(f'last  col: {tensor[..., -1]}')

# concat
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(f'concat:\n{t1}')

# mul
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
