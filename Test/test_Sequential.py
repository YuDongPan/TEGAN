# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/10/1 17:06
import torch
from torch import nn

fcUnit = 100
D = 10

conv_layer = nn.Sequential(
            nn.Linear(fcUnit, D),
            nn.PReLU(),
            nn.Linear(D, 1),
            nn.Sigmoid())

x = torch.randn((10, fcUnit))

out = conv_layer(x)

print("out:", out)