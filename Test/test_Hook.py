# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/10/1 17:16
import torch
from torch import nn

class test_model(nn.Module):
    def __init__(self):
        super(test_model, self).__init__()

        self.conv_16 = nn.Sequential(
            nn.Conv2d(1,16,(3,3),(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )
        self.conv_32 = nn.Sequential(
            nn.Conv2d(16,32,(3,3),(1,1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.linear_1 = nn.Sequential(
            nn.Linear(32,64),
            nn.ReLU()
        )
        self.linear_class = nn.Sequential(
            nn.Linear(64,5),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_16(x)
        x = self.conv_32(x)
        x = x.view(x.shape[0],x.shape[1])
        x = self.linear_1(x)
        return self.linear_class(x)

features = []

def hook(module, input, output):
    features.append(input)
    return None

net = test_model()

#  设置钩子
net.linear_class[0].register_forward_hook(hook)
a = torch.randn((3, 1, 28, 28))
net(a)
print(features)