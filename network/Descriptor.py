import torch
import torch.nn as nn
from Inceptionv4 import InceptionV4


class DP(nn.Module):
    # dilation pyramid
    def __init__(self, in_channel=768, depth=77, gamma=4):
        super(DP, self).__init__()
        self.gamma = gamma
        block = []
        block = []：创建一个空列表，用于存储模型的各个卷积块。
        for i in range(gamma + 1):循环创建金字塔中的每一层。
            block.append(nn.Conv2d(in_channel, depth, 3, 1, padding=2 ** i, dilation=2 ** i))使用空洞卷积，设置对应的填充。设置卷积核的空洞（膨胀）率。
        self.block = nn.ModuleList(block)

    def forward(self, feature):
        for i, block in enumerate(self.block):遍历每一层金字塔，对输入特征应用相应的空洞卷积。
            if i == 0:如果是金字塔的第一层，直接使用当前卷积块计算输出。
                output = block(feature)
            else:
                output = torch.cat([output, block(feature)], dim=1)对于其他层，将当前卷积块的输出与之前层的输出在通道维度上拼接。
        return output
这个模型的作用是构建一个金字塔结构的空洞卷积网络，通过多层次、不同膨胀率的卷积核来捕捉不同尺度的特征。

class Descriptor(nn.Module):
    def __init__(self, input_channel=3, gamma=4):
        super(Descriptor, self).__init__()
        self.backbone = InceptionV4(input_channel)
        self.DP = DP(gamma=gamma)

    def forward(self, img):
        feature = self.backbone(img)
        f = self.DP(feature)
        return f


if __name__ == '__main__':
    device = 'cpu'
    Descriptor_1 = Descriptor().to(device)
    img = torch.zeros([1, 3, 200, 200]).to(device)
    f = Descriptor_1(img)
    f.mean().backward()
    print("finished")
