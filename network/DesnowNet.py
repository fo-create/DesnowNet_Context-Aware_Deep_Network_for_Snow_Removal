import torch
import torch.nn as nn
from Descriptor import Descriptor
from Recovery_Submodule import R_t, Pyramid_maxout


这段代码定义了一个 Translucency Recovery (TR) 模块，其中包含了 Descriptor 和 R_t 两个子模块。
class TR(nn.Module):
    # translucency recovery(TR) module
    def __init__(self, input_channel=3, beta=4, gamma=4):
        super(TR, self).__init__()
        self.D_t = Descriptor(input_channel, gamma)
        self.R_t = R_t(385, beta)
forward 方法定义了模块的前向传播。它接受输入 x，然后通过 Descriptor 模块 D_t 提取特征 f_t。
接着，通过 R_t 模块 R_t 对输入 x 和特征 f_t 进行进一步处理，最终返回输出 y_、特征 f_c、z_hat 和 a。
    def forward(self, x, **kwargs):
        f_t = self.D_t(x)
        y_, f_c, z_hat, a = self.R_t(x, f_t, **kwargs)
        return y_, f_c, z_hat, a
这个模块的整体目的是进行雪花图像的透明度恢复，其中 D_t 负责提取特征，而 R_t 负责使用这些特征进行透明度的预测和修复。

class TR_new(nn.Module):
    这段代码定义了一个新的 Translucency Recovery (TR) 模块，与之前的模块相比，引入了两个 Descriptor 模块，以及两个 Pyramid_maxout 模块。
    # A new translucency recovery(TR) module with two descriptors
    def __init__(self, input_channel=3, beta=4, gamma=4):
        super(TR_new, self).__init__()
        self.D_t_1 = Descriptor(input_channel, gamma)
        self.D_t_2 = Descriptor(input_channel, gamma)
        self.SE = Pyramid_maxout(385, 1, beta)
        self.AE = Pyramid_maxout(385, 3, beta)
SE 和 AE 是两个 Pyramid_maxout 模块的实例化，分别用于生成透明度的估计 z_hat 和透明度相关的特征 a。
    def forward(self, x, **kwargs):
        f_t_1 = self.D_t_1(x)
        z_hat = self.SE(f_t_1)
        z_hat[z_hat >= 1] = 1
        z_hat[z_hat <= 0] = 0
        z_hat_ = z_hat.detach()
        f_t_2 = self.D_t_2(x)
        a = self.AE(f_t_2)
        # yield estimated snow-free image y'
        y_ = (z_hat_ < 1) * (x - a * z_hat_) / (1 - z_hat_ + 1e-8) + (z_hat_ == 1) * x
        y_[y_ >= 1] = 1
        y_[y_ <= 0] = 0
        # yield feature map f_c
        f_c = torch.cat([y_, z_hat_, a], dim=1)
        return y_, f_c, z_hat, a
forward 方法定义了模块的前向传播。首先，通过 D_t_1 提取输入 x 的第一个特征 f_t_1，然后通过 SE 生成透明度的估计 z_hat。
接着，通过 D_t_2 提取输入 x 的第二个特征 f_t_2，并通过 AE 生成透明度相关的特征 a。最后，根据 z_hat 和 a 生成估计的雪花图像 y_，并拼接生成特征图 f_c。
class TR_za(nn.Module):
    # A  translucency recovery(TR) module predict z\times a
    def __init__(self, input_channel=3, beta=4, gamma=4):
        super(TR_za, self).__init__()
        self.D_t = Descriptor(input_channel, gamma)
        self.SE = Pyramid_maxout(385, 1, beta)
        self.SAE = Pyramid_maxout(385, 3, beta)
forward 方法定义了模块的前向传播。首先，通过 D_t 提取输入 x 的特征 f_t，然后通过 SE 生成透明度的估计 z_hat 和通过 SAE 生成透明度相关的特征 za。
接着，对 z_hat 和 za 进行截断，确保它们在合理的范围内。最后，根据 z_hat 和 za 生成估计的雪花图像 y_，并拼接生成特征图 f_c。
    def forward(self, x, **kwargs):
        f_t = self.D_t(x)
        z_hat = self.SE(f_t)
        za = self.SAE(f_t)
        z_hat[z_hat >= 1] = 1
        z_hat[z_hat <= 0] = 0
        za[za >= 1] = 1
        za[za <= 0] = 0
        # yield estimated snow-free image y'
        y_ = (z_hat < 1) * (x - za) / (1 - z_hat + 1e-8) + (z_hat == 1) * x
        y_[y_ >= 1] = 1
        y_[y_ <= 0] = 0
        # yield feature map f_c
        f_c = torch.cat([y_, z_hat, za], dim=1)
        return y_, f_c, z_hat, za
这个模块的设计旨在通过透明度的估计 z_hat 和透明度相关的特征 za 来预测 z * a，其中 z 和 a 对应于透明度和透明度相关的特征。
class RG(nn.Module):
    # the residual generation (RG) module
    def __init__(self, input_channel=7, beta=4, gamma=4):
        super(RG, self).__init__()
        self.D_r = Descriptor(input_channel, gamma)
        block = []
        for i in range(beta):
            block.append(nn.Conv2d(385, 3, 2 * i + 1, 1, padding=i))
        self.conv_module = nn.ModuleList(block)
        self.activation = nn.Tanh()

    def forward(self, f_c):
        f_r = self.D_r(f_c)
        for i, module in enumerate(self.conv_module):
            if i == 0:
                r = module(f_r)
            else:
                r += r + module(f_r)
        r = self.activation(r)
        return r
forward 方法定义了模块的前向传播。首先，通过 D_r 提取输入 f_c 的特征 f_r。然后，通过 conv_module 处理 f_r，其中每个卷积层都被应用于 f_r。
最后，通过 Tanh 激活函数将输出截断到 [-1, 1] 的范围内，得到生成的残差图像 r。

class DesnowNet(nn.Module):
    # the DesnowNet
    def __init__(self, input_channel=3, beta=4, gamma=4, mode='original'):
        super(DesnowNet, self).__init__()
        if mode == 'original':
            self.TR = TR(input_channel, beta, gamma)
        elif mode == 'new_descriptor':
            self.TR = TR_new(input_channel, beta, gamma)
        elif mode == 'za':
            self.TR = TR_za(input_channel, beta, gamma)
        else:
            raise ValueError("Invalid architectural mode")
        self.RG = RG(beta=beta, gamma=gamma)

    def forward(self, x, **kwargs):
        y_, f_c, z_hat, a = self.TR(x, **kwargs)
        r = self.RG(f_c)
        y_hat = r + y_
        return y_hat, y_, z_hat, a


if __name__ == '__main__':
    device = 'cuda'
    net = DesnowNet().to(device)
    mask = torch.zeros([2, 1, 64, 64]).to(device)
    img = torch.zeros([2, 3, 64, 64]).to(device)
    y_hat, y_, z_hat, a = net(img, mask=mask)
    y_hat.mean().backward()
    print("finished")
