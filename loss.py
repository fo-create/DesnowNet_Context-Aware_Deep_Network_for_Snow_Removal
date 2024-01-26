import torch
import torch.nn as nn
from torch.autograd import Variable

权重衰减是通过在损失函数中添加正则化项来限制模型参数的数值范围，从而防止过拟合。
函数计算了模型参数的 L2 范数的平方和，并将其添加到原始损失中。最后，返回添加了权重衰减的新损失值，该值用于进行反向传播和优化。
这有助于使模型参数保持在一个较小的范围内，防止它们在训练过程中变得过大。
def weight_decay_l2(loss, model, lambda_w):
    wdecay = 0
    for w in model.parameters():
        if w.requires_grad:
            wdecay = torch.add(torch.sum(w ** 2), wdecay)

    loss = torch.add(loss, lambda_w * wdecay)
    return loss

m: 一个图像，通常是模型预测的结果之一。
hat_m: 另一个图像，通常是与 m 对应的真实标签或其他预测结果。
tau: 损失金字塔的级别，默认为 6。
具体而言，它首先进行金字塔池化，然后计算相应层次上的均方差。最后，将所有层次的损失进行累加并返回。
def lw_pyramid_loss(m, hat_m, tau=6):
    """
     lightweight pyramid loss function
    :param m: one image
    :param hat_m: another image of the same size
    :param tau:the level of loss pyramid, default 4
    :return: loss
    """
    batch_size = m.shape[0]
    loss = 0
    for i in range(tau + 1):
        block = nn.MaxPool2d(2**i, stride=2**i)
        p1 = block(m)
        p2 = block(hat_m)
        loss += torch.sum((p1-p2)**2)
    return loss/batch_size


if __name__ == '__main__':
    device = 'cuda'
    img1 = torch.zeros([5, 1, 64, 64], device=device).requires_grad_()
    img2 = torch.ones_like(img1, device=device).requires_grad_() * 0.1
    loss = lw_pyramid_loss(img1, img2)
    module1 = nn.Conv2d(3,128,3)
    loss = weight_decay_l2(loss, module1, 0.2)
    print("finished")
