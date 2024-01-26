import torch
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
这段代码加载了一个检查点文件（'./checkpoints_ite100000.pth'），提取了其中的 loss_window，并绘制了迭代次数与对数损失之间的关系曲线。
if __name__ == '__main__':
    fontbd = FontProperties(fname=r'c:\windows\fonts\timesbd.ttf', size=14)
    font = FontProperties(fname=r'c:\windows\fonts\times.ttf',size=14)
    checkpoint = torch.load('./checkpoints_ite100000.pth',map_location='cuda:0')
    loss_window = checkpoint['loss_window']
    loss_window = torch.log10(torch.stack(loss_window).reshape(1000, 100).mean(dim=1)).cpu()
    这里使用 torch.log10 对损失取对数，这是为了更好地显示损失的变化趋势。图中 x 轴表示迭代次数，y 轴表示对数损失。
    step = torch.linspace(1,100001,1000)
    plt.plot(step, loss_window, lw=1.5)
    plt.xlabel('Iterations', fontproperties=font)
    plt.ylabel('$\log_{10}(Loss)$', fontproperties=font)
    plt.show()
