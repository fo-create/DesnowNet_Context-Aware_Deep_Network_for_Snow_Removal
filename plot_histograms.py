import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
这段代码通过直方图可视化了两个图像的像素值分布情况。这两个图像分别是:

(a) 来自 a_gt，计算方法为 (img_tensor - (1 - mask_tensor) * gt_tensor)/(1e-8 + mask_tensor) * (mask_tensor != 0)。
(b) 来自 za，计算方法为 img_tensor - (1 - mask_tensor) * gt_tensor。
在直方图中，x 轴表示像素值的范围，y 轴表示该范围内像素值的数量（对数尺度）。左侧的直方图 (a) 对应 a_gt，右侧的直方图 (b) 对应 za。

这样的可视化可以帮助观察图像的像素值分布，有助于了解两者之间的差异

# Set up the matplotlib figure
f, axes = plt.subplots(1, 2, figsize=(7, 7), sharex=True)

# Generate a random univariate dataset
img_data = Image.open("../data/Test/Snow100K-L/synthetic/beautiful_smile_00003.jpg").convert('RGB')
mask_data = Image.open("../data/Test/Snow100K-L/mask/beautiful_smile_00003.jpg").convert('L')
gt_data = Image.open("../data/Test/Snow100K-L/gt/beautiful_smile_00003.jpg").convert('RGB')

toTensor = transforms.ToTensor()
img_tensor = toTensor(img_data).to(device='cuda:0')
mask_tensor = toTensor(mask_data).to(device='cuda:0')
gt_tensor = toTensor(gt_data).to(device='cuda:0')
img_index = (mask_tensor>0).repeat(3,1,1)
with torch.no_grad():
    a_gt = (img_tensor - (1 - mask_tensor) * gt_tensor)/(1e-8 + mask_tensor) * (mask_tensor != 0)
    za = img_tensor - (1 - mask_tensor) * gt_tensor
    a_gt = a_gt.cpu().numpy().reshape(a_gt.shape[0]*a_gt.shape[1]*a_gt.shape[2])
    za = za.cpu().numpy().reshape(za.shape[0]*za.shape[1]*za.shape[2])
num_bins = 30
plt.subplot(1,2,1)
plt.hist(a_gt, bins=num_bins, color="r", log=True)
plt.title('(a)', y=-0.10)
plt.subplot(1,2,2)
plt.hist(za, bins=num_bins, color="b", log=True)
plt.title('(b)', y=-0.10)

plt.tight_layout()
plt.show()
