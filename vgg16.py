from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from torchvision import models
import matplotlib.pyplot as plt

# 设置每次批量读取的图片数目、学习率和轮数
batch_size = 10
learning_rate = 0.00002
epoch = 100

num_classes=20
train_dir = r'C:\Users\11138\Desktop\训练'
val_dir = r'C:\Users\11138\Desktop\测试'


# 对图像进行归一化处理以适合VGG网络
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

# 读取训练数据
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)

# 读取测试数据
val_datasets = datasets.ImageFolder(val_dir, transform=val_transforms)
val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True)

# 创建VGG网络
class VGGNet(nn.Module):
    def __init__(self, num_classes=2):
        super(VGGNet, self).__init__()
        # 使用官方训练模型
        net = models.vgg16(pretrained=True)
        net.classifier = nn.Sequential()
        self.features = net
        self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 128),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


model = VGGNet(num_classes)
# 使用gpu加速
if torch.cuda.is_available():
    model.cuda()
# 参数记录
params = [{'params': md.parameters()} for md in model.children()
          if md in [model.classifier]]
# 优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# 损失函数为交叉熵
loss_func = nn.CrossEntropyLoss()

Loss_list = []
Accuracy_list = []

# 正确率获取
def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total

for epoch in range(epoch):
    print('epoch {}'.format(epoch + 1))

    train_loss = 0.
    train_acc = 0.
    # 训练网络
    for batch_x, batch_y in train_dataloader:
        batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda()
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        train_loss += loss.item()

        train_acc += get_acc(out, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
        train_datasets)), train_acc / (len(train_datasets))))


    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    # 测试网络
    for batch_x, batch_y in val_dataloader:
        with torch.no_grad():
            batch_x, batch_y = Variable(batch_x, volatile=True).cuda(), Variable(batch_y, volatile=True).cuda()
        with torch.no_grad():
            out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.item()

        eval_acc += get_acc(out, batch_y)
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        val_datasets)), eval_acc / (len(val_datasets))))
        
    Loss_list.append(eval_loss / (len(val_datasets)))
    Accuracy_list.append(100 * eval_acc / (len(val_datasets)))

# 创建可视化图表来观察调整学习率
x1 = range(0, len(Accuracy_list))
x2 = range(0, len(Loss_list))
y1 = Accuracy_list
y2 = Loss_list
plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('Test accuracy vs. epoches')
plt.ylabel('Test accuracy')
plt.subplot(2, 1, 2)
plt.plot(x2, y2, '.-')
plt.xlabel('Test loss vs. epoches')
plt.ylabel('Test loss')
plt.show()
# plt.savefig("accuracy_loss.jpg")
