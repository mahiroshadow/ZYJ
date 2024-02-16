import torch
from torch import nn
import torch.nn.functional as F

# 这里是双分支

class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, k_size=3):
        super(CBAM, self).__init__()

        # 计算一个中间通道数 mid_channel，其值等于总通道数 channel 除以某个缩小因子 reduction。
        # 通过使用中间通道数来减少模型中的参数数量，提高模型的计算效率。
        mid_channel = channel // reduction  # 3
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mid_channel, out_features=channel)
        )
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()

        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1
                                , kernel_size=k_size, stride=1, padding=k_size // 2)

    def channel_attention(self, x):
        # 使用自适应池化缩减map的大小，保持通道不变
        # self.avg_pool(x)：对输入张量 x 进行平均池化操作，降低空间维度。48*1*1
        # .view(x.size(0), -1)：将池化后的结果展平为一维张量，保留批次大小（x.size(0)）。48个元素的一维张量
        # 通过共享的多层感知机（MLP）对展平后的一维张量进行处理，可能包括线性变换和激活函数等。
        avg_out = self.shared_MLP(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        max_out = self.shared_MLP(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        return self.sigmoid1(avg_out + max_out)  # 48个元素的一维张量，激活函数

    def spatial_attention(self, x):
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)  # 通道之间的元素求平均值，形状为1*7*7
        maxout, _ = torch.max(x, dim=1, keepdim=True)  # 通道之间的元素求最大值，形状为1*7*7
        out = torch.cat([avgout, maxout], dim=1)  # 2*7*7
        out = self.sigmoid2(self.conv2d(out))  # 1*7*7
        return out

    def forward(self, x):
        out = self.channel_attention(x) * x  # 与输入相乘，输出为48*7*7
        out = self.spatial_attention(out) * out  # 1*7*7  *  48*7*7  =  48*7*7
        return out


# 一个残差模块
class Block(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1, same_shape=True):
        super(Block, self).__init__()
        self.same_shape = same_shape
        if not same_shape:
            strides = 2
        self.strides = strides
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        if not same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.block(x)
        if not self.same_shape:
            x = self.bn3(self.conv3(x))
        return F.relu(out + x)


class Inception(nn.Module):
    def __init__(self, in_channel,out_channel):  # CNN_embed_dim参数设置中为64
        super().__init__()

        self.ch1, self.ch2= 8,16
        self.k1, self.k2 = (3, 3), (3, 3)  # 2d kernal size
        self.s1, self.s2 = (2, 2), (2, 2)  # 2d strides
        self.pd1, self.pd2, self.pd3 = (0, 0), (0, 0), (0, 0)# 2d padding
        self.drop_p=0.4
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1,
                      padding=self.pd1),
            nn.BatchNorm2d(self.ch1, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(self.ch1, self.ch2, 2, stride=2)  # res

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch2, out_channels=out_channel, kernel_size=self.k2, stride=self.s2,
                      padding=self.pd2),
            nn.BatchNorm2d(out_channel, momentum=0.01),
            nn.ReLU(inplace=True),

        )
        self.layer2 = self._make_layer(out_channel, out_channel, 2, stride=1)  # res

    def _make_layer(self, in_channel, out_channel, block_num, stride=1):
        layers = []
        if stride != 1:
            layers.append(Block(in_channel, out_channel, stride, same_shape=False))
        else:
            layers.append(Block(in_channel, out_channel, stride))

        for i in range(1, block_num):
            layers.append(Block(out_channel, out_channel))
        return nn.Sequential(*layers)

    def forward(self, x_2d):
        # shape:torch.Size([6, 3, 28, 28])
        cnn_embed_seq = []
        # CNNs
        x = x_2d
        x = self.conv1(x)
        x = self.layer1(x)
        x=self.conv2(x)
        x=self.layer2(x)
        x = F.dropout(x, p=self.drop_p)
        cnn_embed_seq.append(x)
        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        output = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1).squeeze(dim=1)
        return output


class MEROIInception1(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(MEROIInception1, self).__init__()
        self.Inception1_1 = Inception(in_channel=3, out_channel=48)

        self.Inception2_1 = Inception(in_channel=3, out_channel=48)

        self.maxpool = nn.MaxPool2d(kernel_size=2)  # 池化层步长默认与池化窗口大小相同
        self.cbam = CBAM(48)  # 单层24，双层48

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=3 * 3 * 48 * 2, out_features=1024),  # 单层14*14*24，双层7*7*48
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=1024, out_features=num_classes),
        )


    def forward(self, x1, x2):
        out_x1 = self.Inception1_1(x1)  #
        out_x1 = self.maxpool(out_x1)  #
        out_x1 = self.cbam(out_x1)
        out_x2= self.Inception2_1(x1)
        out_x2=self.maxpool(out_x2)
        x = torch.cat((out_x1, out_x2), 1)
        x = x.reshape(x.shape[0], -1)  # flatten 变成全连接层的输入
        x1 = out_x1.reshape(x1.shape[0], -1)  # 对比需要展平
        x2 = out_x2.reshape(x2.shape[0], -1)  # 若有对比损失，需要展平x1，x2，并return
        x = self.fc(x)
        return x1, x2, x
