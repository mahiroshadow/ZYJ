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


# 搭一个Inception模块,要么用Inception,要么用ROINet
class Inception(nn.Module):
    def __init__(self, in_channels=3, out_channels=6):
        super(Inception, self).__init__()
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        # 4个分支在dim=1即沿着channel(张量是0batch,1channel,2weight,3height)上进行concatenate。6+6+6+6=24(输出通道数)
        return torch.cat(outputs, dim=1)


# 搭一个ROINet,要么用ROINet,要么用Inception
class MEROINet(nn.Module):
    def __init__(self, in_channels=3):
        super(MEROINet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 12, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(12),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(12, 24, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(24),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MyMEROI(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(MyMEROI, self).__init__()
        self.MEROINet1 = MEROINet()
        self.MEROINet2 = MEROINet()
        self.cbam = CBAM(24)

        self.fc = torch.nn.Sequential(
            # 仅一层Linear的情况：
            # torch.nn.Dropout(0.5),
            # torch.nn.Linear(in_features=2*56 * 56 * 24, out_features=num_classes),
            # 28->2次池化->12*7*7； 224->56*56*24, 且因为后面要concat所以*2

            # 若是两层及以上Linear的情况：
            torch.nn.Linear(in_features=2 * 56 * 56 * 24, out_features=1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=1024, out_features=num_classes),
        )

    def forward(self, x1, x2):
        out_x1 = self.MEROINet1(x1)
        out_x2 = self.MEROINet2(x2)
        out_x1 = self.cbam(out_x1)

        out_x1 = out_x1.reshape(out_x1.shape[0], -1)  # flatten 变成全连接层的输入
        out_x2 = out_x2.reshape(out_x2.shape[0], -1)

        x = torch.cat((out_x1, out_x2), 1)
        # print(x.size())
        x = self.fc(x)

        return x


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


class CNN(nn.Module):
    def __init__(self, in_channel, fc_hidden1, fc_hidden2, drop_p, CNN_embed_dim, fc_in_dim):  # CNN_embed_dim参数设置中为64
        super().__init__()
        # self.high = high
        # self.wide = wide
        # self.input_type = input_type
        self.CNN_embed_dim = CNN_embed_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3 = 24, 48, 48
        self.k1, self.k2, self.k3 = (3, 3), (3, 3), (3, 3)  # 2d kernal size
        self.s1, self.s2, self.s3 = (2, 2), (2, 2), (2, 2)  # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # conv2D output shapes
        # self.conv1_outshape = conv2D_output_size((self.high, self.wide), self.pd1, self.k1,
        #                                          self.s1)  # Conv1 output shape
        # self.conv2_outshape = conv2D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)
        # self.conv3_outshape = conv2D_output_size(self.conv2_outshape, self.pd3, self.k3, self.s3)

        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1,
                      padding=self.pd1),
            nn.BatchNorm2d(self.ch1, momentum=0.01),
            nn.ReLU(inplace=True),
            #
        )

        self.layer1 = self._make_layer(self.ch1, self.ch2, 2, stride=2)  # res

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k2, stride=self.s2,
                      padding=self.pd2),
            nn.BatchNorm2d(self.ch3, momentum=0.01),
            nn.ReLU(inplace=True),

        )
        self.layer2 = self._make_layer(self.ch3, self.ch3, 2, stride=1)  # res

        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3,
        #               padding=self.pd3),
        #     nn.BatchNorm2d(self.ch3, momentum=0.01),
        #     nn.ReLU(inplace=True),
        # )
        # self.layer3 = self._make_layer(self.ch3, self.ch3, 2, stride=2)  # res
        #
        # self.fc1 = nn.Linear(self.CNN_embed_dim,fc_in_dim)  # 输入和输出维度

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
        # dim=[6,8,13,13]
        x = self.conv1(x)
        # dim=[6,8,7,7]
        x = self.layer1(x)
        # x = self.conv2(x)
        # x = self.layer2(x)
        # x = x.view(x.size(0), -1)  # flatten the output of conv
        x = F.dropout(x, p=self.drop_p)
        #
        # # FC layers
        # x = F.relu(self.fc1(x))
        cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        output = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # else:
        #     x = self.conv1(x_2d)
        #     x = self.layer1(x)
        #     x = self.conv2(x)
        #     x = self.layer2(x)
        #     x = self.conv3(x)
        #     x = self.layer3(x)
        #     x = x.view(x.size(0), -1)  # flatten the output of conv
        #     x = F.dropout(x, p=self.drop_p, training=self.training)
        #
        #     # FC layers
        #     output = F.relu(self.fc1(x))
        return output


class RNN(nn.Module):
    def __init__(self, CNN_embed_dim, h_RNN_layers, h_RNN, h_FC_dim, drop_p, num_classes):
        super().__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layers
        self.h_RNN = h_RNN  # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)

    def forward(self, x_RNN):
        rnn_out, (_, _) = self.LSTM(x_RNN, None)

        # FC layers
        x = self.fc1(rnn_out[:, -1, :])  # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        return x


class MEROIInception(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(MEROIInception, self).__init__()
        self.Inception1_1 = Inception(in_channels=3, out_channels=6)
        self.Inception1_2 = Inception(in_channels=24, out_channels=12)
        self.Inception2_1 = Inception(in_channels=3, out_channels=6)
        self.Inception2_2 = Inception(in_channels=24, out_channels=12)
        self.maxpool = nn.MaxPool2d(kernel_size=2)  # 池化层步长默认与池化窗口大小相同
        self.cbam = CBAM(48)  # 单层24，双层48

        self.image_CNN = CNN(in_channel=in_channels, fc_hidden1=64, fc_hidden2=64, drop_p=0.4, CNN_embed_dim=64, fc_in_dim=256)
        # self.image_RNN = RNN(CNN_embed_dim=64, h_RNN_layers=1, h_RNN=64, h_FC_dim=64, drop_p=0.4,
        #                      num_classes=12)  # h_RNN, h_FC_dim

        # 把48*28*28的Inception输出变成24*28*28
        # self.conv = torch.nn.Sequential(
        #     torch.nn.Conv2d(48, 24, kernel_size=1),
        #     torch.nn.BatchNorm2d(24),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.MaxPool2d(kernel_size=2),
        # )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=7 * 7 * 48 * 2, out_features=1024),  # 单层14*14*24，双层7*7*48
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=1024, out_features=num_classes),
        )


    def forward(self, x1, x2):
        # dim=[6,3,28,28]
        # print(x1.shape)
        # print(f"origin_f2_shape:{x2.shape}")
        out_x1 = self.Inception1_1(x1)  # 24*28*28
        out_x1 = self.maxpool(out_x1)  # 24*14*14
        out_x1 = self.Inception1_2(out_x1)  # 24*14*14
        out_x1 = self.maxpool(out_x1)  # 48*7*7
        out_x1 = self.cbam(out_x1)  # 48*7*7
        # out1:torch.Size([x, 48, 7, 7])
        # print(f"out1:{out_x1.shape}")

        out_x2 = self.image_CNN(x2).squeeze(dim=1)
        # out2:torch.Size([6, 1, 256])
        # print(f"out2:{out_x2.shape}")
        # out_x2 = self.image_RNN(out_x2)

        x = torch.cat((out_x1, out_x2), 1)
        # x = self.conv(out_x)
        x = x.reshape(x.shape[0], -1)  # flatten 变成全连接层的输入

        x1 = out_x1.reshape(x1.shape[0], -1)  # 对比需要展平
        x2 = out_x2.reshape(x2.shape[0], -1)  # 若有对比损失，需要展平x1，x2，并return
        # print(x.size())
        x = self.fc(x)

        # return x
        return x1, x2, x
