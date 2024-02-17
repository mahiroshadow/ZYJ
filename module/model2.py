import torch
import torch.nn as nn
import torch.nn.functional as F

class Inception(nn.Module):
    def __init__(self, in_channels=3, out_channels=6,mode='video'):
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
        self.maxpool = nn.MaxPool2d(2)
        self.mode=mode

    def forward(self, x_2d):
        if self.mode =='video':
            image_emb_seq = []
            for t in range(x_2d.shape[1]):
                x = x_2d[:, t, :, :, :]
                branch1x1 = self.branch1x1(x)
                branch3x3 = self.branch3x3(x)
                branch5x5 = self.branch5x5(x)
                branch_pool = self.branch_pool(x)
                outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
                outputs = torch.cat(outputs, dim=1)
                outputs=self.maxpool(outputs)
                # outputs=outputs.reshape(outputs.shape[0],-1)
                image_emb_seq.append(outputs)
            out = torch.stack(image_emb_seq, dim=0).transpose_(0, 1)
        else:
            branch1x1 = self.branch1x1(x_2d)
            branch3x3 = self.branch3x3(x_2d)
            branch5x5 = self.branch5x5(x_2d)
            branch_pool = self.branch_pool(x_2d)
            outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
            # 4个分支在dim=1即沿着channel(张量是0batch,1channel,2weight,3height)上进行concatenate。6+6+6+6=24(输出通道数)
            out=torch.cat(outputs, dim=1)
        return out

class IRNN(nn.Module):
    def __init__(self, h_RNN_layers, h_RNN, h_FC_dim,drop_p):
        super(IRNN,self).__init__()
        self.Inception1_1 = Inception(in_channels=3, out_channels=6,mode='video')
        self.Inception1_2 = Inception(in_channels=24, out_channels=12,mode='video')
        self.Inception2_1 = Inception(in_channels=3, out_channels=6,mode='image')
        self.Inception2_2 = Inception(in_channels=24, out_channels=12,mode='image')
        self.rnn = RNN(48*7*7, h_RNN_layers, 16*7*7, h_FC_dim, drop_p)

    def forward(self,x1,x2):
        '''
        x1 可以用视频
        '''
        out_x1 = self.Inception1_1(x1)  # batch_size*frame*24*14*14
        out_x1 = self.Inception1_2(out_x1)  # batch_size*frame*48*7*7
        out_x1 =out_x1.reshape(out_x1.shape[0],out_x1.shape[1],-1)
        out_x1=self.rnn(out_x1)
        out_x2=self.Inception2_1(x2)
        out_x2=self.Inception2_2(out_x2)
        return out_x1,out_x2


class RNN(nn.Module):
    def __init__(self, CNN_embed_dim, h_RNN_layers, h_RNN, h_FC_dim, drop_p):
        super().__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layers
        self.h_RNN = h_RNN  # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True
        )

        self.fc1 = nn.Linear(self.h_RNN, self.CNN_embed_dim)

    def forward(self, x_RNN):
        rnn_out, (_, _) = self.LSTM(x_RNN, None)

        # FC layers
        x = self.fc1(rnn_out[:, -1, :])  # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        return x