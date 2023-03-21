# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/11/25 20:11
'''
    Name:Time-window extension generative adversarial network(TEGAN)——Version 20
    Function:
    1.t -> 2t
    2.Improved TEGAN——Version 15
    3.Add SN both in generator and discriminator
    4.**Remarkable progress in 12-class dataset(1s-2S)**
'''
import torch
import math
import torch.nn.functional as F
from torch import nn
from Utils import Constraint
import Model.DL_Module as DLM


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, is_generator=True):
        super(LSTM, self).__init__()
        self.is_generator = is_generator
        self.rnn = nn.LSTM(input_size=input_size, batch_first=True, hidden_size=hidden_size,
                           bidirectional=True, num_layers=1)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        b, c, T = x.size()
        x = x.view(-1, T, c)  # (b, c, T) -> (b, T, c)
        r_out, _ = self.rnn(x)  # r_out shape [batch_size, time_step * 2, output_size]
        if self.is_generator:
            out = r_out.reshape(-1, 1, 2 * T * c)
            out = self.pool(out)
        else:
            out = r_out.reshape(-1, 1, 2 * T * c)
        return out


# Discriminator
class Discriminator(nn.Module):
    def calculateOutSize(self, model, nChan, nTime):
        '''
            Calculate the output based on input size
            model is from nn.Module and inputSize is a array
        '''
        data = torch.randn(1, 1, nChan, nTime)
        out = model(data).shape
        return out[1:]

    def spatial_block(self, in_channels, out_channels, dropout_level, kernel_size, stride):
        '''
           Spatial filter block,assign different weight to different channels and fuse them
        '''
        block = []
        block.append(Constraint.Conv2dWithConstraint(in_channels=in_channels, out_channels=out_channels,
                                                     kernel_size=(kernel_size, 1),
                                                     stride=(stride, 1), max_norm=1.0))
        block.append(nn.BatchNorm2d(num_features=out_channels))
        block.append(nn.PReLU())
        block.append(nn.Dropout(dropout_level))
        layer = nn.Sequential(*block)
        return layer

    def enhanced_block(self, in_channels, out_channels, dropout_level, kernel_size, stride):
        '''
           Enhanced structure block,build a CNN block to absorb data and output its stable feature
        '''
        block = []
        block.append(DLM.snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                                  stride=(1, stride)))
        block.append(nn.BatchNorm2d(num_features=out_channels))
        block.append(nn.MaxPool2d(kernel_size=(1, self.factor), stride=(1, self.factor)))
        block.append(nn.PReLU())
        block.append(nn.Dropout(dropout_level))
        layer = nn.Sequential(*block)
        return layer

    def __init__(self, Nc, Nt, Nf, ws, factor, pretrain=False):
        super(Discriminator, self).__init__()
        self.Nc = Nc
        self.Nt = Nt * factor
        self.Nf = Nf
        self.dropout_level = 0.5
        self.factor = factor
        self.ws = ws
        self.K = 10
        self.S = 2

        self.space = self.spatial_block(1, 2 * self.Nc, self.dropout_level, kernel_size=self.Nc, stride=self.Nc)

        # (batch_size, 2 * Nc, 1, Nt)
        self.time = self.enhanced_block(2 * self.Nc, 4 * self.Nc, self.dropout_level, kernel_size=self.K, stride=self.S)

        # (batch_size, 4 * Nc, 1, ((Nt - 10) / 2 + 1) / factor))
        self.rnn = LSTM(input_size=self.Nc * 4, hidden_size=self.Nc * 4, is_generator=False)

        conv_layers = nn.Sequential(self.space, self.time)

        self.fcSize = self.calculateOutSize(conv_layers, self.Nc, self.Nt)
        self.fcUnit = self.fcSize[0] * self.fcSize[1] * self.fcSize[2] * 2

        self.D = self.fcUnit // 20

        if pretrain:
            for p in self.parameters():
                p.requires_grad = False


        self.dense = nn.Sequential(
            nn.Flatten(),
            DLM.snlinear(self.fcUnit, self.D),
            nn.PReLU(),
        )

        self.adv = nn.Sequential(
            DLM.snlinear(self.D, 1),
        )


    def forward(self, x):
        x = self.space(x)
        out = self.time(x)
        out = out.squeeze(2)
        r_out = self.rnn(out)
        d_out = self.dense(r_out)
        real_fake = self.adv(d_out)
        return real_fake


# Down stage block for generator
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dropout_level, spatial=False):
        super(DownBlock, self).__init__()
        if spatial:
            self.conv = Constraint.Conv2dWithConstraint(in_channels, out_channels, kernel_size=(kernel, 1),
                                                        stride=(stride, 1))
        else:
            self.conv = DLM.snconv2d(in_channels, out_channels, kernel_size=(1, kernel),
                                                        stride=(1, stride))

        self.bn = nn.BatchNorm2d(num_features=out_channels)

        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(dropout_level)

    def forward(self, X):
        X = self.conv(X)
        X = self.bn(X)
        X = self.activation(X)
        out = self.dropout(X)
        return out



# Up stage block for generator
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, final_channels, kernel, stride, num_classes, cBN=False,
                 skip_connection=True, spatial=False):
        super(UpBlock, self).__init__()
        self.cBN = cBN
        self.skip = True if skip_connection else False
        if spatial:
            self.conv1 = DLM.sndeconv2d(in_channels, out_channels, kernel_size=(kernel, 1),
                                                        stride=(stride, 1))
        else:
            self.conv1 = DLM.sndeconv2d(in_channels, out_channels, kernel_size=(1, kernel),
                                                        stride=(1, stride))

        if self.cBN:
            self.bn1 = Constraint.CategoricalConditionalBatchNorm2d(num_classes=num_classes, num_features=out_channels)
        else:
            self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        self.activation1 = nn.PReLU()

        if skip_connection:
            self.conv2 = DLM.sndeconv2d(out_channels, final_channels, kernel_size=(1, 1), stride=(1, 1))
            if self.cBN:
                self.bn2 = Constraint.CategoricalConditionalBatchNorm2d(num_classes=num_classes,
                                                                        num_features=final_channels)
            else:
                self.bn2 = nn.BatchNorm2d(num_features=final_channels)
            self.activation2 = nn.PReLU()

    def forward(self, X, label):
        X = self.conv1(X)
        if self.cBN:
            X = self.bn1(X, label)
        else:
            X = self.bn1(X)
        out = self.activation1(X)

        if self.skip:
            out = self.conv2(out)
            if self.cBN:
                out = self.bn2(out, label)
            else:
                out = self.bn2(out)
            out = self.activation2(out)
        return out


# Generator
class Generator(nn.Module):
    def __init__(self, Nc, Nt, Nf, ws, factor):
        super(Generator, self).__init__()
        self.Nc = Nc
        self.Nt1 = Nt
        self.Nt2 = Nt * factor
        self.Nf = Nf
        self.dropout_level = 0.5
        self.factor = factor
        self.ws = ws

        '''Method1:Self-definition'''
        self.down_k = [20, 12, 8]
        for i in range(len(self.down_k)):
            self.down_k[i] = math.ceil(self.down_k[i] * self.ws)

        # '''Method2:Self-Adaptive'''
        # self.down_k = [0, 0, 0]
        # for i in range(len(self.down_k)):
        #    if i == 0:
        #        self.down_k[i] = self.Nt2 // 10
        #    else:
        #        self.down_k[i] = self.Nt1 // 10


        self.up_k = [0, 0, 0]

        self.S = [2, 2, 1]
                                                                      # 4-class  40-class
        self.DF1 = ((self.Nt1 - self.down_k[0]) // self.S[0] + 1)      # 45       120
        self.DF2 = ((self.DF1 - self.down_k[1]) // self.S[1] + 1)      # 18        56
        self.DF3 = ((self.DF2 - self.down_k[2]) // self.S[2] + 1)      # 11        49

        self.up_k[0] = self.DF2 - (self.DF3 - 1) * self.S[2]
        self.UF1 = (self.DF3 - 1) * self.S[2] + self.up_k[0]           # 18        58
        self.up_k[1] = self.DF1 - (self.DF2 - 1) * self.S[1]
        self.UF2 = (self.UF1 - 1) * self.S[1] + self.up_k[1]           # 45        121
        self.up_k[2] = self.Nt1 - (self.DF1 - 1) * self.S[0]
        self.UF3 = (self.UF2 - 1) * self.S[0] + self.up_k[2]           # 100       250


        # self.X_embed = DLM.sn_embedding(num_embeddings=self.Nf, embedding_dim=self.DF3)

        # (batch_size, 1, Nc, Nt1)
        self.down_stage1 = DownBlock(in_channels=1, out_channels=2 * self.Nc, kernel=self.Nc, stride=self.Nc,
                                     dropout_level=self.dropout_level,  spatial=True)

        # (batch_size, 2 * Nc, 1, Nt1)
        self.down_stage2 = DownBlock(in_channels=2 * self.Nc, out_channels=4 * self.Nc, kernel=self.down_k[0],
                                     stride=self.S[0], dropout_level=self.dropout_level)

        # (batch_size, 4 * Nc, 1, DF1)
        self.down_stage3 = DownBlock(in_channels=4 * self.Nc, out_channels=8 * self.Nc, kernel=self.down_k[1],
                                     stride=self.S[1], dropout_level=self.dropout_level)

        # (batch_size, 8 * Nc, 1, DF2)
        self.down_stage4 = DownBlock(in_channels=8 * self.Nc, out_channels=16 * self.Nc, kernel=self.down_k[2],
                                     stride=self.S[2], dropout_level=self.dropout_level)

        # (batch_size, 16 * Nc, 1, DF3)
        self.rnn = LSTM(input_size=self.Nc * 16, hidden_size=self.Nc * 16, is_generator=True)


        self.D = self.Nc * 16 * self.DF3

        self.fc = nn.Sequential(nn.Flatten(),
                                DLM.snlinear(self.D, self.Nf))


        # (batch_size, 16 * Nc, 1, DF3)
        self.up_stage1 = UpBlock(in_channels=16 * self.Nc, out_channels=8 * self.Nc, final_channels=8 * self.Nc,
                                 kernel=self.up_k[0], stride=self.S[2], num_classes=self.Nf, cBN=True, skip_connection=False)

        # (batch_size, 8 * Nc, 1, UF1)
        self.up_stage2 = UpBlock(in_channels=16 * self.Nc, out_channels=8 * self.Nc, final_channels=4 * self.Nc,
                                 kernel=self.up_k[1], stride=self.S[1], num_classes=self.Nf, cBN=True, skip_connection=True)

        # (batch_size, 8 * Nc, 1, UF2)
        self.up_stage3 = UpBlock(in_channels=8 * self.Nc, out_channels=4 * self.Nc, final_channels=2 * self.Nc,
                                 kernel=self.up_k[2], stride=self.S[0], num_classes=self.Nf, cBN=True, skip_connection=True)

        # (batch_size, 2 * Nc, 1, UF3)
        self.up_stage4 = UpBlock(in_channels=4 * self.Nc, out_channels=2 * self.Nc, final_channels=1 * self.Nc,
                                 kernel=self.Nc, stride=self.Nc, num_classes=self.Nf, cBN=True, skip_connection=True,
                                 spatial=True)

        # (batch_size, 2 * Nc, Nc, UF3)
        self.extent_block = UpBlock(in_channels=2 * self.Nc, out_channels=1 * self.Nc, final_channels=1,
                                    kernel=self.factor, stride=self.factor, num_classes=self.Nf, cBN=True,
                                    skip_connection=True)

        # (batch_size, 1, Nc, Nt2)


    def forward(self, X):

        '''Down sample stage'''
        D1 = self.down_stage1(X)
        D2 = self.down_stage2(D1)
        D3 = self.down_stage3(D2)
        D4 = self.down_stage4(D3)

        '''LSTM stage'''
        r_input = D4.squeeze(2)
        r_out = self.rnn(r_input)
        b, c, _, T = D4.size()
        r_out = r_out.view(-1, c, 1, T)

        '''FC stage'''
        fc_out = self.fc(r_out)
        label = fc_out.argmax(dim=1)

        '''Up sample stage'''
        skip_X = X.repeat(1, self.Nc, 1, 1)

        # Data-Label Fusion
        # up_label = self.X_embed(label).reshape(-1, 1, 1, T)
        # up_label = up_label.repeat(1, c, 1, 1)
        # r_out = r_out * up_label

        r_out_up = r_out
        # r_out_up = F.avg_pool2d(r_out, kernel_size=(1, 2), stride=(1, 2))
        U1 = self.up_stage1(r_out_up, label)
        U2 = self.up_stage2(torch.cat([D3, U1], dim=1), label)
        U3 = self.up_stage3(torch.cat([D2, U2], dim=1), label)
        U4 = self.up_stage4(torch.cat([D1, U3], dim=1), label)
        out = self.extent_block(torch.cat([skip_X, U4], dim=1), label)
        # print("out.shape:", out.shape)
        return out, fc_out