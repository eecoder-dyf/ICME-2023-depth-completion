import torch
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F
import copy

class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.Conv_BN_ReLU_2=nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.downsample=nn.Sequential(
            nn.Conv2d(in_channels=out_ch,out_channels=out_ch,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        """
        :param x:
        :return: out输出到深层, out_2输入到下一层
        """
        # print(self.Conv_BN_ReLU_2[0].weight)
        # for i in range(len(self.Conv_BN_ReLU_2)):
        #     if isinstance(self.Conv_BN_ReLU_2[i], nn.Conv2d):
        #         print(self.Conv_BN_ReLU_2[i].weight.device)
        # raise ValueError('stop')
        out=self.Conv_BN_ReLU_2(x)
        out_2=self.downsample(out)
        return out,out_2

class CrossScaleAttention(nn.Module):
    def __init__(self, in_ch_HR, in_ch_LR):
        super().__init__()
        self.in_ch_HR = in_ch_HR
        self.in_ch_LR = in_ch_LR
        self.conv_HR = nn.Sequential(
                            nn.Conv2d(in_channels=in_ch_HR, out_channels=in_ch_HR, kernel_size=1),
                            nn.BatchNorm2d(in_ch_HR)
        )
        self.conv_LR_k = nn.Sequential(
            nn.Conv2d(in_channels=in_ch_LR, out_channels=in_ch_HR, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_ch_HR)
        )
        self.conv_LR_v = nn.Sequential(
            nn.Conv2d(in_channels=in_ch_LR, out_channels=in_ch_HR, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_ch_HR)
        )

    def forward(self, x, y):
        '''
        :param x: input high resolution feature map B*C*H*W
        :param y: input low resolution feature map  B*2C*H/2*W/2
        :output x : x + attn(x, y), B*C*H*W
        '''
        
        B, Cx, Hx, Wx = x.shape
        # print(x.shape, y.shape, self.in_ch_HR)
        assert Cx == self.in_ch_HR, "channel size not match between input x and expected in_ch_HR"
        B, Cy, Hy, Wy = y.shape
        assert Cy == self.in_ch_LR, "channel size not match between input x and expected in_ch_HR"

        Q = self.conv_HR(x)     # output:B*Cx*Hx*Wx
        K = self.conv_LR_k(y)   # output:B*Cx*Hy*Wy
        V = self.conv_LR_v(y)   # output:B*Cx*Hy*Wy
        Q = Q.view(B, Cx, Hx*Wx).permute(0, 2, 1)   # B*(Hx*Wx)*C
        K = K.view(B, Cx, Hy*Wy)                    # B*C*(Hy*Wy)
        V = V.view(B, Cx, Hy*Wy).permute(0, 2, 1)   # B*(Hy*Wy)*C
        qk = torch.bmm(Q, K)   # B*(Hx*Wx)*(Hy*Wy)
        qk = F.softmax(qk, dim=-1)
        
        attn = torch.bmm(qk, V) # B*(Hx*Wx)*C
        attn = attn.view(B, Hx, Wx, Cx).permute(0, 3, 1, 2)
        x = x + attn

        return x

class Decoder(nn.Module):
    def __init__(self, in_ch_LR, in_ch_HR, attention=False, Spade=False):
        super().__init__()
        self.in_ch_LR = in_ch_LR
        self.in_ch_HR = in_ch_HR
        self.Spade = Spade
        self.attention = attention

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_ch_LR, out_channels=in_ch_HR, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(in_ch_HR),
            nn.ReLU(inplace=True)
        )
        # self.upsample = F.interpolate

        if attention == True:
            self.cross_scale_attention = CrossScaleAttention(in_ch_HR=in_ch_HR, in_ch_LR=in_ch_LR)

        if self.Spade == False:
            if attention == True:
                self.Conv_BN_ReLU_2 = nn.Sequential(
                    nn.Conv2d(in_channels=in_ch_HR*2, out_channels=in_ch_HR, kernel_size=3, padding=1),
                    nn.BatchNorm2d(in_ch_HR),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=in_ch_HR, out_channels=in_ch_HR, kernel_size=3, padding=1),
                    nn.BatchNorm2d(in_ch_HR),
                    nn.ReLU(inplace=True)
                )
            else:
                self.Conv_BN_ReLU_2 = nn.Sequential(
                    nn.Conv2d(in_channels=in_ch_HR, out_channels=in_ch_HR, kernel_size=3, padding=1),
                    nn.BatchNorm2d(in_ch_HR),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=in_ch_HR, out_channels=in_ch_HR, kernel_size=3, padding=1),
                    nn.BatchNorm2d(in_ch_HR),
                    nn.ReLU(inplace=True)
                )
        
    def forward(self, x, y):
        '''
        :param x: input high resolution feature map skip-connected from the encoder B*C*H*W
        :param y: input low resolution feature map  B*2C*H/2*W/2
        :output : y, B*C*H*W
        '''
        if self.attention == True:
            x = self.cross_scale_attention(x, y)
            y = self.upsample(y)
            y = torch.cat((x,y), dim=1)
            y = self.Conv_BN_ReLU_2(y)
        else:
            y = self.upsample(y)
            identity = y
            y = self.Conv_BN_ReLU_2(y)
            y = y + identity

        return y 

class SelfCompletion(nn.Module):
    def __init__(self):
        super().__init__()
        down_channels = [16, 32, 64, 128, 256, 512]
        
        self.enc_layers = nn.ModuleList()
        
        self.enc_layer0 = Encoder(in_ch=1, out_ch=down_channels[0])

        for i in range(len(down_channels)-1):
            e_layer = Encoder(in_ch=down_channels[i], out_ch=down_channels[i+1])
            self.enc_layers.append(e_layer)
        
        self.mid_layer = nn.Sequential(
            nn.Conv2d(in_channels=down_channels[-1], out_channels=down_channels[-1], kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(down_channels[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=down_channels[-1], out_channels=down_channels[-1], kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(down_channels[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=down_channels[-1], out_channels=down_channels[-1], kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(down_channels[-1]),
            nn.ReLU(inplace=True)           
        )

        self.dec_layers = nn.ModuleList()
        for i in range(len(down_channels)-1):
            if i < len(down_channels)-3:
                d_layer = Decoder(in_ch_LR=down_channels[len(down_channels)-1-i], in_ch_HR=down_channels[(len(down_channels)-2-i)], attention=True)
            else:
                d_layer = Decoder(in_ch_LR=down_channels[len(down_channels)-1-i], in_ch_HR=down_channels[(len(down_channels)-2-i)])
            self.dec_layers.append(d_layer)

        assert len(self.dec_layers) == len(self.enc_layers), "error, 编码器解码器深度不相等，网络不对称"
        self.layer_depths = len(self.dec_layers)
 
        self.dec_layer0 = nn.Sequential(
            nn.Conv2d(in_channels=down_channels[0], out_channels=down_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(down_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=down_channels[0], out_channels=down_channels[0]//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(down_channels[0]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=down_channels[0]//2, out_channels=1, kernel_size=3, stride=1, padding=1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, depth):
        encoder_id = []
        enc0, depth = self.enc_layer0(depth)

        encoder_id.append(enc0)

        for i in range(self.layer_depths):
            if i < len(self.enc_layers) -1 :
                enc, depth = self.enc_layers[i](depth)
                encoder_id.append(enc)
            else:
                depth, _ = self.enc_layers[i](depth)

        depth = self.mid_layer(depth)

        for i in range(len(self.dec_layers)):
            depth = self.dec_layers[i](encoder_id[self.layer_depths-1-i], depth)
        
        depth = self.dec_layer0(depth)
        return depth

            

        