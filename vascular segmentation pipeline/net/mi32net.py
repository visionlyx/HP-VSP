import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import numpy as np
from net.attention_block import  *



class Conv3d_BN_ReLU(nn.Module):
    def __init__(self,in_channels,out_channels, kernel_size,stride=1,padding=0,groups=1):
        super(Conv3d_BN_ReLU, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels,out_channels,kernel_size,stride,padding=padding,groups=groups),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.layer(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        """
        瓶颈结构的残差块
        conv1x1
        gconv3x3
        conv1x1
        :param in_channels:
        """
        super(ResidualBlock, self).__init__()
        self.layer = nn.Sequential(
            Conv3d_BN_ReLU(in_channels, in_channels//2, 1,1),
            Conv3d_BN_ReLU(in_channels//2, in_channels//2, 3, 1 ,padding=1, groups=2),
            nn.Conv3d(in_channels//2, in_channels, 1, 1),
            nn.BatchNorm3d(in_channels)
        )

        self.active = nn.ReLU(True)

    def forward(self,x):# 残差是两者相加的结果再激活，而不应该是激活后再相加
        return self.active(x + self.layer(x))




class BottleBlock(nn.Module):
    def __init__(self, in_channels):
        """
        瓶颈结构的残差块
        conv1x1
        gconv3x3
        conv1x1
        :param in_channels:
        """
        super(BottleBlock, self).__init__()
        self.layer = nn.Sequential(
            Conv3d_BN_ReLU(in_channels, in_channels//2, 1,1),
            Conv3d_BN_ReLU(in_channels//2, in_channels//2, 3, 1 ,padding=1, groups=2),
            nn.Conv3d(in_channels//2, in_channels, 1, 1),
            nn.BatchNorm3d(in_channels)
        )

        self.active = nn.ReLU(True)

    def forward(self,x):# 残差是两者相加的结果再激活，而不应该是激活后再相加
        return self.active(self.layer(x))



class BottleBlock_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        瓶颈结构的残差块
        conv1x1
        gconv3x3
        conv1x1
        :param in_channels:
        """
        super(BottleBlock_conv, self).__init__()
        self.layer = nn.Sequential(
            Conv3d_BN_ReLU(in_channels, out_channels//2, 1,1),
            Conv3d_BN_ReLU(out_channels//2, out_channels//2, 3, 1 ,padding=1, groups=2),
            nn.Conv3d(out_channels//2, out_channels, 1, 1),
            nn.BatchNorm3d(out_channels)
        )

        self.active = nn.ReLU(True)

    def forward(self,x):# 残差是两者相加的结果再激活，而不应该是激活后再相加
        return self.active(self.layer(x))






class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes[0], kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv3d(planes[0], planes[1], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu2(out)
        return out



class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.Dropout(0.1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)



class MiniLink(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(MiniLink,self).__init__()
        self.conv = nn.Sequential(

            nn.AvgPool3d(2),
            nn.Conv3d(in_ch,out_ch,3,padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose3d(in_ch,out_ch,2,stride=2)
            nn.Upsample(mode="trilinear", scale_factor=2, align_corners=False),
        )
        # self.active = nn.AvgPool3d(2)

    def forward(self, x):
        xx = self.conv(x)
        return (x+xx)


class MiniLink_new(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(MiniLink_new,self).__init__()
        self.conv = nn.Sequential(

            nn.MaxPool3d(2),
            Conv3d_BN_ReLU(in_ch, out_ch // 2, 1, 1),
            Conv3d_BN_ReLU(out_ch // 2, out_ch // 2, 3, 1, padding=1, groups=2),
            nn.Conv3d(out_ch // 2, out_ch, 1, 1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
            # nn.ConvTranspose3d(in_ch,out_ch,2,stride=2)
            nn.Upsample(mode="trilinear", scale_factor=2, align_corners=False),
        )
        # self.active = nn.AvgPool3d(2)

    def forward(self, x):
        xx = self.conv(x)
        return (x+xx)



class MINet32_new_new_new(nn.Module):
    """INPUT - 3DCONV - 3DCONV - 3DCONV - 3DCONV - FCN """

    def __init__(self, nchannels=1, nlabels=1, dim=3, batchnorm=True, dropout=False):
        """
        Builds the network structure with the provided parameters

        Input:
        - nchannels (int): number of input channels to the network
        - nlabels (int): number of labels to be predicted
        - dim (int): dimension of the network
        - batchnorm (boolean): sets if network should have batchnorm layers
        - dropout (boolean): set if network should have dropout layers
        """
        super().__init__()

        self.nchannels = nchannels
        self.nlabels = nlabels
        self.dims = dim
        self.batchnorm = batchnorm
        self.dropout = dropout
        # 3D Convolutional layers
        self.pool1 = nn.AvgPool3d(2)
        self.pool2 = nn.AvgPool3d(4)

        self.link1 = MiniLink(32, 32)
        self.link2 = MiniLink(32, 32)
        self.link3 = MiniLink(32, 32)

        self.conv1 = SingleConv(nchannels, 32)
        self.conv2 = BottleBlock(32)
        self.conv22 = BottleBlock(32)
        self.conv222 = BottleBlock(32)
        #self.conv2222 = ResidualBlock(32)

        self.conv3 = SingleConv(nchannels, 32)
        self.conv4 = BottleBlock(32)
        self.conv44 = BottleBlock(32)
        self.conv444 = BottleBlock(32)
        #self.conv4444 = ResidualBlock(32)

        self.conv5 = SingleConv(nchannels, 32)
        self.conv6 = BottleBlock(32)
        self.conv66 = BottleBlock(32)
        #self.conv666 = ResidualBlock(32)
        #self.conv6666 = ResidualBlock(32)


        self.up1 = nn.ConvTranspose3d(32, 32, 2, stride=2)
        self.up2 = nn.ConvTranspose3d(32, 32, 2, stride=2)
        self.conv7 = BottleBlock_conv(32, 16)
        self.up3 = nn.ConvTranspose3d(16, 16, 2, stride=2)

        self.cbam = CBAMBlock(channel=80, reduction=16, kernel_size=7)
        self.conv8 = BottleBlock_conv(80, 40)

        #self.conv88 = BottleBlock_conv(40, 20)
        self.conv9 = nn.Conv3d(40, nlabels, 1)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        #x0 = x
        x1 = self.pool1(x)
        x2 = self.pool2(x)


        x = self.conv1(x)
        x = self.link1(x)

        x = self.conv2(x)
        x = self.conv22(x)
        x = self.conv222(x)
        #x0 = self.conv2222(x0)


        x1 = self.conv3(x1)
        x1 = self.link2(x1)

        x1 = self.conv4(x1)
        x1 = self.conv44(x1)
        x1 = self.conv444(x1)
        #x1 = self.conv4444(x1)


        x2 = self.conv5(x2)
        x2 = self.link3(x2)
        x2 = self.conv6(x2)
        x2 = self.conv66(x2)
        #x2 = self.conv666(x2)
        #x2 = self.conv6666(x2)

        x1 = self.up1(x1)

        x2 = self.up2(x2)
        x2 = self.conv7(x2)
        x2 = self.up3(x2)

        x = torch.cat([x, x1, x2], dim=1)

        x = self.cbam(x)
        x = self.conv8(x)
        #merge = self.conv88(merge)
        x = self.conv9(x)
        x = self.sig(x)

        return x






if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = MINet32_new2().to(device)
    img = np.ones((1, 1, 100, 100, 100))

    img = np.array(img, dtype=np.float32)

    images = Variable(torch.from_numpy(img).type(torch.FloatTensor)).cuda()

    r_image = net(images)
    print(r_image.size())
    r_image = r_image * 255
    r_image = r_image.cpu()
    r_image = r_image.squeeze(0)
    r_image = r_image.detach().numpy()
    r_image = np.array(r_image, dtype=np.uint8)
