import os, torch
import numpy as np
#import pynvml
from torch import nn
from torch import functional as F
import tifffile
from tqdm import tqdm

#skimage.measure.compare_psnr(im_true, im_test, data_range=None, dynamic_range=None)
#skimage.measure.compare_ssim(


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        #pt = torch.sigmoid(_input)
        pt = _input
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1-self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        # if self.alpha:
        #     loss = loss * self.alpha
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2, weight=None, ignore_index=255):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.weight = weight
#         self.ignore_index = ignore_index
#         #self.bce_fn = nn.BCELoss(weight=self.weight)
#
#     def forward(self, preds, labels):
#         #logpt = -self.bce_fn(preds, labels)
#         pt = torch.exp(logpt)
#         loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
#         return loss


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, probs, targets):#logits,
        num = targets.size(0)
        smooth = 1

        #probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

# class SoftDiceLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(SoftDiceLoss, self).__init__()
#
#     def make_one_hot(self,input, num_classes):
#         """Convert class index tensor to one hot encoding tensor.
#         Args:
#              input: A tensor of shape [N, 1, *]
#              num_classes: An int of number of class
#         Returns:
#             A tensor of shape [N, num_classes, *]
#         """
#         shape = np.array(input.shape)
#         shape[1] = num_classes
#         shape = tuple(shape)
#         result = torch.zeros(shape)
#         result = result.scatter_(1, input.cpu(), 1)
#         result = result.cuda(1)
#         return result
#
#     def forward(self, probs, targets):#logits,
#         targets = self.make_one_hot(targets,2)
#         num = targets.size(0)
#         smooth = 1
#
#         #probs = F.sigmoid(logits)
#         m1 = probs.view(num, -1)
#         m2 = targets.view(num, -1)
#         intersection = (m1 * m2)
#
#         score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
#         score = 1 - score.sum() / num
#         return score
#
# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2, alpha=0.5,batch_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.batch_average = batch_average
#
#     def forward(self, logit, target):
#         n, c, d, h, w = logit.size()
#         criterion = nn.BCEWithLogitsLoss()
#         #criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
#                                         #size_average=self.size_average)
#         # if self.cuda:
#         #     criterion = criterion.cuda()
#
#         logpt = -criterion(logit, target.float())
#         pt = torch.exp(logpt)
#         if self.alpha is not None:
#             logpt *= self.alpha
#         loss = -((1 - pt) ** self.gamma) * logpt
#
#         if self.batch_average:
#             loss /= n
#
#         return loss


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias)


def default_conv3d(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias)


def prepare(dev, *args):
    # print(dev)
    device = torch.device(dev)
    if dev == 'cpu':
        device = torch.device('cpu')
    return [a.to(device) for a in args]


def calc_psnr(sr, hr, scale):
    diff = (sr - hr)
    # shave = scale + 6
    # valid = diff[..., shave:-shave, shave:-shave,:]#2，2，1
    # mse = valid.pow(2).mean()
    mse = np.mean(diff * diff) + 0.0001
    return -10 * np.log10(mse / (4095 ** 2))


def RestoreNetImg(img, mean, max):
    # rImg = (img - self.mean1) / self.std1
    #rImg = np.maximum(np.minimum(img * max + mean, 255), 0)
    rImg = img * max + mean
    maxVal = np.max(rImg)
    minVal = np.min(rImg)
    if maxVal <= minVal:
        rImg *= 0
    else:
        rImg = 255./(maxVal - minVal) * (rImg - minVal)
        rImg = np.maximum(np.minimum(rImg, 255), 0)
    return rImg

def RestoreNetImgV2(img, mean, max):
    # rImg = (img - self.mean1) / self.std1
    #rImg = np.maximum(np.minimum(img * max + mean, 255), 0)
    rImg = img * max + mean
    rImg = np.maximum(np.minimum(rImg, 255), 0)
    return rImg

class WDSRBBlock3D(nn.Module):
    def __init__(
        self, n_feats, kernel_size, wn, act=nn.ReLU(True), res_scale=1):
        super(WDSRBBlock3D, self).__init__()
        self.res_scale = res_scale
        body = []
        expand = 6
        linear = 0.8
        body.append(
            wn(nn.Conv3d(n_feats, n_feats*expand, 1, padding=1//2)))
        body.append(act)
        body.append(
            wn(nn.Conv3d(n_feats*expand, int(n_feats*linear), 1, padding=1//2)))
        body.append(
            wn(nn.Conv3d(int(n_feats*linear), n_feats, kernel_size, padding=kernel_size//2)))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        # res = self.body(x) * self.res_scale
        # res += x
        res = self.body(x) + x
        return res

class ResBlock3D(nn.Module):
    def __init__(self,
                 conv=default_conv3d,
                 n_feats=64,
                 kernel_size=3,
                 bias=True,
                 bn=False,
                 act=nn.ReLU(inplace=True),  # nn.LeakyReLU(inplace=True),
                 res_scale=1):

        super(ResBlock3D, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm3d(n_feats))
            m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class ConvLayer(nn.Module):
    def __init__(self,
                 inplane = 64,
                 n_feats=32,
                 stride = 1,
                 kernel_size=3,
                 bias=True,
                 bn=nn.BatchNorm3d,
                 padding = 1,
                 act=nn.ReLU(inplace=True),  # nn.LeakyReLU(inplace=True),
                 res_scale=1):

        super(ConvLayer, self).__init__()
        m = []
        m.append(nn.Conv3d(inplane, n_feats,kernel_size = kernel_size,
                           stride = stride,padding = padding, bias=bias))
        if bn is not None:
            m.append(bn(n_feats))
        if act is not None:
            m.append(act)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        return res

class UpLayer(nn.Module):
    def __init__(self,
                 inplane = 64,
                 n_feats=32,
                 scale_factor=2,
                 bn = nn.BatchNorm3d,
                 act=nn.ReLU(inplace=True)  # nn.LeakyReLU(inplace=True),
                 ):

        super(UpLayer, self).__init__()
        m = []
        m.append(nn.Upsample(scale_factor=scale_factor,mode='trilinear'))

        m.append(nn.Conv3d(in_channels=inplane,out_channels = n_feats,
                           kernel_size=3,padding=3//2 ))
        if bn is not None:
            m.append(bn(n_feats))
        m.append(act)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        return res


class PixelUpsampler3D(nn.Module):
    def __init__(self,
                 upscale_factor,
                 # conv=default_conv3d,
                 # n_feats=32,
                 # kernel_size=3,
                 # bias=True
                 ):
        super(PixelUpsampler3D, self).__init__()
        self.scaleFactor = upscale_factor

    def _pixel_shuffle(self, input, upscale_factor):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        channels //= upscale_factor[0] * upscale_factor[1] * upscale_factor[2]
        out_depth = in_depth * upscale_factor[0]
        out_height = in_height * upscale_factor[1]
        out_width = in_width * upscale_factor[2]
        input_view = input.contiguous().view(
            batch_size, channels, upscale_factor[0], upscale_factor[1], upscale_factor[2], in_depth,
            in_height, in_width)
        shuffle_out = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return shuffle_out.view(batch_size, channels, out_depth, out_height, out_width)

    def forward(self, x):
        # x = self.conv(x)
        up = self._pixel_shuffle(x, self.scaleFactor)
        return up




class GetTrainDataSet2():
    def __init__(self, lrDir, hrDir ):
        self.lrDir = lrDir
        self.hrDir = hrDir
        self.lrFileList = []
        self.hrFileList = []
        for file in os.listdir(self.lrDir):
            if file.endswith('.tif'):
                self.lrFileList.append(file)

        for file in os.listdir(self.hrDir):
            if file.endswith('.tif'):
                self.hrFileList.append(file)

        if len(self.lrFileList) != len(self.hrFileList):
            self.check = False

    def Check(self):
        return self.check

    def DataNum(self):
        return len(self.hrFileList)

    def __len__(self):
        return len(self.hrFileList)

    def __getitem__(self, ind):
        # load the image and labels
        imgName = self.hrFileList[ind]
        lrName = os.path.join(self.lrDir, imgName)
        hrName = os.path.join(self.hrDir, imgName)

        lrImg = tifffile.imread(lrName)
        hrImg = tifffile.imread(hrName)
        #lrImg = lrImg[:32,:32,:32]
        #hrImg = hrImg[:96, :96, :96]
        # print(lrImg.shape)
        # print(hrImg.shape)

        # randX = np.random.randint(0, 100-32 - 1)
        # randY = np.random.randint(0, 100 - 32 -1 )
        # #randZ = np.random.randint(0, 30 - 24 - 1)
        # lrImg = lrImg[:, randY:randY+32, randX:randX+32]#randZ:randZ+24
        # hrImg = hrImg[:,#randZ*3:randZ*3 + 72,
        #         randY*3:randY*3 + 96,
        #         randX*3:randX*3 + 96]
        # print(lrImg.shape)
        # print(hrImg.shape)
        lrImg = np.expand_dims(lrImg, axis=0)
        hrImg = np.expand_dims(hrImg, axis=0)

        lrImg = np.array(lrImg, dtype=np.float32)
        hrImg = np.array(hrImg, dtype=np.float32)

        lrImg = torch.from_numpy(lrImg).float()
        hrImg = torch.from_numpy(hrImg).float()
        return lrImg, hrImg


class GetTrainDataSet3():
    def __init__(self, lrDir, midDir, hrDir ):
        self.lrDir = lrDir
        self.midDir = midDir
        self.hrDir = hrDir
        self.lrFileList = []
        self.midFileList = []
        self.hrFileList = []
        for file in os.listdir(self.lrDir):
            if file.endswith('.tif'):
                self.lrFileList.append(file)

        for file in os.listdir(self.midDir):
            if file.endswith('.tif'):
                self.midFileList.append(file)

        for file in os.listdir(self.hrDir):
            if file.endswith('.tif'):
                self.hrFileList.append(file)

        if len(self.lrFileList) != len(self.hrFileList):
            self.check = False

    def Check(self):
        return self.check

    def DataNum(self):
        return len(self.hrFileList)

    def __len__(self):
        return len(self.hrFileList)

    def __getitem__(self, ind):
        # load the image and labels
        imgName = self.hrFileList[ind]
        lrName = os.path.join(self.lrDir, imgName)
        midName = os.path.join(self.midDir, imgName)
        hrName = os.path.join(self.hrDir, imgName)

        lrImg = tifffile.imread(lrName)
        midImg = tifffile.imread(midName)
        hrImg = tifffile.imread(hrName)

        midImg = midImg > 125
        hrImg = hrImg / 255.

        lrImg = np.expand_dims(lrImg, axis=0)
        midImg = np.expand_dims(midImg, axis=0)
        hrImg = np.expand_dims(hrImg, axis=0)

        lrImg = np.array(lrImg, dtype=np.float32)
        midImg = np.array(midImg, dtype=np.float32)
        hrImg = np.array(hrImg, dtype=np.float32)

        lrImg = torch.from_numpy(lrImg).float()
        midImg = torch.from_numpy(midImg).float()
        hrImg = torch.from_numpy(hrImg).float()
        return lrImg, midImg, hrImg



class GetTestDataSet():
    def __init__(self, testDir, mean, std):
        self.testDir = testDir
        self.testFileList = os.listdir(self.testDir)

        # self.mean1=np.array([160],dtype=np.float32)
        self.mean1 = mean  # np.array([127], dtype=np.float32)
        self.std1 = std  # np.array([350], dtype=np.float32)

    def Check(self):
        return self.check

    def DataNum(self):
        return len(self.testFileList)

    def __len__(self):
        return len(self.testFileList)

    def __getitem__(self, ind):
        # load the image and labels
        imgName = self.testFileList[ind]
        lrName = os.path.join(self.testDir, imgName)
        lrImg = tifffile.imread(lrName)

        lrImg = np.array(lrImg, dtype=np.float32)
        lrImg = (lrImg - self.mean1) / self.std1
        lrImg = np.expand_dims(lrImg, axis=0)
        # torch.set_grad_enabled(True)
        lrImg = torch.from_numpy(lrImg).float()
        return lrImg


class GetMemoryDataSet3:
    def __init__(self, lrDir, midDir, hrDir):
        self.lrDir = lrDir
        self.midDir = midDir
        self.hrDir = hrDir
        self.lrFileList = []
        self.midFileList = []
        self.hrFileList = []
        for file in os.listdir(self.lrDir):
            if file.endswith('.tif'):
                self.lrFileList.append(file)

        for file in os.listdir(self.midDir):
            if file.endswith('.tif'):
                self.midFileList.append(file)

        for file in os.listdir(self.hrDir):
            if file.endswith('.tif'):
                self.hrFileList.append(file)

        if len(self.lrFileList) != len(self.hrFileList):
            self.check = False

        self.lrImgList = []
        self.midImgList = []
        self.hrImgList = []

        for name in tqdm(self.lrFileList):
            lrName = os.path.join(self.lrDir,name)
            midName = os.path.join(self.midDir, name)
            hrName = os.path.join(self.hrDir, name)
            lrImg = tifffile.imread(lrName)
            midImg = tifffile.imread(midName)
            hrImg = tifffile.imread(hrName)

            lrImg = np.expand_dims(lrImg, axis=0)
            midImg = np.expand_dims(midImg, axis=0)
            hrImg = np.expand_dims(hrImg, axis=0)

            lrImg = lrImg.astype(np.float)
            #midImg = midImg.astype(np.float)
            #hrImg = hrImg.astype(np.float)

            midImg = midImg / 255
            hrImg = hrImg / 255

            self.lrImgList.append(lrImg)
            self.midImgList.append(midImg)
            self.hrImgList.append(hrImg)

    def Check(self):
        return self.check

    def DataNum(self):
        return len(self.hrFileList)

    def __len__(self):
        return len(self.hrFileList)

    def len(self):
        return len(self.hrFileList)

    def __getitem__(self, ind):
        # while torch.max(self.midImgList[ind]) < 0.8:
        #     prob = np.random.rand()
        #     if prob > 0.8:
        #         ind = np.random.randint(0, len(self.hrFileList)-1)
        rid = np.random.randint(0,6)
        if rid == 0:
            lrImg,midImg,hrImg = self.lrImgList[ind], self.midImgList[ind], self.hrImgList[ind]
        if rid == 1:
            lrImg,midImg,hrImg = self.lrImgList[ind][:,:,::-1,:], self.midImgList[ind][:,:,::-1,:], self.hrImgList[ind][:,:,::-1,:]
        if rid == 2:
            lrImg,midImg,hrImg =  self.lrImgList[ind][:,:,::-1,:], self.midImgList[ind][:,:,::-1,:], self.hrImgList[ind][:,:,::-1,:]
        if rid == 3:
            lrImg,midImg,hrImg =  self.lrImgList[ind][:,:,:,::-1], self.midImgList[ind][:,:,:,::-1], self.hrImgList[ind][:,:,:,::-1]
        if rid == 4:
            lrImg,midImg,hrImg =  self.lrImgList[ind][:,::-1,::-1,:], self.midImgList[ind][:,::-1,::-1,:], self.hrImgList[ind][:,::-1,::-1,:]
        if rid == 5:
            lrImg,midImg,hrImg =  self.lrImgList[ind][:,:,::-1,::-1], self.midImgList[ind][:,:,::-1,::-1], self.hrImgList[ind][:,:,::-1,::-1]

        lrImg = torch.from_numpy(lrImg.copy()).float()
        midImg = torch.from_numpy(midImg.copy()).float()
        hrImg = torch.from_numpy(hrImg.copy()).float()
        return lrImg, midImg, hrImg

        # load the image and labels
        # imgName = self.hrFileList[ind]
        # lrName = os.path.join(self.lrDir, imgName)
        # midName = os.path.join(self.midDir, imgName)
        # hrName = os.path.join(self.hrDir, imgName)
        # #print(imgName)
        #
        # lrImg = tifffile.imread(lrName)
        # midImg = tifffile.imread(midName)
        # hrImg = tifffile.imread(hrName)
        #
        # lrImg = np.expand_dims(lrImg, axis=0)
        # midImg = np.expand_dims(midImg, axis=0)
        # hrImg = np.expand_dims(hrImg, axis=0)
        #
        # lrImg = lrImg.astype(np.float)
        # midImg = midImg.astype(np.float)
        # hrImg = hrImg.astype(np.float)
        #
        # midImg = midImg /255.
        # hrImg = hrImg / 255.
        #
        # lrImg = torch.from_numpy(lrImg).float()
        # midImg = torch.from_numpy(midImg).float()
        # hrImg = torch.from_numpy(hrImg).float()
        # return lrImg, midImg, hrImg

class GetMemoryDataSetAndCrop:
    def __init__(self, lrDir, hrDir,lineDir, cropSize, epoch):
        self.lrDir = lrDir
        #self.midDir = midDir
        self.hrDir = hrDir
        self.lineDir = lineDir

        self.epoch = epoch

        self.lrFileList = []
        #self.midFileList = []
        self.hrFileList = []
        #self.lineFileList = []

        self.lrImgList = []
        #self.midImgList = []
        self.hrImgList = []
        self.lineImgList = []

        self.beg = [0, 0, 0]
        self.cropSz = cropSize

        for file in os.listdir(self.lrDir):
            if file.endswith('.tif'):
                self.lrFileList.append(file)

        # for file in os.listdir(self.midDir):
        #     if file.endswith('.tif'):
        #         self.midFileList.append(file)

        for file in os.listdir(self.hrDir):
            if file.endswith('.tif'):
                self.hrFileList.append(file)

        self.crossPlace = []
        if len(self.lineDir) > 0:
            for file in os.listdir(self.lineDir):
                if file.endswith('.swc'):
                    self.crossPlace.append(np.loadtxt(os.path.join(self.lineDir,file)))

        if len(self.lrFileList) != len(self.hrFileList):
            self.check = False

        for name in tqdm(self.lrFileList):
            lrName = os.path.join(self.lrDir,name)
            #midName = os.path.join(self.midDir, name)
            hrName = os.path.join(self.hrDir, name)
            #lineName = os.path.join(self.lineDir, name)
            lrImg = tifffile.imread(lrName)
            #midImg = tifffile.imread(midName)
            hrImg = tifffile.imread(hrName)
            #lineImg = tifffile.imread(lineName)

            lrImg = np.expand_dims(lrImg, axis=0)
            #midImg = np.expand_dims(midImg, axis=0)
            hrImg = np.expand_dims(hrImg, axis=0)
            #lineImg = np.expand_dims(lineImg, axis=0)

            #lrImg = lrImg.astype(np.float)
            #midImg = midImg.astype(np.float)
            #hrImg = hrImg.astype(np.float)

            #midImg = midImg / 255
            #hrImg = hrImg / 255

            self.lrImgList.append(lrImg)
            #self.midImgList.append(midImg)
            self.hrImgList.append(hrImg)
            #self.lineImgList.append(lineImg)

    def Check(self):
        return self.check

    def DataNum(self):
        return len(self.hrFileList)

    def __len__(self):
        return self.epoch#len(self.hrFileList)

    def len(self):
        return self.epoch#len(self.hrFileList)

    def __getitem__(self, ind):
        flag = True
        while flag:
            ind = np.random.randint(0,len(self.hrFileList)+len(self.crossPlace))
            #ind = np.random.randint(44, len(self.hrFileList))
            if ind < len(self.hrFileList):
                # if ind > 2 and ind < len(self.hrFileList)+8:
                #     ind = 3
                sz = self.lrImgList[ind].shape
                self.beg[0] = np.random.randint(0, sz[1] - self.cropSz[0] - 1)
                self.beg[1] = np.random.randint(0, sz[2] - self.cropSz[1] - 1)
                self.beg[2] = np.random.randint(0, sz[3] - self.cropSz[2] - 1)

                # lrImg = self.lrImgList[ind][:,self.beg[0]:self.beg[0] + self.cropSz[0],
                #           self.beg[1]:self.beg[1] + self.cropSz[1],
                #           self.beg[2]:self.beg[2] + self.cropSz[2]]
                #
                # hrImg = self.hrImgList[ind][:,self.beg[0]*4:self.beg[0]*4 + self.cropSz[0]*4,
                #         self.beg[1]*2:self.beg[1]*2 + self.cropSz[1]*2,
                #         self.beg[2]*2:self.beg[2]*2 + self.cropSz[2]*2]
            else:
                if ind < len(self.hrFileList)+len(self.crossPlace):
                    curInd = ind-len(self.hrFileList)
                else:
                    curInd = len(self.hrFileList)-1
                ind = curInd
            #     sz = self.lrImgList[curInd].shape
            #     placeInd = np.random.randint(0,len(self.crossPlace[curInd]))
            #     curPlace = (np.round(self.crossPlace[curInd][placeInd])).astype(np.int)
            #     xMin = np.minimum(np.maximum(0, curPlace[2]-12),sz[3] - self.cropSz[2] - 3)
            #     xMax = np.maximum(np.minimum(sz[3] - self.cropSz[2] - 1, curPlace[2] - 5),1)
            #     yMin = np.minimum(np.maximum(0, curPlace[3]-12),sz[2] - self.cropSz[1] - 3)
            #     yMax = np.maximum(np.minimum(sz[2] - self.cropSz[1] - 1, curPlace[3] - 5),1)
            #     zMin = np.minimum(np.maximum(0, curPlace[4]-8),sz[1] - self.cropSz[0] - 3)
            #     zMax = np.maximum(np.minimum(sz[1] - self.cropSz[0] - 1, curPlace[4] - 3),1)
            #     self.beg[0] = np.random.randint(zMin, zMax)
            #     self.beg[1] = np.random.randint(yMin, yMax)
            #     self.beg[2] = np.random.randint(xMin, xMax)
            #
            hrImg = self.hrImgList[ind][:, self.beg[0]:self.beg[0]  + self.cropSz[0] ,
                    self.beg[1] :self.beg[1]  + self.cropSz[1] ,
                    self.beg[2] :self.beg[2]  + self.cropSz[2] ]

            if np.sum(hrImg) < 2500:
                pass
            else:
                lrImg = self.lrImgList[ind][:, self.beg[0]:self.beg[0] + self.cropSz[0],
                        self.beg[1]:self.beg[1] + self.cropSz[1],
                        self.beg[2]:self.beg[2] + self.cropSz[2]]
                flag = False


        rid = np.random.randint(0,6)
        if rid == 0:
            pass#return lrImg, midImg, hrImg
        if rid == 1:
            lrImg,hrImg = lrImg[:,::-1,:,:], hrImg[:,::-1,:,:]
        if rid == 2:
            lrImg,hrImg =  lrImg[:,:,::-1,:], hrImg[:,:,::-1,:]
        if rid == 3:
            lrImg,hrImg =  lrImg[:,:,:,::-1], hrImg[:,:,:,::-1]
        if rid == 4:
            lrImg,hrImg = lrImg[:,::-1,::-1,:],  hrImg[:,::-1,::-1,:]
        if rid == 5:
            lrImg,hrImg =  lrImg[:,:,::-1,::-1], hrImg[:,:,::-1,::-1]

        lrImg = torch.from_numpy(lrImg.copy().astype(np.float)).float()
        #midImg = torch.from_numpy(midImg.copy().astype(np.float)).float()
        hrImg = torch.from_numpy(hrImg.copy().astype(np.float)).float()
        #midImg = midImg / 255
        hrImg = hrImg / 255
        return lrImg,  hrImg

class SRGetMemoryDataSetAndCrop:
    def __init__(self, lrDir, hrDir,lineDir, cropSize, epoch):
        self.lrDir = lrDir
        #self.midDir = midDir
        self.hrDir = hrDir
        self.lineDir = lineDir

        self.epoch = epoch

        self.lrFileList = []
        #self.midFileList = []
        self.hrFileList = []
        #self.lineFileList = []

        self.lrImgList = []
        #self.midImgList = []
        self.hrImgList = []
        self.lineImgList = []

        self.beg = [0, 0, 0]
        self.cropSz = cropSize

        for file in os.listdir(self.lrDir):
            if file.endswith('.tif'):
                self.lrFileList.append(file)

        # for file in os.listdir(self.midDir):
        #     if file.endswith('.tif'):
        #         self.midFileList.append(file)

        for file in os.listdir(self.hrDir):
            if file.endswith('.tif'):
                self.hrFileList.append(file)

        self.crossPlace = []
        if len(self.lineDir) > 0:
            for file in os.listdir(self.lineDir):
                if file.endswith('.swc'):
                    self.crossPlace.append(np.loadtxt(os.path.join(self.lineDir,file)))

        if len(self.lrFileList) != len(self.hrFileList):
            self.check = False

        for name in tqdm(self.lrFileList):
            lrName = os.path.join(self.lrDir,name)
            #midName = os.path.join(self.midDir, name)
            hrName = os.path.join(self.hrDir, name)
            #lineName = os.path.join(self.lineDir, name)
            lrImg = tifffile.imread(lrName)
            #midImg = tifffile.imread(midName)
            hrImg = tifffile.imread(hrName)
            #lineImg = tifffile.imread(lineName)

            lrImg = np.expand_dims(lrImg, axis=0)
            #midImg = np.expand_dims(midImg, axis=0)
            hrImg = np.expand_dims(hrImg, axis=0)
            #lineImg = np.expand_dims(lineImg, axis=0)

            #lrImg = lrImg.astype(np.float)
            #midImg = midImg.astype(np.float)
            #hrImg = hrImg.astype(np.float)

            #midImg = midImg / 255
            #hrImg = hrImg / 255

            self.lrImgList.append(lrImg)
            #self.midImgList.append(midImg)
            self.hrImgList.append(hrImg)
            #self.lineImgList.append(lineImg)

    def Check(self):
        return self.check

    def DataNum(self):
        return len(self.hrFileList)

    def __len__(self):
        return self.epoch#len(self.hrFileList)

    def len(self):
        return self.epoch#len(self.hrFileList)

    def __getitem__(self, ind):
        flag = True
        while flag:
            ind = np.random.randint(0,len(self.hrFileList)+len(self.crossPlace))
            #ind = np.random.randint(44, len(self.hrFileList))
            if ind < len(self.hrFileList):
                # if ind > 2 and ind < len(self.hrFileList)+8:
                #     ind = 3
                sz = self.lrImgList[ind].shape
                self.beg[0] = np.random.randint(0, sz[1] - self.cropSz[0] - 1)
                self.beg[1] = np.random.randint(0, sz[2] - self.cropSz[1] - 1)
                self.beg[2] = np.random.randint(0, sz[3] - self.cropSz[2] - 1)
            else:
                if ind < len(self.hrFileList)+len(self.crossPlace):
                    curInd = ind-len(self.hrFileList)
                else:
                    curInd = len(self.hrFileList)-1
                ind = curInd

            #
            hrImg = self.hrImgList[ind][:, self.beg[0] :self.beg[0]  + self.cropSz[0] ,
                    self.beg[1] :self.beg[1]  + self.cropSz[1] ,
                    self.beg[2] :self.beg[2]  + self.cropSz[2] ]
            # hrImg = self.hrImgList[ind][:, self.beg[0] * 4:self.beg[0] * 4 + self.cropSz[0] * 4,
            #         self.beg[1] * 2:self.beg[1] * 2 + self.cropSz[1] * 2,
            #         self.beg[2] * 2:self.beg[2] * 2 + self.cropSz[2] * 2]

            if np.sum(hrImg) < 100:
                pass
            else:
                lrImg = self.lrImgList[ind][:, self.beg[0]:self.beg[0] + self.cropSz[0],
                        self.beg[1]:self.beg[1] + self.cropSz[1],
                        self.beg[2]:self.beg[2] + self.cropSz[2]]
                flag = False


        rid = np.random.randint(0,6)
        if rid == 0:
            pass#return lrImg, midImg, hrImg
        if rid == 1:
            lrImg,hrImg = lrImg[:,::-1,:,:], hrImg[:,::-1,:,:]
        if rid == 2:
            lrImg,hrImg =  lrImg[:,:,::-1,:], hrImg[:,:,::-1,:]
        if rid == 3:
            lrImg,hrImg =  lrImg[:,:,:,::-1], hrImg[:,:,:,::-1]
        if rid == 4:
            lrImg,hrImg = lrImg[:,::-1,::-1,:],  hrImg[:,::-1,::-1,:]
        if rid == 5:
            lrImg,hrImg =  lrImg[:,:,::-1,::-1], hrImg[:,:,::-1,::-1]

        lrImg = torch.from_numpy(lrImg.copy().astype(np.float)).float()
        #midImg = torch.from_numpy(midImg.copy().astype(np.float)).float()
        hrImg = torch.from_numpy(hrImg.copy().astype(np.float)).float()
        #midImg = midImg / 255
        hrImg = hrImg / 255
        return lrImg,  hrImg