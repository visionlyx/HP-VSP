"""This function is used for data augument for npy dataset"""


import random
import numpy as np

class JointCompose(object):
    def __init__(self,transforms):
        self.transforms=transforms

    def __call__(self, img, mask):
        assert img.shape == mask.shape
        for t in self.transforms:
            img,mask=t(img,mask)
        return img,mask


class JointRandomGaussianNoise(object):
    """define the noise amplitude and weight : img float and label long"""
    def __init__(self,amplitude=10,std1=15):
        self.amplitude=amplitude
        self.std1=std1

    def __call__(self, img, mask):
        # eliminate the std of the image
        nlevel=random.random()*self.amplitude/self.std1

        p=0.5
        if random.random()<p:
            # print('transform nlevel:{}'.format(nlevel))
            noise=nlevel*np.random.normal(0,1,img.shape)
            img=img+noise
            img= np.clip(img,0,1)

        return img,mask


class JointRandomBrightness(object):
    def __init__(self,limit):
        self.limit=limit

    def __call__(self, img,mask):
        p=0.5
        if random.random()<p:
            alpha=1.0+np.random.uniform(self.limit[0],self.limit[1])
            # print(alpha)
            img=alpha*img
            img = np.clip(img, 0, 1)
        return img,mask

class JointRandomIntensityChange(object):
    def __init__(self,limit,std1=15):
        self.limit=limit
        self.std1 = std1

    def __call__(self, img,mask):
        p=0.5
        if random.random()<p:
            alpha=np.random.uniform(self.limit[0],self.limit[1])/self.std1
            # print(alpha)
            img=img+alpha
            img = np.clip(img, 0, 1)
        return img,mask



