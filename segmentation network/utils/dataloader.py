from random import shuffle
import numpy as np

from torch.utils.data.dataset import Dataset
import random
from libtiff import TIFF
from numba import jit
import math
from PIL import Image, ImageDraw, ImageFont
from ctypes import *
from utils.tiff_read import *
from utils.joint_transform import *
#from dataEnhance.joint_transform import *

class Seg_Dataset(Dataset):
    def __init__(self, train_lines, is_train, augument,pixel_means):
        super(Seg_Dataset, self).__init__()
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.is_train = is_train
        self.augument = augument
        self.limit = [0,0.5]
        self.pixel_means= pixel_means



    def __len__(self):
        return self.train_batches

    def random_transpone(self, img, label):

        index = random.randint(0, 5)

        if(index == 0):
            return img, label
        if(index == 1):
            return img.transpose(0, 2, 1), label.transpose(0, 2, 1)
        if (index == 2):
            return img.transpose(1, 0, 2), label.transpose(1, 0, 2)
        if (index == 3):
            return img.transpose(1, 2, 0), label.transpose(1, 2, 0)
        if (index == 4):
            return img.transpose(2, 0, 1), label.transpose(2, 0, 1)
        if (index == 5):
            return img.transpose(2, 1, 0), label.transpose(2, 1, 0)

    def change(self,image,label):
        image, label = self.random_transpone(image, label)
        # imgae, label = self.JointRandomRotation(image, label)
        #image, label = self.JointRandomBrightness(image, label)
        return image,label

    def get_data(self, annotation_line):

        line = annotation_line.split()
        #tifffile读取
        image = libTIFFRead(line[0])
        label = libTIFFRead(line[1])

        image = np.array(image, dtype=np.float32)
        label = np.array(label, dtype=np.float32)

        if(image.shape[1]==160 or image.shape[1]==192 ):

            image = np.where(image < self.pixel_means*2, 0, image)
            image=np.clip(image,0,self.pixel_means*10)

            if (self.is_train == True):
                # random crop
                z_b = np.random.randint(160 - 128)
                y_b = np.random.randint(160 - 128)
                x_b = np.random.randint(160 - 128)
                z_e = z_b + 128
                y_e = y_b + 128
                x_e = x_b + 128
                image = image[z_b:z_e, y_b:y_e, x_b:x_e]
                label = label[z_b:z_e, y_b:y_e, x_b:x_e]

                image,label = self.change(image,label)

            if (self.is_train == False):
                image = image[16:144, 16:144, 16:144]
                label = label[16:144, 16:144, 16:144]

            image = np.transpose((image) / (self.pixel_means*10), (0, 1, 2))  # z y x
            label = np.transpose((label) / 255, (0, 1, 2))   # z y x

        if self.augument:
            image_patch, label_patch = self.augument(image, label)


        return image, label

    def __getitem__(self, index):

        #t1 = time.clock()

        if index == 0:
            shuffle(self.train_lines)
        lines = self.train_lines
        n = self.train_batches
        index = index % n
        tmp_inp, tmp_targets = self.get_data(lines[index])
        tmp_inp = tmp_inp[np.newaxis, :]    # c z y x
        tmp_targets = tmp_targets[np.newaxis, :]


        return tmp_inp, tmp_targets

def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    bboxes = np.array(bboxes)
    return images, bboxes

