
import os
from torch.autograd import Variable
import numpy as np

from libtiff import TIFF

def libTIFFRead(src):
    tif = TIFF.open(src, mode="r")
    im_stack = list()
    for im in list(tif.iter_images()):
        im_stack.append(im)
    tif.close()
    im_stack = np.array(im_stack)
    if(im_stack.shape[0]==1):
        im_stack= im_stack[0]
    return im_stack


def libTIFFWrite(path,img):
    tif = TIFF.open(path, mode='w')
    if (img.ndim == 2):
        tif.write_image(img, compression='lzw')
    if(img.ndim==3):
        for i in range(0, img.shape[0]):
            im = img[i]
            tif.write_image(im, compression='lzw')
    tif.close()

def file_list(dirname, ext='.tif'):
    src_path = list(filter(
        lambda filename: os.path.splitext(filename)[1] == ext,
        os.listdir(dirname)))
    src_path.sort()

    return src_path



path = 'data/detect/label_modify/'
dst = 'data/detect/label_modify/'
filelist = file_list(path)

for i in range(0, len(filelist)):
    temp_path = os.path.join(path, filelist[i])
    img = libTIFFRead(temp_path)
    dst_path  = os.path.join(dst, filelist[i])
    libTIFFWrite(dst_path,img)
