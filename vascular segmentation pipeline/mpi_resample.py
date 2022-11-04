
import mpi4py.MPI as MPI
import numpy as np
import scipy.ndimage
import os
from skimage import measure, color
import torch
import torch.nn as nn
from torch.autograd import Variable
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





def max_2x_resample_z(image1,image2):
    image = np.maximum(image1,image2)
    return image

def max_2x_resample_xy(image):
    image = image/1.0
    image = Variable(torch.from_numpy(image).type(torch.FloatTensor))
    image = image.unsqueeze(0)
    m = nn.MaxPool2d(kernel_size=2, stride=2)
    output = m(image)
    output = output.squeeze(0)
    output = np.array(output, dtype=np.uint16)
    return output

def avg_2x_resample_label(image):
    img_out = scipy.ndimage.zoom(image, 0.5, order=2)

    
    return img_out


def max_2x_resample_label(image):
    image = image/1.0
    image = Variable(torch.from_numpy(image).type(torch.FloatTensor))
    image = image.unsqueeze(0)
    m = nn.MaxPool2d(kernel_size=2, stride=2)
    output = m(image)
    output = output.squeeze(0)
    output = np.array(output, dtype=np.uint8)
    return output


if __name__ == '__main__':


    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    #original 2D slices
    src = '/lustre/ExternalData/liyuxin/dataset/hip/193882/left_merge/'
    #resampled 2D slices
    dst = '/lustre/ExternalData/liyuxin/dataset/hip/193882/left_merge2x2x2/'
    list_file = os.listdir(src)
    list_file.sort()
        
    if comm_rank == 0:
        for index in range(1,comm_size,1):
            slice_num = (index-1)*2
            comm.send(slice_num,dest=index)
            #print(list_file[index])
        for index in range((comm_size-1)*2,len(list_file)+(comm_size-1)*2,2):
            proc_num = comm.recv()
            slice_num = index
            comm.send(slice_num,dest=proc_num)
            #print(list_file[index])
        print("Finish")
    else:
        while(True):
            slice_num = comm.recv()
            if slice_num >= len(list_file):
                break
            #print(list_file[slice_num])
            path1 = os.path.join(src, list_file[slice_num])
            img1 = libTIFFRead(path1)
            img1 = max_2x_resample_label(img1)

            path2 = os.path.join(src, list_file[slice_num+1])
            img2 = libTIFFRead(path2)
            img2 = max_2x_resample_label(img2)

            img = max_2x_resample_z(img1,img2)
            out = os.path.join(dst, list_file[slice_num])
            libTIFFWrite(out, img)
            
            comm.send(comm_rank,dest=0)

    
    
    
    
    
    
    
    
    
    
