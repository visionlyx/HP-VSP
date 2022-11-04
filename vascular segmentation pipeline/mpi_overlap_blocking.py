import numpy as np
import tifffile
import random
import os
import cv2
import matplotlib.image as mpig
from libtiff import TIFF
import mpi4py.MPI as MPI
import math
import time
from scipy import ndimage
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
    return list(filter(
        lambda filename: os.path.splitext(filename)[1] == ext,
        os.listdir(dirname)))


if __name__ == '__main__':

    #2D slices path
    src = '/lustre/ExternalData/liyuxin/dataset/kidney/194798/2x2x2_ch2/'
    #overlapped 3D blocks save path
    dst = '/lustre/ExternalData/liyuxin/dataset/kidney/194798/ch2_block_2x2x2/'
    #size of blocks
    block_size = 160
    #size of  overlap area
    overlap = 20
    
    
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    list_file = os.listdir(src)
    list_file.sort()
    z = len(list_file)
    file1 = list_file[0]
    path0 = os.path.join(src, file1)
    img0 = libTIFFRead(path0)

    y = img0.shape[0]
    x = img0.shape[1]
    
    
    x_step = block_size-overlap
    y_step = block_size-overlap
    z_step = block_size-overlap

    x_cell = int(math.ceil((x-block_size) / x_step))+1
    y_cell = int(math.ceil((y-block_size) / y_step))+1
    z_cell = int(math.ceil((z-block_size) / z_step))+1
    
        
    if comm_rank == 0:

        start = time.perf_counter()
        f = dst + "size_overlap.txt"
        with open(f, "a") as file:
            file.write(str(block_size) + "      " + str(overlap) + "\n")

        f = dst + "res.txt"
        with open(f, "a") as file:
            file.write(str(z) + "      " + str(y) + "      " + str(x) + "\n")
            
            
        for i in range(0,z_cell):
            for j in range(0, y_cell):
                for k in range(0, x_cell):
                    f = dst + "sq.txt"
                    with open(f, "a") as file:
                        file.write(str(i) + "      " + str(j) + "      " + str(k) + "\n")
            

        for index in range(1,comm_size,1):
            section_num = (index-1)
            comm.send(section_num,dest=index)
        for index in range((comm_size-1),z_cell+(comm_size-1),1):
            proc_num = comm.recv()
            section_num = index
            comm.send(section_num,dest=proc_num)
        print("Finish")
        end = time.perf_counter()
        print("total time：",end-start)
    
    else:
        while(True):
            section_num = comm.recv()
            if section_num >= z_cell:
                break
            
            
            img_stack = list()
            zb = section_num * z_step
            zend = zb + block_size

            if (section_num == z_cell - 1):
                zend = z

            for dddd in range(zb, zend):
                temp_slice = libTIFFRead(os.path.join(src, list_file[dddd]))
                img_stack.append(temp_slice)

            img_stack = np.array(img_stack)
            
            for j in range(0, y_cell):
                for k in range(0, x_cell):

                    xb = k * x_step
                    yb = j * y_step

                    xend = xb + block_size
                    yend = yb + block_size


                    if(j==y_cell-1):
                        yend = y
                    if(k==x_cell-1):
                        xend = x

                    index = section_num*y_cell*x_cell+j*x_cell+k + 1
                    #index = index + 1
                    image_block =np.zeros(shape=(block_size,block_size,block_size),dtype=np.uint16)

                    image_block[0:zend-zb,0:yend-yb,0:xend-xb]=img_stack[0:zend-zb, yb: yend, xb:xend]

                    outname = dst + str(index).zfill(7) + '.tif'

                    libTIFFWrite(outname, image_block) 
                    
            comm.send(comm_rank,dest=0)
