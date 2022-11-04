import numpy as np
import random
import os
import matplotlib.image as mpig
from libtiff import TIFF
import mpi4py.MPI as MPI
import math
import threading
from skimage import  measure,color
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




def fill_hole_in_2Dz(image):

    for i in range(0, image.shape[0]):
        temp = ndimage.binary_fill_holes(image[i]).astype(np.uint8)
        image[i] = temp

    return image * 255


def fill_hole_in_2Dxyz(image):

    for i in range(0, image.shape[0]):
        temp = ndimage.binary_fill_holes(image[i]).astype(np.uint8)
        image[i] = temp

    for i in range(0, image.shape[1]):

        temp = ndimage.binary_fill_holes(image[:, i, :]).astype(np.uint8)
        image[:, i, :] = temp

    for i in range(0, image.shape[2]):
        temp = ndimage.binary_fill_holes(image[:, :, i]).astype(np.uint8)
        image[:, :, i] = temp

    for i in range(0, image.shape[0]):
        temp = ndimage.binary_fill_holes(image[i]).astype(np.uint8)
        image[i] = temp


    return image * 255


"""remove_small_region 
   @param image: ndarry ，ndim==3
   @param border : If the center of mass of the region is within border pixel from the block boundary, the region is processed.
   @param area_size: region size
   @param extent: If the ratio of isolated region to external boxes is higher than extent, then it is non-vascular
   @:param small_region_size 
   @:return : ndim==3 np.uint8，
 """
def remove_small_region(image,border=16,area_size=100, extent =0.4, small_region_size = 40):

    labels = measure.label(image)
    properties = measure.regionprops(labels)

    for i in range(0, len(properties)):
        if ((properties[i].area < area_size and properties[i].extent >= extent
             and properties[i].centroid[0] >= border and properties[i].centroid[0] < image.shape[0]-border
             and properties[i].centroid[1] >= border and properties[i].centroid[1] < image.shape[1]-border
             and properties[i].centroid[2] >= border and properties[i].centroid[2] < image.shape[2]-border)
                or properties[i].area < small_region_size

        ):
            temp = ~(labels == properties[i].label)
            temp = temp.astype(np.uint8)
            image = image * temp

    return image

def block_read_process_thread(id,total_tread,i,xy_size,list_file,data, x_cell, y_cell, xsize,ysize, temp_block,step,block_size):

    for j in range(id, xy_size, total_tread):
        temp_cell = libTIFFRead(os.path.join(src, list_file[i * xy_size + j]))
        # temp_cell = temp_cell[0]
        
        temp_cell1 = ndimage.binary_dilation(temp_cell).astype(temp_cell.dtype)*255
        temp_cell = ndimage.median_filter(temp_cell1,3)

        temp_cell = fill_hole_in_2Dz(temp_cell)
        #temp_cell = fill_hole_in_2Dxyz(temp_cell)
        temp_cell = remove_small_region(temp_cell)



        #print(str(data[i * xy_size + j][0]) + " " + str(data[i * xy_size + j][1]) + " " + str(data[i * xy_size + j][2]))

        zb = int(data[i * xy_size + j][0] * step)
        zend = int(data[i * xy_size + j][0] * step + block_size)
        yb = int(data[i * xy_size + j][1] * step)
        yend = int(data[i * xy_size + j][1] * step + block_size)
        xb = int(data[i * xy_size + j][2] * step)
        xend = int(data[i * xy_size + j][2] * step + block_size)

        if (data[i * xy_size + j][1] == y_cell - 1):
            yend = ysize
        if (data[i * xy_size + j][2] == x_cell - 1):
            xend = xsize
        temp_block[0:block_size, yb:yend, xb:xend] = temp_block[0:block_size, yb:yend, xb:xend] | temp_cell[0:zend - zb,
                                                                                                  0:yend - yb,
                                                                                                  0:xend - xb]


def block_read_process_thread2(id,total_tread,i,xy_size,list_file,data, x_cell, y_cell, xsize,ysize, temp_block,step,block_size):

    for j in range(id, xy_size, total_tread):
        temp_cell = libTIFFRead(os.path.join(src, list_file[i * xy_size + j]))
        # temp_cell = temp_cell[0]
        temp_cell1 = ndimage.binary_dilation(temp_cell).astype(temp_cell.dtype)*255
        temp_cell = ndimage.median_filter(temp_cell1,3)
		
        temp_cell = fill_hole_in_2Dz(temp_cell)
        #temp_cell = fill_hole_in_2Dxyz(temp_cell)
        temp_cell = remove_small_region(temp_cell)

        #print(str(data[i * xy_size + j][0]) + " " + str(data[i * xy_size + j][1]) + " " + str(data[i * xy_size + j][2]))

        zb = int(data[i * xy_size + j][0] * step)
        zend = int(data[i * xy_size + j][0] * step + block_size)
        yb = int(data[i * xy_size + j][1] * step)
        yend = int(data[i * xy_size + j][1] * step + block_size)
        xb = int(data[i * xy_size + j][2] * step)
        xend = int(data[i * xy_size + j][2] * step + block_size)

        if (data[i * xy_size + j][1] == y_cell - 1):
            yend = ysize
        if (data[i * xy_size + j][2] == x_cell - 1):
            xend = xsize
        temp_block[block_size-overlap:block_size+block_size-overlap, yb:yend, xb:xend] =temp_block[block_size-overlap:block_size+block_size-overlap, yb:yend, xb:xend] | temp_cell[0:zend - zb,
                                                                                                  0:yend - yb,  0:xend - xb]





if __name__ == '__main__':

    # segmented 3D blocks path
    src = '/lustre/ExternalData/liyuxin/dataset/hip/193882/right_seg/'
    # 2D slices save path
    dst = '/lustre/ExternalData/liyuxin/dataset/hip/193882/right_merge/'
    # block size
    block_size = 192
    # overlap area
    overlap = 32

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    res = src + 'res.txt'
    res_data = np.loadtxt(res)

    zsize = int(res_data[0])
    ysize = int(res_data[1])
    xsize = int(res_data[2])

    list_file = file_list(src)
    list_file.sort()
    path = os.path.join(src, list_file[0])
    img0 = libTIFFRead(path)

    z = img0.shape[0]
    y = img0.shape[1]
    x = img0.shape[2]

    step = block_size - overlap

    x_cell = int(math.ceil((xsize - block_size) / step)) + 1
    y_cell = int(math.ceil((ysize - block_size) / step)) + 1
    z_cell = int(math.ceil((zsize - block_size) / step)) + 1

    if comm_rank == 0:
        start = time.perf_counter() #记录结束时间
    
        for index in range(1, comm_size, 1):
            section_num = (index - 1)
            print(section_num)
            comm.send(section_num, dest=index)
        for index in range((comm_size - 1), z_cell-1 + (comm_size - 1), 1):
            proc_num = comm.recv()
            section_num = index
            print(section_num)
            comm.send(section_num, dest=proc_num)
        print("Finish")
        end = time.perf_counter()
        print("total time：",end-start)
    else:
        while(True):

            section_num = comm.recv()
            if section_num >= z_cell-1:
                break

            txt = src + 'sq.txt'
            data = np.loadtxt(txt)

            xy_size = x_cell * y_cell
            i = section_num
            temp_block = np.zeros((int(block_size+block_size-overlap), int(ysize), int(xsize)), dtype=np.uint8)


            thread_list = []      
            thread_num = 12
            for tt in range(0, thread_num):
                thread = threading.Thread(target=block_read_process_thread, args=[tt,thread_num,i,xy_size,list_file,data, x_cell, y_cell, xsize,ysize, temp_block,step,block_size])
                thread.start()
                thread_list.append(thread)
            for t in thread_list:
                t.join()

            for tt in range(0, thread_num):
                thread = threading.Thread(target=block_read_process_thread2, args=[tt,thread_num,i+1,xy_size,list_file,data, x_cell, y_cell, xsize,ysize, temp_block,step,block_size])
                thread.start()
                thread_list.append(thread)
            for t in thread_list:
                t.join()

            temp_block = fill_hole_in_2Dxyz(temp_block)

            if (i == 0):
                for kk in range(0, block_size):
                    out = dst + str(kk).zfill(5) + '.tif'
                    libTIFFWrite(out, temp_block[kk])

            if (i >=1 and i < z_cell - 2):
                for kk in range(overlap, block_size):
                    out = dst + str(i * step + kk).zfill(5) + '.tif'
                    libTIFFWrite(out, temp_block[kk])
            else:
                for kk in range(overlap, block_size+block_size-overlap):

                    if (i * step + kk >= zsize):
                        break
                    out = dst + str(i * step + kk).zfill(5) + '.tif'
                    libTIFFWrite(out, temp_block[kk])

            comm.send(comm_rank, dest=0)
