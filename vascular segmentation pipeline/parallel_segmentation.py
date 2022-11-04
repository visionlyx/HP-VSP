import torch
import os
from torch.autograd import Variable
from libtiff import *
from scipy.ndimage import filters
from net.mi32net import *
from random import shuffle
import numpy as np
import tifffile
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from skimage import morphology
import threading
import time #导入模块

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


class Detect_dataset(Dataset):
    def __init__(self, src_path, pixel_mean):
        super(Detect_dataset, self).__init__()

        self.train_src = src_path
        self.filelist = file_list(src_path)
        self.train_batches = len(self.filelist)
        self.pixel_mean = pixel_mean
    def __len__(self):
        return self.train_batches

    def __getitem__(self, index):


        temp_path = os.path.join(self.train_src, self.filelist[index])
        img = libTIFFRead(temp_path)
        #img = filters.gaussian_filter(img,2)
        #img = ndimage.median_filter(img,3)
        #libTIFFWrite(self.filelist[index],img)

        

        img = np.where(img < self.pixel_mean*2, 0, img)
        img=np.clip(img,0,self.pixel_mean*10)
        

        #mask = morphology.remove_small_objects(mask, min_size=4, connectivity=1, in_place=True)
        img = np.array(img, dtype=np.float32)

        img = (img) / (self.pixel_mean*10)  # z y x

        img = img[np.newaxis, :]  # c z y x

        img = np.array(img, dtype=np.float32)

        return img, self.filelist[index]


def yolo_dataset_collate(batch):
    images = []
    filenames = []
    for img, filename in batch:
        images.append(img)
        filenames.append(filename)
    images = np.array(images)
    return images, filenames


def file_list(dirname, ext='.tif'):
    src_path = list(filter(
        lambda filename: os.path.splitext(filename)[1] == ext,
        os.listdir(dirname)))
    src_path.sort()

    return src_path



def block_write_process_thread(id,images,dst,file_names):

        temp = images[id].cpu()
        temp = temp.squeeze(0)
        temp = temp.detach().numpy()
        temp = np.where(temp < 40, 0, 255)
        temp = np.array(temp,dtype=np.uint8)
        outfile = os.path.join(dst, file_names[id])
        libTIFFWrite(outfile, temp)



if __name__ == '__main__':

    #avg pixel value of dataset
    all_mean1 = np.array([40], dtype=np.float32)

    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'

    #overlapped 3D blocks path
    tiff_path = '/lustre/ExternalData/liyuxin/dataset/hip/193882/right_block/'

    #segmented blocks save path
    dst = '/lustre/ExternalData/liyuxin/dataset/hip/193882/right_seg/'


    net = MINet32_new_new_new()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #pertrained network
    temp = torch.load("model_pretrained.pth")
    
    net.load_state_dict(temp['model'])

    print('Loading weights into state dict...')

    
    net = torch.nn.DataParallel(net.cuda(), device_ids=[0,1,2,3,4,5,6,7])
    #cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    test_dataset = Detect_dataset(tiff_path,pixel_mean=all_mean1)
    test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False, num_workers=8, pin_memory=True, drop_last=False,
                              collate_fn=yolo_dataset_collate)
                              
    net = net.eval()
                      
    start = time.perf_counter() #begin time
    for iteration_train, batch in enumerate(test_loader):


        images = batch[0]
        filenames = batch[1]
        #images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
        images = Variable(torch.from_numpy(images)).cuda()
        with torch.no_grad():
            r_image = net(images)

        #r_image = torch.where(r_image < 0.5, 0, 1)
        r_image = r_image * 255

        thread_list = []
        thread_num = 24
        for tt in range(0, thread_num):
            thread = threading.Thread(target=block_write_process_thread, args=[tt,r_image,dst,filenames])
            thread.start()
            thread_list.append(thread)
        for t in thread_list:
            t.join()
        
    end = time.perf_counter() #end time
    print("total time：",end-start)





        
