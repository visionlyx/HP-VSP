from utils.tiff_read import *
import torch.optim as optim
import cv2 as cv
import numpy as np
import time
import torch
from utils.dataloader import Seg_Dataset, yolo_dataset_collate
import  os
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils.logger import *
from lossfunc import  *
from utils.joint_transform import *

from net.mi32net import *

from Util import  *


if __name__ == "__main__":

    #-----参数设置-------------------
    #图像灰度均值
    all_mean1 = np.array([97], dtype=np.float32)
    #tensorboard路径
    logger = Logger("tensorboard/")
    #learn_rate
    lr = 0.0001

    Batch_size = 12
    start_epoch = 1
    epochs = 5000
    Cuda = True
    Use_Data_Loader = True
    weights_path = "logs/Epoch636_train_loss_0.004304486885666847_val_loss_0.004263845272362232.pth"

    os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'



    train_path = 'train.txt'
    with open(train_path) as f:
        lines_train = f.readlines()

    eval_path = 'val.txt'
    with open(eval_path) as f:
        lines_val = f.readlines()

    model = MINet32_new_new_new()

    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)

    if Use_Data_Loader:
        print('Loading weights into state dict...')
        checkpoint = torch.load(weights_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        start_epoch = checkpoint['epoch'] + 1
        print('Finished!')

    criteria = FocalWBCEloss(weight=1,gamma=0)
    criteria.cuda()

    net = torch.nn.DataParallel(model.cuda(), device_ids=[0, 1, 2, 3])
    cudnn.benchmark = False

    train_aug = JointCompose([
                              JointRandomBrightness([-0.2, 0.2]),
                              ])

    train_dataset = Seg_Dataset(lines_train,  is_train=True, augument=train_aug,pixel_means= all_mean1)
    val_dataset = Seg_Dataset(lines_val, is_train=False,augument=train_aug,pixel_means=all_mean1)

    train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True,
                     collate_fn=yolo_dataset_collate)
    val_loader = DataLoader(val_dataset, batch_size=Batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True,
                         collate_fn=yolo_dataset_collate)

    for epoch in range(start_epoch, epochs):

        total_loss = 0
        val_loss = 0
        net.train()
        for iteration_train, batch in enumerate(train_loader):

            batches_done = len(train_loader) * epoch + iteration_train
            images, target = batch[0], batch[1]

            images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
            target = Variable(torch.from_numpy(target).type(torch.FloatTensor)).cuda()

            optimizer.zero_grad()

            output = net(images)


            loss = criteria(output, target)

            loss.backward()
            optimizer.step()

            tensorboard_log = []
            tensorboard_log += [("train loss iter", loss.item())]
            logger.list_of_scalars_summary(tensorboard_log, batches_done)

            total_loss += loss
            #print("iter: %d, train loss is %f"%(iteration_train, loss.item()))
        scheduler.step()
        #print('start validation:')

        model.eval()
        for iteration_val, batch_val in enumerate(val_loader):
            with torch.no_grad():
                images_val, targets_val = batch_val[0], batch_val[1]
                images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                targets_val = Variable(torch.from_numpy(targets_val).type(torch.FloatTensor)).cuda()
                optimizer.zero_grad()
                output = net(images_val)
                loss = criteria(output, targets_val)
                val_loss += loss
                #print("iter: %d, validation loss is %f"%(iteration_val, loss.item()))

        print('--------Epoch %d total train loss: %f  total val loss: %f--------' % (
            epoch, total_loss.item() / (iteration_train + 1), val_loss.item() / (iteration_val + 1)))


        loss_epochs = [
            ("train loss epoch", total_loss.item() / (iteration_train + 1)),
            ("val loss epoch", val_loss.item() / (iteration_val + 1)),
        ]
        logger.list_of_scalars_summary(loss_epochs, epoch)

        if (epoch % 1 ==0):
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            path = 'logs/Epoch' +str(epoch)+'_train_loss_'+str(total_loss.item()/(iteration_train+1))+'_val_loss_'+ str(val_loss.item()/ (iteration_val+1))+ '.pth'
            torch.save(state, path)
