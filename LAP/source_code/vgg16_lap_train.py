import argparse
import os
import shutil
import time
import pandas as pd
from skimage import io, transform
from scipy import ndimage as sio
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as misc
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
from torch.autograd import Variable
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
from PIL import Image


plt.ion()   # interactive mode

best_prec1 = 0

#Learning rate
L_Rate= 0.0001

# Number of epochs
Startepoch = 0
Endepoch = 40

# ********************************************************************************************************************
# Image Processing
# ********************************************************************************************************************
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, age = sample['images'], sample['ages']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        #landmarks = landmarks * [new_w / w, new_h / h]

        return {'images': img, 'ages': age}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, age = sample['images'], sample['ages']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'images': image, 'ages': age}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, age = sample['images'], sample['ages']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #print(image.shape)


        image = image.transpose((2, 0, 1))
        images = torch.from_numpy(image)
        #ages = torch.DoubleTensor()
        #ages=age
        return images,age
        #return {'images': torch.from_numpy(image),
        #        'ages': age}

# ********************************************************************************************************************
# Loading csv file
# ********************************************************************************************************************

class AgeEstimationDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.face_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.face_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.face_frame.ix[idx, 0])
        #print(img_name)
        #image = Image.open(img_name)
        image = misc.imread(img_name,mode='RGB')
        #image = (image - image.mean()) / image.std()

        age = self.face_frame.ix[idx, 1].astype('float')
        sample = {'images':image, 'ages':age}
        #print(img_name)
        if self.transform:
            sample = self.transform(sample)
        
        return sample


use_gpu = torch.cuda.is_available()

# ********************************************************************************************************************
# Main function
# ********************************************************************************************************************

def main():
    global print_freq,best_prec1
    print_freq=10
   
   # Loading the weights for VGG16 model trained on IMDB-WIKI dataset
    vggmodel = models.vgg16(pretrained=True)   
    net= nn.Sequential(vggmodel,
           nn.ReLU(True),              
           nn.Linear(1000, 100)                                           
           )
    net.load_state_dict(torch.load('imdb_best_final.pth.tar'))
    model = net   
    
    model.cuda() 
    print(model)
    
   
    
    criterion1 = nn.CrossEntropyLoss().cuda()
    criterion2 = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), L_Rate,
                                momentum=0.9,
                                weight_decay=1e-4)
                                                       
    cudnn.benchmark = True

    
     #Loading the data through csv and preparing dataset
    transformed_train_dataset = AgeEstimationDataset(csv_file='LAP/train1.csv',
                                           root_dir='LAP/',
                                           transform=transforms.Compose([
                                           Rescale(256),
                                           RandomCrop(224),                                           
                                           ToTensor()
                                           #normalize                                                                        
                                           ]))

    transformed_valid_dataset = AgeEstimationDataset(csv_file='LAP/valid_gt.csv',
                                           root_dir='LAP/',
                                           transform=transforms.Compose([
                                           Rescale(256),
                                           RandomCrop(224),                                           
                                           ToTensor()
                                           #normalize                                                                          
                                           ]))
    # Loading dataset into dataloader
    train_loader =  torch.utils.data.DataLoader(transformed_train_dataset, batch_size=32,
                                               shuffle=True, num_workers=8)

    val_loader =  torch.utils.data.DataLoader(transformed_valid_dataset, batch_size=32,
                                               shuffle=True, num_workers=8)

    start_time= time.time()
    
     # Train for all epochs between Startepoch, Endepoch
    for epoch in range(Startepoch, Endepoch):      
        adjust_learning_rate(optimizer, epoch)

        
        # train for one epoch
        train(train_loader, model, criterion1, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion2,epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(model.state_dict(),is_best,epoch)

    end_time = time.time() 
    duration= (end_time - start_time)/3600
    print("Duration:")
    print(duration)

# ********************************************************************************************************************
# Training the model
# ********************************************************************************************************************  
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)        
        input1= torch.FloatTensor()
        input1=input
        if use_gpu:         
           input_var = torch.autograd.Variable(input1.float().cuda())
           target_var = torch.autograd.Variable(target.long().cuda())

        else:
           input_var = torch.autograd.Variable(input1.float())
           target_var = torch.autograd.Variable(target.long())
        

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        #print(output.data)
        # measure accuracy and record loss        
        prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    f = open('lap_train.txt', 'a')
    f.write('Epoch: {0}\t''Loss: {loss.avg:.4f}\t''Prec@1 {top1.avg:.3f}\t''Prec@5 {top5.avg:.3f}\n'.format(epoch, loss=losses,top1=top1,top5=top5))        
    f.close()
        
# ********************************************************************************************************************
# Validating the model
# ********************************************************************************************************************
def validate(val_loader, model, criterion,epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input= input.cuda()
        #target = target.cuda(async=True)
        #input_var = torch.autograd.Varmodel.parameters(), lr=0.1, momentum=0.9)iable(input, volatile=True)
        #target_var = torch.autograd.Variable(target, volatile=True)
        
        input1= torch.FloatTensor()
        input1=input
        if use_gpu:           
           input_var = torch.autograd.Variable(input1.float().cuda())
           target_var = torch.autograd.Variable(target.long().cuda())

        else:
           input_var = torch.autograd.Variable(input1.float())
           target_var = torch.autograd.Variable(target.long())


        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    f = open('lap_val.txt', 'a')
    f.write('Epoch: {0}\t''Loss: {loss.avg:.4f}\t''Prec@1 {top1.avg:.3f}\t''Prec@5 {top5.avg:.3f}\n'.format(epoch,loss=losses,top1=top1,top5=top5))        
    f.close()
    return top1.avg

# ********************************************************************************************************************
# Save a copy of the model with best precision
# ********************************************************************************************************************
def save_checkpoint(state, is_best, epoch, filename='LAP/checkpoint_lap1.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'LAP/lap_best1.pth.tar')
   


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = L_Rate * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()   
    correct = pred.eq(target.view(1, -1).expand_as(pred))    
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
   main()
