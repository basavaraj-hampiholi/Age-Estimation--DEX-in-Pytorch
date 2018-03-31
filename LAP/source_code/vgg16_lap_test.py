import argparse
import os
import shutil
import time
import pandas as pd
from skimage import io, transform
from scipy import ndimage as sio
import numpy as np
import matplotlib.pyplot as plt

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
from collections import OrderedDict
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


plt.ion()   # interactive mode

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
        #print(image)
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
        image,age = sample['images'], sample['ages']['age']
        fname,std = sample['ages']['stdv'], sample['ages']['filename'] 
        
        #print(sample['stdv'])
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #print(image.shape)


        image = image.transpose((2, 0, 1))
        images = torch.from_numpy(image)
       
        return images,age
        

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
        image = sio.imread(img_name,mode='RGB')
        #image = (image - image.mean()) / image.std()
        age = self.face_frame.ix[idx, 1]
        std = self.face_frame.ix[idx, 2]        
        sample = {'images': image, 'ages':{'age':age, 'stdv': std,'filename': img_name}}        
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
    
    # Loading the weights for VGG16 model trained on LAP dataset
    tmodel = models.vgg16(pretrained=True)   
    net= nn.Sequential(tmodel,
           nn.LogSoftmax(),
           nn.Linear(1000, 100),
           nn.Softmax()                    
           )  
        
    net.load_state_dict(torch.load('LAP/lap_best_1.pth.tar'))
    model = net
    model.cuda() 
   
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
                                                   
    cudnn.benchmark = False
   
    #Loading the data through csv and preparing dataset   
    transformed_test_dataset = AgeEstimationDataset(csv_file='LAP/test_gt.csv',
                                           root_dir='LAP/',
                                           transform=transforms.Compose([
                                           Rescale(256),
                                           RandomCrop(224),                                           
                                           ToTensor()
                                           #normalize                                                                          
                                           ]))
        
    # Loading dataset into dataloader
    test_loader =  torch.utils.data.DataLoader(transformed_test_dataset, batch_size=1,
                                               shuffle=True, num_workers=8)
    

    
    start_time= time.time()
    
    #Test the model
    prec1 = test(test_loader, model, criterion)

    end_time = time.time() 
    duration= (end_time - start_time)/3600
    print("Duration:")
    print(duration)

use_gpu = torch.cuda.is_available()


# ********************************************************************************************************************
# Test the model
# ********************************************************************************************************************

def test(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input= input.cuda()
        
        
        input1= torch.FloatTensor()
        input1= input
        
     
        target_age = target['age']
      
        
        if use_gpu:         
           input_var = torch.autograd.Variable(input1.float().cuda())
           target_var = torch.autograd.Variable(target_age.long().cuda())
           

        else:
           input_var = torch.autograd.Variable(input1.float())
           target_var = torch.autograd.Variable(target_age.long())
          
        fname = target['filename']
        sigma= target['stdv']
       
        total_sigma= 2*sigma*sigma

        # compute output
        output = model(input_var)
        confusion_matrix.add(output.data.squeeze(), target_var)
        loss = criterion(output, target_var)
        
        # Softmax probabilities and predicted age
        prob,label= torch.topk(output.data, 100)
        #Expected value
        total= prob.mul(label.float())
        estimated_age= total.sum()
        
        #Writing filename and estimated age to text file 
        f = open('lap6.txt', 'a')
        f.write('{0}\t {1}\n'.format(fname,estimated_age))
        f.close()

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
    
   
    #f.write('Output: {output:.4f}\t' 'Target: {target:.4f}\t' 'Prec@1 {top1.avg:.3f}\t''Prec@5 {top5.avg:.3f}\n'.format(output,target,top1=top1,top5=top5))        
    
    return top1.avg

def save_checkpoint(state, is_best, epoch, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
    


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
    lr = L_Rate * (0.1 ** (epoch // 40))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
   
    batch_size = target.size(0)
    predicted = AverageMeter()
    _, pred = output.topk(maxk, 1, True, True)
    
    pred = pred.t()
    predicted=pred
    correct = pred.eq(target.view(1, -1).expand_as(pred))
 
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        #print(correct_k)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
   

if __name__ == '__main__':
   main()
