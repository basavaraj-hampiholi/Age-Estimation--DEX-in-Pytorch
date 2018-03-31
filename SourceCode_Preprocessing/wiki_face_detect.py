from __future__ import division
#import _init_paths
#from fast_rcnn.config import cfg
#from fast_rcnn.test import im_detect
#from fast_rcnn.nms_wrapper import nms
#from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os, sys
import argparse
import sys


def get_imdb_fddb(data_dir):
  imdb = []
  #nfold = 10
  #for n in xrange(nfold):
  file_name = 'Train_data.txt' 
  file_name = os.path.join(data_dir, file_name)
  fid = open(file_name, 'r')
  image_names = []
  for im_name in fid:
      image_names.append(im_name.strip('\n'))
      print(im_name[1])
  imdb.append(image_names)

  return imdb

data_dir = '/home/hampiholi/Project/Datasets/wiki_crop/'
imdb = get_imdb_fddb(data_dir)
#print(imdb)

