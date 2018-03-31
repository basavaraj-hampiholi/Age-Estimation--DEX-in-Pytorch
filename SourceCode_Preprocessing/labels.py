import os,glob
import scipy.misc as si
from tqdm import tqdm
import numpy as np
from random import shuffle 

TRAIN_DIR = '/home/hampiholi/Project/Datasets/imdb/imdb_crop'
IMG_SIZE=50



def scanfolder():
    #imgs= []
    #labels= []
    text_file = open("Train_data_imdb.txt", "a")
    training_data= []
    for path,dirs, files in tqdm(os.walk(TRAIN_DIR)):
        print(path)
        for img in tqdm(files):
            if img.endswith('.jpg'):
                age= int(img[-8:-4])-int(img[-19:-15])
                if age >= 0:
                  #imgs.append(img)
                  #labels.append(age)
                  pth = os.path.join(path,img)
                  text_file.write(pth)
                  text_file.write(" ")
                  text_file.write(str(age))
                  text_file.write('\n')
                  img = si.imread(pth,mode='RGB')
                  img = si.imresize(img, (IMG_SIZE,IMG_SIZE))
                  #training_data.append([np.array(img), age])
    #shuffle(training_data)
    #np.save('train_data.npy', training_data)
    print(len(training_data))
    text_file.close()

scanfolder()
