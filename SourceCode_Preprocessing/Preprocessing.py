import cv2                 
import numpy as np         
import os                 
from random import shuffle 
from tqdm import tqdm      

TRAIN_DIR = '/home/basavaraj/WindowsShare/set11'
TEST_DIR = '/home/basavaraj/Downloads/train/test'
IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'planesvsfacesvsbikes-{}-{}.model'.format(LR, '2conv-basic') 

def label_img(img):
    word_label = img.split('.')[-3]
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == 'cat': return [1,0]
    #                             [no cat, very doggo]
    elif word_label == 'dog': return [0,1]

def create_train_data():
    training_data = []
    tr=1
    #onlyfiles = [ f for f in listdir(TRAIN_DIR) if isfile(join(mypath,f)) ]
    #images = np.empty(len(onlyfiles), dtype=object)
    #for n in range(0, len(onlyfiles)):
    #images[n] = cv2.imread(os.path.join(TRAIN_DIR,onlyfiles[n]))
    for img in tqdm(os.listdir(TRAIN_DIR)):  
	label = label_img(img)      
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data11.npy', training_data)
    return training_data

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('(')[-2]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

train_data = create_train_data()
#test_data = process_test_data()
