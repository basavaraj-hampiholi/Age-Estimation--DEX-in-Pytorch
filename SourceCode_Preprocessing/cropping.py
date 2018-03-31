import fnmatch
import os,sys
import pandas as pd
import shutil
import numpy as np
from PIL import Image
from PIL import ImageOps

df = pd.read_csv('/hampiholidata/Project/Datasets/LAP/train1_face_locs.csv')
img_files = df['Images']
x1 = df['x1'].values.tolist()
y1 = df['y1'].values.tolist()
x2 = df['x2'].values.tolist()
y2 = df['y2'].values.tolist()
mypath='/hampiholidata/Project/Datasets/LAP/train/'

for i, images in enumerate(img_files):
   im=os.path.join(mypath,images)
   img = Image.open(im)
   img.show()   
   imgs= img.crop((x1[i],y1[i],x2[i],y2[i]))
   #imgs.show()
   print(im)
   im_sz= imgs.size
   #print(im_sz[0])
   new_sz_x= im_sz[0] + im_sz[0] % 40
   new_sz_y= im_sz[1] + im_sz[1] % 40
   new_sz = (new_sz_x, new_sz_y)
   new_im= ImageOps.expand(imgs,border=100,fill='black')
   new_im.show()
   new_im.save('/hampiholidata/Project/Datasets/LAP/laptrain/'+images)
      #shutil.copy(os.path.join(pth[i],images[0]),'/hampiholidata/Project/Datasets/LAP/laptrain')

#np.savetxt('/home/hampiholi/Project/Datasets/imdb/Adience_refined.csv',file_list)
print('Done')
