import cv2
import fnmatch
import os,sys
import pandas as pd

df = pd.read_csv('/hampiholidata/Project/Datasets/imdb/adience_face_locs.csv')
img_files = df['Images']
x1 = df['x1'].values.tolist()
y1 = df['y1'].values.tolist()
x2 = df['x2'].values.tolist()
y2 = df['y2'].values.tolist()
mypath='/hampiholidata/Project/Datasets/imdb/adience/'
for i, images in enumerate(img_files):
   im=os.path.join(mypath,images)
   img = cv2.imread(im)
   #cv2.imshow("Original",img)
   print(int(y1[i]),int(y2[i]),int(x1[i]),int(x2[i]))
   crop_img = img[int(x1[i]):int(y1[i]), int(x2[i]):int(y2[i])] # Crop from x, y, w, h -> 100, 200, 300, 400
   # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
   cv2.imshow("cropped", crop_img)
   cv2.waitKey(0)
   if i== 5: 
     break
print('Done')
