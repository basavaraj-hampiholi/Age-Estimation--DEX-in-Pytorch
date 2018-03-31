import numpy as np
import csv
file= open("/home/hampiholi/Downloads/csvlist.txt",'r')
lines= file.readlines()
file.close()
f= open("/home/hampiholi/Desktop/file.csv",'w')
#name= lines.split("/")
#print(len(name))
for line in lines:
 name = line.split("/")
 print(len(name))
 for temp in name:
  f.write(temp)
  f.write('\n')
f.close() 
  
