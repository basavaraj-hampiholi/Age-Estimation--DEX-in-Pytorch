import pandas as pd

df=pd.read_csv('/hampiholidata/Multiview/train.csv')

grouped=df.groupby('Age')['Images'].nunique()
for age in grouped.index:
#for i, a in enumerate(df['Age']):
 print(age)
 df_age= df.ix[df['Age'] == age]
 if(len(df_age) > 2000):
   n=len(df_age)-2000
   df_sample= df_age.sample(n)
   df=df.drop(df_sample.index) 

df.to_csv('/hampiholidata/Project/Datasets/imdb/Augment/IMDB_Wiki_Adience_Train2000_undersample.csv') 
df1= df.groupby('Age')['Images'].nunique()
df1.to_csv('/hampiholidata/Project/Datasets/imdb/Augment/IMDB_Wiki_Adience_Train2000_undersample_dis.csv')  
print('done')
