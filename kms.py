import numpy as np
import pandas as pd
import pyBigWig as bw
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


path='/home/vzscyborg/datasets/given.csv'
df=pd.DataFrame(pd.read_csv(path))
print(df.head())

dfpp=StandardScaler().fit_transform(df.select_dtypes(include=['float64','int64']))

wcss=[]
for k in range(1,11):
    kms=KMeans(n_clusters=k,  init='k-means++' ,random_state=50)
    kms.fit(dfpp)
    wcss.append(kms.inertia_)

plt.plot(range(1,11), wcss, marker='o', color='black')
plt.show()