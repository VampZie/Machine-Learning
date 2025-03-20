import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df=pd.DataFrame(pd.read_csv('/home/vzscyborg/datasets/given.csv'))
x=df.select_dtypes(include=['float64','int64'])
dfpp=StandardScaler().fit_transform(x)

wcss=[]
for k in range(1,11):
    kms=KMeans(n_clusters=k, init='k-means++', random_state=42)
    kms.fit_transform(dfpp)
    wcss.append(kms.inertia_)

plt.plot(range(1,11), wcss, marker='o', linestyle='--')
plt.xlabel('number of clusters k')
plt.ylabel('wcss probability')
plt.show()
