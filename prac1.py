import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import mplcursors 

path='/home/vzscyborg/datasets/given.csv'
df=pd.DataFrame(pd.read_csv(path))
dfpp=StandardScaler().fit_transform(df.select_dtypes(include=['float64','int64']))

wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++', random_state=50)
    kmeans.fit_transform(dfpp)
    wcss.append(kmeans.inertia_)


kms=KMeans(n_clusters=6, init='k-means++', random_state=50)
clstrs=kms.fit_predict(dfpp)
df['clusters']=clstrs

pca=PCA(n_components=2)
dfpp_pca=pca.fit_transform(dfpp)

cntrd=pca.transform(kms.cluster_centers_)

plt.scatter(dfpp_pca[:,0], dfpp_pca[:,1], cmap='viridis', c=clstrs, alpha=0.7)
plt.scatter(cntrd[:,0], cntrd[:,1], s=50, c='red', marker='o')
cursor = mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(f"Cluster: {clstrs[sel.index]}"))
plt.show()