import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import kmeans_elbow
from sklearn.decomposition import PCA
df=pd.DataFrame(pd.read_csv('/home/vzscyborg/datasets/given.csv'))
x=df.select_dtypes(include=['float64','int64'])
dfpp=StandardScaler().fit_transform(x)
ok=int(input('Select the optimum k: '))
kms=KMeans(n_clusters=ok, init='k-means++', random_state=42)
clusters=kms.fit_predict(dfpp)
df['cluster']=clusters
pca=PCA(n_components=2)
dfpp_pca=pca.fit_transform(dfpp)
cntrd=pca.transform(kms.cluster_centers_)
plt.scatter(dfpp_pca[:,0],dfpp_pca[:,1], c=clusters, cmap='viridis')
plt.scatter(cntrd[:,0], cntrd[:,1], s=200, c='red', marker='x', label='centroid')
plt.legend()
plt.show()
