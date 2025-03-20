import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

df=pd.DataFrame(pd.read_csv('/home/vzscyborg/datasets/given.csv'))
dfpp=StandardScaler().fit_transform(df.select_dtypes(include=['float64','int64']))

def elbow():
    wcss=[]
    for k in range(1,11):
        kms=KMeans(n_clusters=k, init='k-means++', random_state=50)
        kms.fit(dfpp)
        wcss.append(kms.inertia_)
    return wcss,kms

def plotting(wcss):
    plt.scatter(range(1,11),wcss, marker='o', color='green')
    plt.show()
    ok=5
    return ok

def km(wcss,kms,ok):
    kms=KMeans(n_clusters=ok, init='k-means++', random_state=50)
    clstr=kms.fit_predict(dfpp)
    df['cluster']=clstr
    pca=PCA(n_components=2)
    dfpp_pca=pca.fit_transform(dfpp)
    cntrd=pca.transform(kms.cluster_centers_)
    plt.scatter(dfpp_pca[:,0],dfpp_pca[:,1], c=clstr, cmap='viridis',alpha=0.6, label='KMeans with PCA')
    plt.scatter(cntrd[:,0], cntrd[:,1], s=100, marker='o', color='red')
    plt.legend()
    plt.show()


def minib(wcss,kms,ok):
    mb=MiniBatchKMeans(n_clusters=ok, batch_size=200, random_state=50)
    clstr=mb.fit_predict(dfpp)
    df['cluster']=clstr
    pca=PCA(n_components=2)
    dfpp_pca=pca.fit_transform(dfpp)
    cntrd=mb.cluster_centers_
    plt.scatter(dfpp_pca[:,0],dfpp_pca[:,1], c=clstr, cmap='viridis', alpha=0.6)
    plt.scatter(cntrd[:,0], cntrd[:,1], s=100, marker='o', color='red')
    plt.show()


def main():
    wcss,kms=elbow()
    ok=plotting(wcss)
    km(wcss,kms,ok)
    minib(wcss,kms,ok)

if __name__=='__main__':
    main()