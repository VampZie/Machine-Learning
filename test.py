import pyBigWig as bw
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

path = "/home/vzscyborg/datasets/barret.bw"
fl = bw.open(path)

for chrom in fl.chroms():
    chrom_length = fl.chroms()[chrom]
    grid_size = max(1, chrom_length // 2500) 

    values = [] 
    for i in range(0, chrom_length, grid_size):
        value = fl.stats(chrom, i, min(i + grid_size, chrom_length), type="mean")[0]
        values.append(value if value is not None else 0) 

    if len(values) > 1:
        values = np.array(values).reshape(-1, 1)

        linkage_matrix = linkage(values, method='ward')

        plt.figure(figsize=(8, 6))
        dendrogram(linkage_matrix)
        plt.title(f"Hierarchical Clustering Dendrogram for {chrom}")
        plt.xlabel("Genomic Bins")
        plt.ylabel("Distance")
    else:
        print(f"Skipping {chrom} due to insufficient data.")

        
    coph_corr, _ = cophenet(linkage_matrix, pdist(values))
    print(f"Cophenetic Correlation Coefficient for {chrom}:", coph_corr)

fl.close()
