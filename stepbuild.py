import pyBigWig as bw
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

path = "/home/vzscyborg/datasets/barret.bw"
fl = bw.open(path)

for i, j in fl.chroms().items(): 
    p = np.arange(0, j, 100000)  
    values = [
        fl.stats(i, pos, min(pos + 100000, j), type="mean")[0] or 0
        for pos in p
    ]

    heatmap_data = np.array(values).reshape(-1, 1)

    plt.figure(figsize=(6, 10))
    sns.heatmap(heatmap_data, cmap="viridis", linewidths=0.5)
    plt.title(f"Heatmap of {i}")
    plt.xlabel("Signal Intensity")
    plt.ylabel("Genomic Bins")
    plt.show()

fl.close()
