import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bioframe as bf
import seaborn as sns
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
path='/home/vzscyborg/datasets/GSM2559141_HS5-Barretts.bw'
s='chr'
for i in range(1,31):
    df=bf.read_bigwig(path,chrom=(s+str(i)))
    print(df)
