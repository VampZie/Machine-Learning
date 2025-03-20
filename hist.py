import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

px=[0.05,0.1,0.2,0.35,0.2,0.1]
f=[0,1,2,3,4,5]

plt.bar(f,px,width=0.5,align='center')
plt.xlabel('f')
plt.ylabel('px')
plt.show()