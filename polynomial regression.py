import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split as tts

np.random.seed(100)
x=np.linspace(0,10,100).reshape(-1,1)
y=np.sin(x).ravel()+np.random.normal(0,0.15,x.shape[0])

xtrain,xtest,ytrain,ytest=tts(x,y, test_size=0.2, random_state=42)

degrees=[1,3,10]

for i,d in enumerate(degrees,1):
    plt.subplot(1,3,i)
    model=make_pipeline(PolynomialFeatures(d), LinearRegression())
    model.fit(xtrain,ytrain)
    ypred=model.predict(xtest)

    plt.scatter(xtest,ytest, color='blue', label='actual' )
    plt.plot(xtest,ytest, color='red', label=f'Degree {d}')
    plt.legend()
plt.show()