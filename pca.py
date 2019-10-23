import numpy as np
import pandas as pd
from numpy.linalg import eig

arr = np.array([[1,2],[3,4],[5,6]])
at = arr.T

m1 = np.mean(at[0])
m2 = np.mean(at[1])

a = at[0]-m1
b = at[1]-m2

at = np.array([a,b])
arr_scale = at.T
cv = np.cov(at)

w = eig(cv)
#%%
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(arr)

pca.components_ #it gives the eigen values
pca.explained_variance_ #it gives the eigen vectors
#%%
t = arr*pca.explained_variance_
