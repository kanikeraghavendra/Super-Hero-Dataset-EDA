import seaborn as sns
import numpy as np

data = sns.load_dataset('iris')
data = data.iloc[:, :4]

C = (data - data.T.mean(axis = 1))

np.dot(np.linalg.eig(np.cov(C.T))[1].T, C.T).T


from sklearn.decomposition import PCA

pca = PCA(4).fit(C)

pca.components_
pca.explained_variance_
pca.explained_variance_ratio_

pca.transform(C)[0]
