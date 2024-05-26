from sklearn.decomposition import PCA
import os
import numpy as np

data = []

with open("desc.txt") as f:
    content = f.readlines()
    for line in content:
        line = line.split(",")
        v = [float(x) for x in line]
        data.append(np.array(v))
data = np.array(data)


pca = PCA(n_components=36)
data = pca.fit_transform(data)
mean = data.mean(axis=0)

np.set_printoptions(threshold=np.inf)
print(np.argmax(data / np.sqrt(pca.explained_variance_), axis=1))

np.save("mean.npy", mean)
np.save("pca.npy", pca.components_)
np.save("variance.npy", pca.explained_variance_)
