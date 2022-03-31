import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

digits_train = pd.read_csv('../data/train.csv')

x_train = digits_train.iloc[:, 1:] / 255

train_pca = PCA().fit(x_train)

x = np.arange(0, 50, step=1)
y = train_pca.explained_variance_[0:50]

plt.plot(x, y, '-o')
plt.show()