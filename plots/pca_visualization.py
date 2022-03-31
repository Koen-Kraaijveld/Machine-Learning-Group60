import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pca_2 import train_pca as train_pca_2

digits_train = pd.read_csv('../data/train.csv')

x_train = digits_train.iloc[:, 1:] / 255

train_pca = PCA().fit(x_train)

plt.rcParams["figure.figsize"] = (12, 6)

fig, ax = plt.subplots()
xi = np.arange(1, 785, step=1)
y = np.cumsum(train_pca.explained_variance_ratio_)

plt.ylim(0.0, 1.1)
plt.plot(xi, y)

plt.xlabel('Number of Components')
plt.xticks(np.arange(0, 785, step=50))  # change from 0-based array index to 1-based human-readable label
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')
plt.margins(x=0)


for line in ax.get_lines():
    x, y = line.get_data()

    try:
        ind_90 = np.argwhere(y >= 0.90)[0][0]
        plt.plot(x[ind_90], y[ind_90], 'mo')
        plt.axhline(y=0.90, color='m', linestyle='dashed')
        ax.annotate('90% variance: ' + str(ind_90), xy=(ind_90, 0.90), xytext=(ind_90, 0.82), fontsize=12)

        ind_95 = np.argwhere(y >= 0.95)[0][0]
        plt.plot(x[ind_95], y[ind_95], 'ro')
        plt.axhline(y=0.95, color='r', linestyle='dashed')
        ax.annotate('95% variance: ' + str(ind_95), xy=(ind_95, 0.95), xytext=(ind_95, 1.03), fontsize=12)

        ind_99 = np.argwhere(y >= 0.99)[0][0]
        plt.plot(x[ind_99], y[ind_99], 'go')
        plt.axhline(y=0.99, color='g', linestyle='dashed')
        ax.annotate('99% variance: ' + str(ind_99), xy=(ind_99, 0.99), xytext=(ind_99, 1.03), fontsize=12)

    except TypeError:
        print("")

ax.grid(axis='x')
plt.show()
