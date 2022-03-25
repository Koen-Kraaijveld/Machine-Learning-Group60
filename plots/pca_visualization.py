import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

plt.axhline(y=0.95, color='r', linestyle='dashed')
plt.axhline(y=0.99, color='g', linestyle='dashed')
# plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

for line in ax.get_lines():
    x, y = line.get_data()

    try:
        ind_95 = np.argwhere(y >= 0.95)[0][0]
        print("Number of components to achieve variance 0.95: " + str(ind_95))
        plt.plot(x[ind_95], y[ind_95], 'ro')
        ax.annotate('95% variance: ' + str(ind_95), xy=(ind_95, 0.95), xytext=(ind_95, 0.88), fontsize=12)

        ind_99 = np.argwhere(y >= 0.99)[0][0]
        print("Number of components to achieve variance 0.99: " + str(ind_99))
        plt.plot(x[ind_99], y[ind_99], 'go')
        ax.annotate('99% variance: ' + str(ind_99), xy=(ind_99, 0.99), xytext=(ind_99, 1.03), fontsize=12)

    except TypeError:
        print("")

ax.grid(axis='x')
plt.show()
