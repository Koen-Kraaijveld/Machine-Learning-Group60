import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

digits_train = pd.read_csv('data/train.csv')

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
# plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

for line in ax.get_lines():
    x, y = line.get_data()
    ind = np.argwhere(y >= 0.95)[0][0]  # first index where y is larger than y_special
    print(ind)
    ax.text(x[ind], 95, f' {x[ind]:.1f}', ha='left', va='top')  # maybe color=line.get_color()

ax.grid(axis='x')
plt.show()
