import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

digits = pd.read_csv('data/train.csv')

x = digits.iloc[:, 1:]
y = digits.loc[:, 'label']

figure = plt.figure()

for i in range(0, 9):
    ax = figure.add_subplot(3, 3, i + 1)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_title("Digit with label " + str(y.loc[i]))
    img = x.loc[i].values.reshape(28, 28)
    plt.imshow(img, cmap='gray')

plt.show()

pca = PCA(n_components=2)
projected = pca.fit_transform(x)
plt.scatter(projected[:, 0], projected[:, 1], c=y, cmap="Paired", s=5)
plt.xlabel("Principal component #1 values")
plt.ylabel("Principal component #2 values")
plt.colorbar()
plt.title("Training data projected onto first 2 principal components")

plt.show()
