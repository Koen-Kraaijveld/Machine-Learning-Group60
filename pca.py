import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

digits = pd.read_csv('data/train.csv')

x = digits.iloc[:, 1:]
y = digits.loc[:, 'label']

numberOfComponents = 20
pca = PCA(n_components=numberOfComponents)
principalComponents = pca.fit_transform(x)

# principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
# finalDf = pd.concat([principalDf, digits[['label']]], axis=1)
# print(finalDf)

columns = []
for i in range(numberOfComponents):
    columns.append('#' + str(i+1))

variance = pca.explained_variance_ratio_
plt.scatter(columns, variance)
plt.plot(columns, variance)

plt.title('Variance of each Principal Component')
plt.xlabel('Component')
plt.ylabel('Variance')

plt.show()
