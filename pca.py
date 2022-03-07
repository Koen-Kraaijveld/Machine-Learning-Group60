import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

digits = pd.read_csv('data/train.csv')

x = digits.iloc[:, 1:]
y = digits.loc[:, 'label']

n_components = 12
pca = PCA(n_components)
components = pca.fit_transform(x)

columns = []
for i in range(n_components):
    columns.append('PC' + str(i+1))

digits_pca_df = pd.DataFrame(data=components, columns=columns)
digits_pca = pd.concat([digits[['label']], digits_pca_df], axis=1)

# variance = pca.explained_variance_ratio_
# plt.scatter(columns, variance)
# plt.plot(columns, variance)
#
# plt.title('Variance of each Principal Component')
# plt.xlabel('Component')
# plt.ylabel('Variance')
#
# plt.show()

