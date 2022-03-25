import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

digits_train = pd.read_csv('../data/train.csv')
digits_test = pd.read_csv('../data/test.csv')

x_train = digits_train.iloc[:, 1:] / 255
y_train = digits_train.loc[:, 'label']

x_test = digits_test / 255

n_components = 330

train_pca = PCA(n_components)
train_components = train_pca.fit_transform(x_train)
test_components = train_pca.transform(x_test)

columns = []
for i in range(n_components):
    columns.append('PC' + str(i+1))


digits_train_pca_df = pd.DataFrame(data=train_components, columns=columns)
digits_train_pca = pd.concat([digits_train[['label']], digits_train_pca_df], axis=1)

digits_test_pca = pd.DataFrame(data=test_components, columns=columns)