import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt
from pca_2 import digits_train_pca, digits_test_pca

# digits_test = pd.read_csv('data/test.csv')

x = digits_train_pca.iloc[:, 1:]
y = digits_train_pca.loc[:, 'label']

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

svm = SVC(C=10)

svm.fit(x_train, y_train)
predicted_y = svm.predict(x_val)

scores = cross_val_score(svm, x_val, y_val, cv=5)
print('scores per fold ', scores)
print('mean score    ', np.mean(scores))
print('standard dev. ', np.std(scores))

print(classification_report(y_val, predicted_y))

confusion_matrix = plot_confusion_matrix(svm, x_val, y_val, normalize='true', values_format='.2f')
confusion_matrix.figure_.suptitle("Support Vector Machine Confusion Matrix")
plt.show()
