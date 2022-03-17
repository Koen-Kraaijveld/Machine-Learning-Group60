import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt
from pca import digits_pca

# digits = pd.read_csv('data/train.csv')

x = digits_pca.iloc[:, 1:]
y = digits_pca.loc[:, 'label']

param_grid = {
    "criterion":['gini','entropy'],
    "min_samples_split":range(2,10),
    "min_samples_leaf":range(3,10)
}
tree = DecisionTreeClassifier()
grid = GridSearchCV(tree, param_grid, cv=4, scoring='accuracy', return_train_score=False,verbose=1,n_jobs=-1)
grid_search=grid.fit(x, y)
print(grid_search.best_params_)
accuracy = grid_search.best_score_ *100
print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy) )
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

print(len(x_train))
print(len(y_train))

print(len(x_val))
print(len(y_val))

tree = grid.best_estimator_
tree.fit(x_train, y_train)
predicted = tree.predict(x_val)

scores = cross_val_score(tree, x_val, y_val, cv=5)
print('scores per fold ', scores)
print('mean score    ', np.mean(scores))
print('standard dev. ', np.std(scores))

print(classification_report(y_val, predicted))

confusion_matrix = plot_confusion_matrix(tree, x_val, y_val)
confusion_matrix.figure_.suptitle("Decision Tree Confusion Matrix")
plt.show()