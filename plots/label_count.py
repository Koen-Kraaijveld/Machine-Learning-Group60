import pandas as pd
import matplotlib.pyplot as plt

digits = pd.read_csv('../data/train.csv')

label = digits.loc[:, 'label']
count = label.value_counts().sort_index().plot(kind='barh')

plt.title('Occurrences of each digit in the dataset')
plt.ylabel('Digit label')
plt.xlabel('Number of occurrences')

plt.show()
