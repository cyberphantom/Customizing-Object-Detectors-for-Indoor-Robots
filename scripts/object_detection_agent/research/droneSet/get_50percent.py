import numpy as np
import pandas as pd
np.random.seed(1)
from random import shuffle

#full_labels = pd.read_csv('shuffled/droneset_train_shuffled.csv')
#full_labels = pd.read_csv('shuffled/droneset_val_shuffled.csv')
full_labels = pd.read_csv('shuffled/droneset_test_shuffled.csv')

full_labels.head()

grouped = full_labels.groupby('image_name')

grouped.apply(lambda x: len(x)).value_counts()

gb = full_labels.groupby('image_name')

grouped_list = [gb.get_group(x) for x in gb.groups]

#train_index = np.random.choice(len(grouped_list), size=1782, replace=False) #3564 train
#val_index = np.random.choice(len(grouped_list), size=357, replace=False) #714 train
test_index = np.random.choice(len(grouped_list), size=234, replace=False) #469 train

#train = pd.concat([grouped_list[i] for i in train_index]) 
#val = pd.concat([grouped_list[i] for i in val_index]) 
test = pd.concat([grouped_list[i] for i in test_index]) 
print(len(test))

test.to_csv('annotations/test/test_shuffled_50p.'
             'csv', index=None)

