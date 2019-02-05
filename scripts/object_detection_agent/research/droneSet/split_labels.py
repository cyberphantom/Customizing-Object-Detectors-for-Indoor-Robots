import numpy as np
import pandas as pd
np.random.seed(1)

full_labels = pd.read_csv('labels/tennis_racket.csv')

full_labels.head()

grouped = full_labels.groupby('image_name')

grouped.apply(lambda x: len(x)).value_counts()

gb = full_labels.groupby('image_name')

grouped_list = [gb.get_group(x) for x in gb.groups]

train_index = np.random.choice(len(grouped_list), size=321, replace=False)
val_test_index  = np.setdiff1d(list(range(427)), train_index)


val_index = np.random.choice(val_test_index, size=64, replace=False)
test_index  = np.setdiff1d(val_test_index, val_index)



train = pd.concat([grouped_list[i] for i in train_index])
val = pd.concat([grouped_list[i] for i in val_index])
test = pd.concat([grouped_list[i] for i in test_index])

print(len(train), len(val), len(test))

train.to_csv('annotations/tennis_racket_train.'
             'csv', index=None)
val.to_csv('annotations/tennis_racket_val.csv', index=None)
test.to_csv('annotations/tennis_racket_test.csv', index=None)
