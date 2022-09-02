import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

df_train = pd.read_csv('datasets/classifier_multiclass/fruits.csv')

seed = 10

x_train = df_train.drop('fruit_name', axis = 1)

df_train['fruit_name'] = df_train['fruit_name'].replace(to_replace=['apple', 'orange', 'lemon'], value = 0)
df_train['fruit_name'] = df_train['fruit_name'].replace(to_replace=['mandarin'], value = 1)
y_train = df_train['fruit_name'].values

logisticregression_model = LogisticRegression(random_state=seed).fit(x_train, y_train)
train_score = logisticregression_model.score(x_train, y_train)
print(logisticregression_model.coef_)
print(logisticregression_model.intercept_)

print('logisticRegression accuracy train:', train_score)