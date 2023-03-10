import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import numpy as np

df = pd.read_csv("F:\\Setups\\py-master\\py-master\\ML\\7_logistic_reg\\Exercise\\HR_comma_sep.csv")
ohe = ColumnTransformer([("one_hot_encoding", OneHotEncoder(), [3])], remainder='passthrough')
le = LabelEncoder()

df["salary"] = le.fit_transform(df["salary"])
X = df[["satisfaction_level", "average_montly_hours", "promotion_last_5years", "salary"]].values
X = np.array(ohe.fit_transform(X), dtype='float')

X = X[:,:2]
y = df["left"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
reg = LogisticRegression()
reg.fit(X_train, y_train)
print(reg.score(X_test, y_test))