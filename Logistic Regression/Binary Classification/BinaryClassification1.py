import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("F:\\Setups\\py-master\\py-master\\ML\\7_logistic_reg\\Exercise\\HR_comma_sep.csv")
# print(df)

'''
plotting graph of salary vs left. Used LabelEncoder to deal with the numeric data of salary
'''

le = preprocessing.LabelEncoder()
ohe = ColumnTransformer([("one_hot_encoding", OneHotEncoder(), [0])], remainder="passthrough")

X = df[["average_montly_hours"]]
y = df[["left"]]
plt.scatter(X, y)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=4)
reg = LogisticRegression()  # creating an object of this class as 'reg'
reg.fit(X_train, y_train)


def best_score(x, y):
    for i in range(30):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)
        reg.fit(X_train, y_train)
        a = reg.score(X_test, y_test)
        print("score: {}, iteration: {}".format(a, i))


def salary_left_relation():
    df["salary"] = le.fit_transform(df["salary"])

    X = df[["salary"]]
    # print(X)
    y = df[["left"]]
    plt.scatter(X, y)
    plt.show()


def department_left_relation():
    df["Department"] = le.fit_transform(df["Department"])

    X = df[["Department"]]
    y = df[["left"]]
    plt.scatter(X, y)
    plt.show()

department_left_relation()
salary_left_relation()
best_score(X,y)
print(reg.score(X_test, y_test))            # accuracy of the test subject
