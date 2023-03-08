from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import pandas as pd

df = pd.read_csv("F:\\Setups\\py-master\\py-master\\ML\\7_logistic_reg\\insurance_data.csv")
# print(df)

X = df[["age"]]
y = df["bought_insurance"]
plt.scatter(X, y)
plt.show()
'''
reg = LogisticRegression()
reg.fit(X,y)
print(reg.predict([[24]]))
'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
reg = LogisticRegression()              # creating an object of this class as 'reg'
reg.fit(X_train, y_train)
'''
In test and train after splitting, we create model with trained part of the whole data and further use it to predict for
the test data. Predicted X_test is basically y_test. If for that predicted(X_test) == y_test then accuracy is 1.0 . 

'''
print(reg.predict(X_test))              # predicting y_test based on values of X_test
print(y_test)                           # further actually checking y_value
print(X_test)                           # inputted X_test values for prediction

# print(y_test)
print(reg.score(X_test, y_test))

print(reg.predict_proba(X_test))        # probability of customer buying or not-buying insurance(classed as NO - YES)

print(reg.predict([[69]]))                  # predicting if 69 and 18 YO will buy insurance or not
print(reg.predict([[18]]))


'''
def best_score(x,y):
    for i in range(30):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        reg = LogisticRegression()
        reg.fit(X_train, y_train)
        a = (reg.score(X_train, y_train))
        print("score: {}, iteration: {}".format(a,i))


best_score(X,y)
'''
