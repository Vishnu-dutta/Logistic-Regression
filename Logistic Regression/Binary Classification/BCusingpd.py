import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("F:\\Setups\\py-master\\py-master\\ML\\7_logistic_reg\\Exercise\\HR_comma_sep.csv")

subdf = df[["satisfaction_level", "average_montly_hours", "promotion_last_5years", "salary"]]
# print(subdf.head())

salary_dummies = pd.get_dummies(subdf.salary, prefix="salary")
df_with_dummies = pd.concat([subdf, salary_dummies], axis="columns")

df_with_dummies.drop(["salary"], axis="columns", inplace=True)
X = df_with_dummies
y = df["left"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

reg = LogisticRegression()
reg.fit(X_train,y_train)
print(reg.score(X_test,y_test))