from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn

load = load_iris()
print(dir(load))
# print(load.DESCR)
# print(load.data)
# a = load.data
# print(len(a))         calculation of rows
# print(len(a[0]))      calculation of columns
# print(load.target_names)

'''
We have target and target_name. In this dataset the string attribute is already LabelEncoded. We have 'target_name' and 
its LabelEncoded value as 'target'. 
Iris_Setosa         as 0 
Iris_Versicolour    as 1
Iris_Virginica      as 2
'''

X = load.data
y = load.target

reg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50)
reg.fit(X_train, y_train)

print(reg.score(X_test, y_test))

print(load.target[50])  # 1 meaning versicolour
print(load.data[50])    # [7.0, 3.2, 4.7, 1.4]

print(reg.predict([[7.0, 3.2, 4.7, 1.4]]))  # changing 7. to 7 or 7.0 doesn't create any errors

y_predicted = reg.predict(X_test)
cm = confusion_matrix(y_test, y_predicted)
print(cm)

' Here employing heatmap for visualization of the data'

plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()