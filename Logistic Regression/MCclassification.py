import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sn

digits = load_digits()
print(dir(digits))
# print(digits.data[0])
plt.gray()
# plt.imshow(digits.images[0])
# plt.show()
# print(digits.target[0:5])

X_train, X_test, y_train, y_test = train_test_split = train_test_split(digits.data, digits.target, test_size=0.2)
reg = LogisticRegression()
reg.fit(X_train, y_train)
print(reg.score(X_train, y_train))

# plt.imshow(digits.image[67])
# print(digits.target[67])
# print(reg.predict([digits.data[67]]))


y_predicted = reg.predict(X_test)
cm = confusion_matrix(y_test, y_predicted)
# print(cm)

plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()


