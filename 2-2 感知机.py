import sklearn
from sklearn.linear_model import Perceptron

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target

df.columns = [
    'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
]

data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:,:-1], data[:,-1]
y = np.array([1 if i == 1 else -1 for i in y])

clf = Perceptron(fit_intercept=True, 
                 max_iter = 1000, 
                 tol=None,
                 shuffle=True)
clf.fit(X, y)

print(clf.coef_)
print(clf.intercept_)

# 画布大小
plt.figure(figsize=(10, 10))

# 中文标题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title('鸢尾花线性数据示例')

plt.scatter(data[:50, 0], data[:50, 1], c = 'b', label = 'Iris-setosa',)
plt.scatter(data[50:100, 0], data[50:100, 1], c = 'orange', label = 'Iris-versicolor')

# 画感知机的线
x_points = np.arange(4, 8)
y_ = -(clf.coef_[0][0]*x_points + clf.intercept_) / clf.coef_[0][1]
plt.plot(x_points, y_)

plt.legend()
plt.grid(False)
plt.xlabel('speal length')
plt.ylabel('speal width')
plt.legend()
plt.show()