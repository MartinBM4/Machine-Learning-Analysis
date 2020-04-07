# =============================================================================
# LIBRERIAS
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

from sklearn.datasets import load_boston

datos = load_boston(return_X_y=True)

X = datos[0]
y = datos[1]

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.166, random_state=6)

miStdScaler = StandardScaler()
X_train = miStdScaler.fit_transform(X_train)
X_test = miStdScaler.transform(X_test)

# fit the model
# clf = svm.SVC(kernel='linear',gamma='auto')
clf = svm.SVR(kernel='rbf',C=100000,gamma=100)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from matplotlib import pyplot as plt

plt.scatter(y_test,y_pred)
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)])



