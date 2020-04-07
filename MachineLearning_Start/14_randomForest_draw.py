# =============================================================================
# LIBRERIAS
# =============================================================================

from sklearn.datasets import load_digits

datos = load_digits(return_X_y=True)

X = datos[0]
y = datos[1]

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.166, stratify=y, random_state=6)

miStdScaler = StandardScaler()
X_train = miStdScaler.fit_transform(X_train)
X_test = miStdScaler.transform(X_test)

miPCA = PCA(n_components=40)
miPCA.fit(X_train)
X_train_pca = miPCA.transform(X_train)
X_test_pca = miPCA.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

miModelo = RandomForestClassifier()

# =============================================================================
# CV
# =============================================================================

from sklearn.model_selection import GridSearchCV

migrid = {'n_estimators':[10,30,50,100]}

migscv = GridSearchCV(estimator=miModelo,param_grid=migrid,scoring='accuracy',cv=10,verbose=2)

migscv.fit(X_train_pca,y_train)

midtopt = migscv.best_estimator_

midtopt.fit(X_train_pca,y_train)

y_pred = midtopt.predict(X_test_pca)

from sklearn.metrics import accuracy_score

print(100*accuracy_score(y_test,y_pred))

midtopt.feature_importances_

sum(midtopt.feature_importances_)

from matplotlib import pyplot as plt

plt.bar(range(40),midtopt.feature_importances_)

'''
from matplotlib import pyplot as plt
plt.subplot(121)

plt.scatter(X_train_pca[:,0],X_train_pca[:,1],s=80,c=y_train)

X_test_pca = miPCA.transform(X_test)
plt.scatter(X_test_pca[:,0],X_test_pca[:,1],s=80,marker='s',c=y_test)

plt.subplot(111)
plt.scatter(X_train_pca[:,0],X_train_pca[:,1],s=80,c=y_train)
plt.scatter(X_test_pca[:,0],X_test_pca[:,1],s=80,marker='s',c=y_pred)

import numpy as np

x_min, x_max = X_train_pca[:, 0].min(), X_train_pca[:, 0].max()
y_min, y_max = X_train_pca[:, 1].min(), X_train_pca[:, 1].max()
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = midtopt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


plt.contourf(xx, yy, Z, alpha=0.3)
'''
