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

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,y_pred,labels=range(10)))




