# =============================================================================
# LIBRERIAS
# =============================================================================

from sklearn.datasets import load_iris

datos = load_iris(return_X_y=True)

X = datos[0]
y = datos[1]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.166, stratify=y, random_state=6)

from sklearn.neighbors import KNeighborsClassifier
miknn = KNeighborsClassifier()

# =============================================================================
# CV
# =============================================================================

from sklearn.model_selection import GridSearchCV

migrid = {'n_neighbors':[1,3,5,7,9],'weights':['uniform','distance'],'p':[1,2,3,4,5]}

migscv = GridSearchCV(estimator=miknn,param_grid=migrid,scoring='accuracy',cv=3,verbose=2)

migscv.fit(X_train,y_train)

miknnopt = migscv.best_estimator_

miknnopt.fit(X_train,y_train)

y_pred = miknnopt.predict(X_test)

from sklearn.metrics import accuracy_score

print(100*accuracy_score(y_test,y_pred))

#migscv.best_estimator_
#migscv.score
#migscv.cv_results_
#kk= migscv.cv_results_





