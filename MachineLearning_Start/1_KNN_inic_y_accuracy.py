# =============================================================================
# LIBRERIAS
# =============================================================================

from sklearn.datasets import load_digits

datos = load_digits(return_X_y=True)

X = datos[0]
y = datos[1]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.166, stratify=y, random_state=6)

from sklearn.neighbors import KNeighborsClassifier

miknn = KNeighborsClassifier(n_neighbors=5,weights='uniform',metric='minkowski',p=2)

miknn.fit(X_train,y_train)

y_pred = miknn.predict(X_test)

#accuracy = 100.*sum(y_pred==y_test)/len(y_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test,y_pred)