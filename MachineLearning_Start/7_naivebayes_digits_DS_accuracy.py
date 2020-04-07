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
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.166, stratify=y, random_state=6)

miStdScaler = StandardScaler()
X_train = miStdScaler.fit_transform(X_train)
X_test = miStdScaler.transform(X_test)

#miPCA = PCA(n_components=40)
#miPCA.fit(X_train)
#X_train_pca = miPCA.transform(X_train)
#X_test_pca = miPCA.transform(X_test)

from sklearn.naive_bayes import GaussianNB

mignb = GaussianNB()

# =============================================================================
# CV
# =============================================================================

mignb.fit(X_train,y_train)

y_pred = mignb.predict(X_test)

from sklearn.metrics import accuracy_score

print(100*accuracy_score(y_test,y_pred))
