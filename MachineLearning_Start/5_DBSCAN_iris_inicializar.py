# =============================================================================
# LIBRERIAS
# =============================================================================

from sklearn.datasets import load_iris

datos = load_iris(return_X_y=True)

X = datos[0]
y = datos[1]

from sklearn.preprocessing import StandardScaler

miStdScaler = StandardScaler()
X = miStdScaler.fit_transform(X)

from sklearn.cluster import KMeans
miKmeans = KMeans(n_clusters=3)

miKmeans.fit(X)

from sklearn.cluster import DBSCAN
miDBSCAN = DBSCAN(eps=2,min_samples=2)

miDBSCAN.fit(X)