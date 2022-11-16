from sklearn.cluster import SpectralClustering
from sklearn import metrics
from sklearn.cluster import OPTICS
from sklearn.cluster import KMeans
import numpy as np

def evaluate(pred_labels, true_labels):
    return metrics.rand_score(true_labels, pred_labels)

def spectral_clustering(X):
    clustering = SpectralClustering(n_clusters=2, assign_labels='discretize').fit(X)
    return clustering.labels_


def kmeans(X):
    X = np.array(X)
    kmeans = KMeans(n_clusters=2).fit(X)
    return kmeans.labels_

def optics(X):
    X = np.array(X)
    clustering = OPTICS().fit(X)
    return clustering.labels_
