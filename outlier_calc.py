from scipy.stats import chi2
from sklearn.covariance import EmpiricalCovariance, MinCovDet
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from hyperparameters import hyperparameters

def compute_roc_curve(scores, labels):

    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    area_under_curve = auc(false_positive_rate, true_positive_rate)
    
    return false_positive_rate, true_positive_rate, area_under_curve

def plot_roc_curve(scores, labels):
    plt.figure(figsize=(6, 6))
    for key in scores.keys():
      false_positive_rate, true_positive_rate, area_under_curve = compute_roc_curve(scores[key],
                                                                                    labels[key])
      plt.plot(false_positive_rate, true_positive_rate,
            linewidth=3, label='{} (AUC={:0.3f})'.format(key, area_under_curve))
    plt.plot([0, 1], [0, 1], color='k', linestyle='--', linewidth=3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='center left', bbox_to_anchor=(1.04,0.5), borderaxespad=0)
    plt.grid(True)
    plt.show()

def estimate_covariance(X, hps = hyperparameters):
      
      if hps.covariance_method == "MinCovDet":
            cov = MinCovDet().fit(X)
            covariance = cov.covariance_
      if hps.covariance_method == "Emperical":
            cov = EmpiricalCovariance().fit(X)
            covariance = cov.covariance_
      covariance = covariance + np.identity(len(covariance)) * 1e-6
      return covariance

def mah(cov):
      return DistanceMetric.get_metric('mahalanobis', V=cov)

#MAH distance estimation
def mah_distance(signatures, hps = hyperparameters):
      
      covariance = estimate_covariance(signatures, hps)
      # Covariance matrix power of -1
      covariance_pm1 = np.linalg.matrix_power(covariance, -1)

      # Center point
      centerpoint = np.mean(signatures , axis=0)

      distances = []
      for val in signatures:
            p1 = val
            p2 = centerpoint
            distance = (p1-p2).T.dot(covariance_pm1).dot(p1-p2)
            distances.append(distance)
      distances = np.array(distances)

      certainty = [chi2.cdf(distances[i], len(signatures[0])) for i in range(len(distances))]
      return certainty



#Isolation Forest

def isolation_forest(clean_signatures, signatures, hps = hyperparameters):
      samp = 20
      est = 130
      feat = 1.0

      rng = np.random.RandomState(42)
      clf = IsolationForest(max_samples=samp, n_estimators=est, max_features= feat, random_state=rng, n_jobs = -1)
      clf.fit(clean_signatures) 
      
      outlier_predict = clf.predict(signatures)
      predict_label = []
      for i in range(len(outlier_predict)):
            if outlier_predict[i] == -1:
                  predict_label.append(1)
            else:
                  predict_label.append(0)

      return predict_label


#K nearest neighbours

def knn(X, hps = hyperparameters):
      if hps.knn_metric == 'minkowski':
             clf = LocalOutlierFactor(n_neighbors=hps.neighbours, metric='minkowski')

      if hps.knn_metric == 'mahalanobis':
            covariance = estimate_covariance(X, hps)
            covariance_pm1 = np.linalg.matrix_power(covariance, -1)
            clf = LocalOutlierFactor(n_neighbors=hps.neighbours, metric='mahalanobis', 
                               metric_params={'VI': covariance_pm1})
      outlier_predict = clf.fit_predict(X)
      negative_predict_label = clf.negative_outlier_factor_ 
      predict_label = []
      for label in negative_predict_label:
            predict_label.append(-label)
      
      return predict_label

def knn_predict(X, hps = hyperparameters):
      if hps.knn_metric == 'minkowski':
             clf = LocalOutlierFactor(n_neighbors=hps.neighbours, metric='minkowski')

      if hps.knn_metric == 'mahalanobis':
            covariance = estimate_covariance(X, hps)
            covariance_pm1 = np.linalg.matrix_power(covariance, -1)
            clf = LocalOutlierFactor(n_neighbors=hps.neighbours, metric='mahalanobis', 
                               metric_params={'VI': covariance_pm1})
      outlier_predict = clf.fit_predict(X)
      predict_label = []
      for i in range(len(outlier_predict)):
            if outlier_predict[i] == -1:
                  predict_label.append(1)
            else:
                  predict_label.append(0)

      return predict_label

# One class SVM

def one_class_SVM_unsupervised(X, hps = hyperparameters):
      clf = OneClassSVM().fit(X)
      return clf.score_samples(X)

def one_class_SVM_semisepervised(X, corpus, hps=  hyperparameters):
      clf = OneClassSVM().fit(corpus)
      return clf.score_samples(X)