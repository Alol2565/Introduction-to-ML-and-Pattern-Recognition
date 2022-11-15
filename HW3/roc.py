import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from sklearn.mixture import GaussianMixture

def classifier(x_set, threshold, m, c, w):
    likelihood_ratio = mvn.pdf(x_set, m[2], c[2]) / (w[0] * mvn.pdf(x_set, m[0], c[0]) + w[1] * mvn.pdf(x_set, m[1], c[1]))
    return np.where(likelihood_ratio > threshold, 1, 0)

def roc_plot(samples, labels, p, m , c, w, name):
    sample_size = np.shape(samples)[0]
    thresholds = np.arange(0, 100, 0.01)
    results = np.array(list(map(classifier, [samples] * len(thresholds), thresholds, [m] * len(thresholds), [c] * len(thresholds), [w] * len(thresholds))))
    tp = np.sum(np.logical_and(results == 1, labels == 1), axis=1)
    fp = np.sum(np.logical_and(results == 1, labels == 0), axis=1)
    tn = np.sum(np.logical_and(results == 0, labels == 0), axis=1)
    fn = np.sum(np.logical_and(results == 0, labels == 1), axis=1)
    p_error = (fn + fp) / sample_size
    plt.figure()
    plt.plot(thresholds, p_error)
    plt.xlabel('Threshold')
    plt.ylabel('P-Error')
    plt.title(name + ' P-Error vs Threshold')
    plt.grid()
    plt.savefig(name + 'p_error.png')
    # find the best threshold
    best_threshold = thresholds[np.argmin(p_error)]
    print(name + ' Best Threshold: ', best_threshold)
    print(name + ' P-Error: ', p_error[np.argmin(p_error)])
    tp_rate = tp / np.sum(labels == 1)
    fp_rate = fp / np.sum(labels == 0)


    tp_theoritical = 0
    fp_theoritical = 0
    fn_theoritical = 0
    results = classifier(samples, p[0]/p[1], m, c, w)
    tp_theoritical = np.sum(np.logical_and(results == 1, labels == 1))
    fp_theoritical = np.sum(np.logical_and(results == 1, labels == 0))
    fn_theoritical = np.sum(np.logical_and(results == 0, labels == 1))
    print(name + 'Theoritical P-Error: ', (fn_theoritical + fp_theoritical) / sample_size)
    print(name + 'Theoritical Threshold: ', p[0]/p[1])
    plt.figure()
    plt.plot(fp_rate, tp_rate, fp_rate[int(np.argmin(p_error))], tp_rate[int(np.argmin(p_error))], 'gx', fp_theoritical / np.sum(labels == 0), tp_theoritical / np.sum(labels == 1), 'rx')
    plt.xlabel('False Alarm Rate')
    plt.ylabel('Hit Rate')
    plt.title(name + ' ROC')
    plt.grid()
    plt.savefig(name + 'roc.png')


    plt.figure()
    plt.plot(samples[labels == 0, 0], samples[labels == 0, 1], 'bx', samples[labels == 1, 0], samples[labels == 1, 1], 'rx')
    x = np.arange(-5, 10, 0.1)
    y = np.arange(-5, 10, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = classifier(np.array([X[i, j], Y[i, j]]), best_threshold, m, c, w)
    plt.contour(X, Y, Z, levels=[0.5])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(name + 'Decision Boundary')
    plt.savefig(name+  'decision_boundary.png')

def mle_estimation(samples, labels):
    est_p = np.array([np.sum(labels == 0) / labels.shape[0], np.sum(labels == 1) / labels.shape[0]])
    est_m = np.zeros((3, 2))
    est_c = np.zeros((3, 2, 2))
    est_w = np.zeros(2)
    GMM_1 = GaussianMixture(n_components=1, random_state=0).fit(samples[labels == 1])
    GMM_0 = GaussianMixture(n_components=2, random_state=0).fit(samples[labels == 0])
    [est_m[0], est_m[1], est_m[2]] = GMM_0.means_[0], GMM_0.means_[1], GMM_1.means_[0]
    [est_c[0], est_c[1], est_c[2]] = GMM_0.covariances_[0], GMM_0.covariances_[1], GMM_1.covariances_[0]
    est_w[0] = GMM_0.weights_[0]
    est_w[1] = GMM_0.weights_[1]
    return est_p, est_m, est_c, est_w