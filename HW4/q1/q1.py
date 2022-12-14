import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from sklearn.mixture import GaussianMixture
from mlp import MLP
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm


def map_classifier(x_set, m, c):
    likelihood_ratio = np.zeros((x_set.shape[0], m.shape[0]))
    for i in range(m.shape[0]):
        likelihood_ratio[:, i] = mvn.pdf(x_set, m[i], c[i])
    return np.argmax(likelihood_ratio, axis=1)
    

m = np.array([[-2, -2, -2], [1, 1, 1], [-1, -1, -1], [2, 2, 2]])
c = np.array([[[1.7, -0.5, 0.9], [0.5, 1.8, -0.5], [0.9, -0.5, 1.8]], 
                [[1.7, -0.5, 0.9], [0.5, 1.8, -0.5], [0.9, -0.5, 1.8]], 
                [[1.7, -0.5, 0.9], [0.5, 1.8, -0.5], [0.9, -0.5, 1.8]], 
                [[1.7, -0.5, 0.9], [0.5, 1.8, -0.5], [0.9, -0.5, 1.8]]])
c *= 0.5
p = np.array([0.25 , 0.25, 0.25, 0.25])
sample_sizes = np.array([100, 200, 500, 1000, 2000, 5000, 10000, 100000])
best_hidden_dim_list = []
test_size = 100000
test_labels = np.random.choice([0, 1, 2, 3], test_size, p=p)
test_samples = np.zeros((test_size, 3))
for i, test_label in enumerate(test_labels):
    test_samples[i] = np.random.multivariate_normal(m[test_label], c[test_label])

for sample_size in sample_sizes:
    labels = np.random.choice([0, 1, 2, 3], sample_size, p=p)
    samples = np.zeros((sample_size, 3))

    for i, label in enumerate(labels):
        samples[i] = np.random.multivariate_normal(m[label], c[label])

    X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=0.1, random_state=0)

    best_acc = 0
    best_hidden_dim = 0 

    for hidden_dim in range(1, 256):
        clf = MLP(hidden_dim)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        if acc > best_acc:
            best_acc = acc
            best_hidden_dim = hidden_dim

    best_hidden_dim_list.append(best_hidden_dim)
    print(sample_size)
    print('Best accuracy: {:.2f}%'.format(best_acc * 100))
    print('Best number of hidden units: {}'.format(best_hidden_dim))

    # calculate the error rate of the best model

    clf = MLP(best_hidden_dim)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(test_samples)
    y_pred = np.argmax(y_pred, axis=1)
    p_error = np.sum(y_pred != test_labels) / test_size
    print('Error rate for MLP classifier: {:.2f}%'.format(p_error * 100))
    y_pred = map_classifier(test_samples, m, c)
    p_error = np.sum(y_pred != test_labels) / test_size
    print('Error rate for MAP classifier: {:.2f}%'.format(p_error * 100))

plt.plot(sample_sizes, best_hidden_dim_list)
plt.xlabel('Number of training samples')
plt.ylabel('Number of hidden units')
plt.savefig('q1.png')





