import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from mlp import MLP
from sklearn.model_selection import KFold


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
c *= 0.4
p = np.array([0.25 , 0.25, 0.25, 0.25])
sample_sizes = np.array([100, 200, 500, 1000, 2000, 5000])
best_hidden_dim_list = []
test_size = 100000
test_labels = np.random.choice([0, 1, 2, 3], test_size, p=p)
test_samples = np.zeros((test_size, 3))
for i, test_label in enumerate(test_labels):
    test_samples[i] = np.random.multivariate_normal(m[test_label], c[test_label])
p_error_list = []
for sample_size in sample_sizes:
    labels = np.random.choice([0, 1, 2, 3], sample_size, p=p)
    samples = np.zeros((sample_size, 3))
    for i, label in enumerate(labels):
        samples[i] = np.random.multivariate_normal(m[label], c[label])
    best_acc = 0
    best_hidden_dim = 0 
    for hidden_dim in range(1, 512, 20):
        kf = KFold(n_splits=10)
        acc = 0
        for train_index, val_index in kf.split(samples):
            X_train, X_val = samples[train_index], samples[val_index]
            y_train, y_val = labels[train_index], labels[val_index]
            clf = MLP(hidden_dim)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            y_pred = np.argmax(y_pred, axis=1)
            acc += np.sum(y_pred == y_val) / len(y_val)
        acc /= 10
        if acc > best_acc:
            best_acc = acc
            best_hidden_dim = hidden_dim
    best_hidden_dim_list.append(best_hidden_dim)
    print(sample_size)
    print('Best accuracy: {:.2f}%'.format(best_acc * 100))
    print('Best number of hidden units: {}'.format(best_hidden_dim))

    clf = MLP(best_hidden_dim)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(test_samples)
    y_pred = np.argmax(y_pred, axis=1)
    p_error = np.sum(y_pred != test_labels) / test_size
    p_error_list.append(p_error)
    print('Error rate for MLP classifier: {:.2f}%'.format(p_error * 100))
    y_pred = map_classifier(test_samples, m, c)
    p_error = np.sum(y_pred != test_labels) / test_size
    print('Error rate for MAP classifier: {:.2f}%'.format(p_error * 100))

plt.figure()
plt.semilogx(sample_sizes, p_error_list)
plt.semilogx(sample_sizes, np.ones(len(sample_sizes)) * 0.16, '--')
plt.ylim(0, 1)
plt.xlim(100, 5000)
plt.xlabel('Number of training samples')
plt.ylabel('Error rate')
plt.savefig('q1_error.png')

plt.figure()
plt.bar(list(map(str, sample_sizes)), best_hidden_dim_list)
plt.xlabel('Number of training samples')
plt.ylabel('Number of hidden units')
plt.savefig('q1_hidden.png')

with open('q1.csv', 'w') as f:
    f.write('sample_size,hidden_dim,p_error')
    for i in range(len(sample_sizes)):
        f.write('\n{},{},{}'.format(sample_sizes[i], best_hidden_dim_list[i], p_error_list[i]))





