import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold

true_gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=0).fit(np.random.rand(10, 2))

sample_sizes = np.array([10, 100, 1000, 10000])
for sample_size in sample_sizes:
    samples, labels = true_gmm.sample(sample_size)
    best_n_components_list = []
    for i in range(50):
        kf = KFold(n_splits=10, shuffle=True)
        best_acc = -np.inf
        best_n_components = 0
        for n_components in range(1, 7):
            acc = -1e6
            for train_index, test_index in kf.split(samples):
                X_train, X_test = samples[train_index], samples[test_index]
                y_train, y_test = labels[train_index], labels[test_index]
                gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0).fit(X_train)
                acc += gmm.score(X_test)
            if acc > best_acc:
                best_acc = acc
                best_n_components = n_components
        best_n_components_list.append(best_n_components)

    best_n_components_list = np.array(best_n_components_list)
    plt.figure()
    plt.hist(best_n_components_list, bins=6, range=(0.5, 6.5), rwidth=0.5)
    plt.title('Histogram of the best n_components for {} samples'.format(sample_size))
    plt.xlabel('n_components')
    plt.ylabel('Frequency')
    plt.savefig('res_q2/' + str(sample_size) + '.png')



    
