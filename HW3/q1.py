import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from sklearn.mixture import GaussianMixture
from roc import roc_plot
from roc import mle_estimation
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score 
p = np.array([0.6 , 0.4])
m = np.array([[5 , 0], [0, 4], [3, 2]])
c = np.array([[[4, 0], [0, 2]], [[1, 0], [0, 3]], [[2, 0], [0, 2]]])
w = np.array([0.5, 0.5])

sample_size = np.array([100, 1000, 10000, 20000])
p = np.array([0.6 , 0.4])

labels = np.random.choice([0, 1], size=np.sum(sample_size), p=p)
samples = np.zeros((np.sum(sample_size), 2))

for i, label in enumerate(labels):
    if label == 0:
        if(np.random.choice([0, 1], p=[0.5, 0.5]) == 0):
            samples[i] = np.random.multivariate_normal(m[0], c[0])
        else:
            samples[i] = np.random.multivariate_normal(m[1], c[1])
    if label == 1:
        samples[i] = np.random.multivariate_normal(m[2], c[2])


validation_samples = samples[-sample_size[3]:]
validation_labels = labels[-sample_size[3]:]
# roc_plot(validation_samples, validation_labels, p, m, c, w, 'Validation_')

# Part 2 

training_samples = samples[sample_size[0] + sample_size[1]:-sample_size[3]]
training_labels = labels[sample_size[0] + sample_size[1]:-sample_size[3]]
p, m, c, w = mle_estimation(training_samples, training_labels)
roc_plot(validation_samples, validation_labels, p, m, c, w, 'Training_0_')

training_samples = samples[sample_size[0] :- sample_size[2] - sample_size[3]]
training_labels = labels[sample_size[0] :- sample_size[2] - sample_size[3]]
p, m, c, w = mle_estimation(training_samples, training_labels)
roc_plot(validation_samples, validation_labels, p, m, c, w, 'Training_1_')

training_samples = samples[:sample_size[0]]
training_labels = labels[:sample_size[0]]
p, m, c, w = mle_estimation(training_samples, training_labels)
roc_plot(validation_samples, validation_labels, p, m, c, w, 'Training_2_')

training_samples = samples[sample_size[0] + sample_size[1]:-sample_size[3]]
training_labels = labels[sample_size[0] + sample_size[1]:-sample_size[3]]
model = LogisticRegression()
model.fit(training_samples, training_labels)
predicted_classes = model.predict(validation_samples)
accuracy = accuracy_score(validation_labels.flatten(),predicted_classes)
print(1 - accuracy)
# plot decision boundary visually with validation set
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
xy = np.vstack([X.ravel(), Y.ravel()]).T
Z = model.predict_proba(xy)[:, 1]
Z = Z.reshape(X.shape)
plt.figure()
plt.scatter(validation_samples[:, 0], validation_samples[:, 1], c=validation_labels, s=40, cmap='viridis')
plt.contour(X, Y, Z, [0.5], linewidths=2., colors='k')
plt.savefig('lrl.png')


poly = PolynomialFeatures(degree = 2, interaction_only=False, include_bias=False)
lr = LogisticRegression()
pipe = Pipeline([('polynomial_features',poly), ('logistic_regression',lr)])
pipe.fit(training_samples, training_labels)
# plot decision boundary visually with validation set
predicted_classes = pipe.predict(validation_samples)
accuracy = accuracy_score(validation_labels.flatten(),predicted_classes)
print(1 - accuracy)
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
xy = np.vstack([X.ravel(), Y.ravel()]).T
Z = pipe.predict_proba(xy)[:, 1].reshape(X.shape)
plt.figure()
plt.scatter(validation_samples[:, 0], validation_samples[:, 1], c=validation_labels, s=40, cmap=plt.cm.Spectral)
plt.contour(X, Y, Z, [0.5], linewidths=2., colors='k')
plt.title('Validation Set')
plt.savefig('lrq.png')