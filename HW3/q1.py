import numpy as np
import matplotlib.pyplot as plt

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

loss = [[0, 1], [1, 0]]
def cal_prob(x, m, c):
    return np.exp(-0.5 * (x - m).T @ np.linalg.inv(c) @ (x - m)) / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(c))


def classifier(x, threshold, m, c):
    likelihood_ratio = cal_prob(x, m[2], c[2]) / (w[0] * cal_prob(x, m[0], c[0]) + w[1] * cal_prob(x, m[1], c[1]))
    if(likelihood_ratio > threshold):
        return 1
    else:
        return 0

validation_samples = samples[-sample_size[3]:].T
validation_labels = labels[-sample_size[3]:]

thresholds = np.arange(0, 1, 0.1)
results = np.zeros(sample_size[3])
tp = np.zeros(len(thresholds))
fp = np.zeros(len(thresholds))
fn = np.zeros(len(thresholds))
for t in range(len(thresholds)):
    for i in range(sample_size[3]):
        results[i] = classifier(validation_samples[:, i], thresholds[t], m, c)
        tp[t] += (results[i] == validation_labels[i]) and (validation_labels[i] == 1)
        fp[t] += (results[i] != validation_labels[i]) and (validation_labels[i] == 0)
        fn[t] += (results[i] != validation_labels[i]) and (validation_labels[i] == 1)
    print('Threshold: {0:.3f}'.format(thresholds[t]), "Progress: {0:.2f}%".format(t / len(thresholds) * 100), end='\r')

# calculate the probability of error
p_error = (fn + fp) / sample_size[3]
plt.figure()
plt.plot(thresholds, p_error)
plt.xlabel('Threshold')
plt.ylabel('P-Error')
plt.title('P-Error vs Threshold')
plt.grid()
plt.savefig('p_error.png')
# find the best threshold
best_threshold = thresholds[np.argmin(p_error)]
print('Best Threshold: ', best_threshold)
print('P-Error: ', p_error[np.argmin(p_error)])
tp_rate = tp / np.sum(labels == 1)
fp_rate = fp / np.sum(labels == 0)


tp_theoritical = 0
fp_theoritical = 0
fn_theoritical = 0

for i in range(sample_size[3]):
    results[i] = classifier(validation_samples[:,i], p[0]/p[1], m, c)
    tp_theoritical += (results[i] == validation_labels[i]) and (validation_labels[i] == 1)
    fp_theoritical += (results[i] != validation_labels[i]) and (validation_labels[i] == 0)
    fn_theoritical += (results[i] != validation_labels[i]) and (validation_labels[i] == 1)
    print('Classifying:', i, '/', sample_size[3], end='\r')
p_error_theoritical = (fn_theoritical + fp_theoritical) / sample_size[3]
print('Theoritical P-Error: ', p_error_theoritical)
print('Theoritical Threshold: ', p[0]/p[1])
plt.figure()
plt.plot(fp_rate, tp_rate, fp_rate[int(np.argmin(p_error))], tp_rate[int(np.argmin(p_error))], 'gx', fp_theoritical / np.sum(validation_labels == 0), tp_theoritical / np.sum(validation_labels == 1), 'rx')
plt.xlabel('False Alarm Rate')
plt.ylabel('Hit Rate')
plt.title('ROC')
plt.grid()
plt.savefig('roc.png')

# Draw the decision boundary on the likelihood ratio of the validation set with the best threshold found
plt.figure()
plt.plot(validation_samples[0, validation_labels == 0], validation_samples[1, validation_labels == 0], 'bx', validation_samples[0, validation_labels == 1], validation_samples[1, validation_labels == 1], 'rx')
x = np.arange(-5, 10, 0.1)
y = np.arange(-5, 10, 0.1)
X, Y = np.meshgrid(x, y)
Z = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = classifier(np.array([X[i, j], Y[i, j]]), best_threshold, m, c)
plt.contour(X, Y, Z, levels=[0.5])
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Decision Boundary')
plt.savefig('decision_boundary.png')


# Part 2 


training_samples = samples[sample_size[0] + sample_size[1]:-sample_size[3]].T
training_labels = labels[sample_size[0] + sample_size[1]:-sample_size[3]]

est_m = np.zeros((2, 2))
est_c = np.zeros((2, 2, 2))
est_p = np.zeros(2)
for i in range(2):   
    est_m[i] = np.mean(training_samples[:, training_labels == i], axis=1)
    est_c[i] = np.cov(training_samples[:, training_labels == i])

est_p[0] = np.sum(training_labels == 0) / training_labels.shape[0]
est_p[1] = np.sum(training_labels == 1) / training_labels.shape[0]
print(f"Estimated m: {est_m} \nEstimated c: {est_c} \nEstimated p: {est_p}")
r = int(10)
m_candidates = []
for i in np.arange(-r, r, r/10):
    for j in np.arange(-r, r, r/10):
        m_candidates.append([i - r/2, j - r/2])

c_candidates = []

for i in np.arange(-r, r, r/10):
    for j in np.arange(-r, r, r/10):
        for k in np.arange(-r, r, r/10):
            for l in np.arange(-r, r, r/10):
                c_candidates.append([[i - r/2, j - r/2 + 0.1], [k - r/2 + 0.1, l - r/2]])

likelihood = []
max_likelihood = 0
mle_m_best = np.zeros((2, 2))
mle_c_best = np.zeros((2, 2, 2))

for i in range(len(m_candidates)):
    for j in range(len(c_candidates)):
        print(f"Progress: {i * len(c_candidates) + j} / {len(m_candidates) * len(c_candidates)}", end='\r')
        area = 0
        for l in range(2):
            for k in range(len(training_samples)):
                if(training_labels[k] == l):
                    area += cal_prob(training_samples[:, k], m_candidates[i], c_candidates[j])
            if area > max_likelihood:
                max_likelihood = area
                mle_m_best[l] = m_candidates[i]
                mle_c_best[l] = c_candidates[j]


print(f"Best m: {mle_m_best} \nBest c: {mle_c_best}")
