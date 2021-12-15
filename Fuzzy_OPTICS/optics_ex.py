import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import OPTICS
from sklearn.datasets import make_blobs

# Configuration options
num_samples_total = 1000

epsilon = 2.0
min_samples = 22
cluster_method = 'xi'
metric = 'minkowski'

# Generate data
cluster_centers = [(3, 3), (7, 7)]
num_classes = len(cluster_centers)
X, y = make_blobs(n_samples=num_samples_total, centers=cluster_centers, n_features=num_classes, center_box=(0, 1),
                  cluster_std=0.5)

# C1 = [-5, -2] + 0.8 * np.random.randn(num_samples_total, 2)
# C2 = [4, -1] + 0.1 * np.random.randn(num_samples_total, 2)
# C3 = [1, -2] + 0.2 * np.random.randn(num_samples_total, 2)
# C4 = [-2, 3] + 0.3 * np.random.randn(num_samples_total, 2)
# C5 = [3, -2] + 1.6 * np.random.randn(num_samples_total, 2)
# C6 = [5, 6] + 2 * np.random.randn(num_samples_total, 2)
# X = np.vstack((C1, C2, C3, C4, C5, C6))


# Compute OPTICS
db = OPTICS(max_eps=epsilon, min_samples=min_samples, cluster_method=cluster_method, metric=metric).fit(X)
labels = db.labels_

no_clusters = len(np.unique(labels))
no_noise = np.sum(np.array(labels) == -1, axis=0)

print('Estimated no. of clusters: %d' % no_clusters)
print('Estimated no. of noise points: %d' % no_noise)

# Generate scatter plot for training data
colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426', labels))
plt.scatter(X[:, 0], X[:, 1], c=colors, marker="o", picker=True)
plt.title(f'OPTICS clustering')
plt.xlabel('Axis X[0]')
plt.ylabel('Axis X[1]')
plt.show()

# Generate reachability plot
reachability = db.reachability_[db.ordering_]
plt.plot(reachability)
plt.title('Reachability plot')
plt.show()
