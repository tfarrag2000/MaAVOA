# from collections import Counter
# from sklearn.cluster import OPTICS
# from sklearn.datasets import make_blobs
# from fcmeans import FCM  # pip install fuzzy-c-means
# from matplotlib import pyplot as plt
# from pymoo.core.initialization import Initialization
# from pymoo.factory import get_problem
# from pymoo.operators.sampling.rnd import FloatRandomSampling
# import numpy as np
#
#
# problem = get_problem("dtlz1", n_var=7, n_obj=3)
#
# initialization = Initialization(FloatRandomSampling())
# init_pop = initialization.do(problem,1000)
# X=init_pop.get("X")
#
# fcm = FCM(n_clusters=3)
# fcm.fit(X)
# # outputs
#
# fcm_centers = fcm.centers
# fcm_labels = fcm.predict(X)
# fcm_stat=Counter(fcm_labels)
# init_pop.set("Cluster",fcm_labels)
# # plot result
# f, axes = plt.subplots(1, 2, figsize=(11, 5))
# axes[0].scatter(X[:, 0], X[:, 1], alpha=.1)
# axes[1].scatter(X[:, 0], X[:, 1], c=fcm_labels, alpha=.1)
# axes[1].scatter(fcm_centers[:, 0], fcm_centers[:, 1], marker="+", s=200, c='black')
# plt.savefig('basic-clustering-output.jpg')
# plt.show()
#
#
#
#
# # Optics
#
# min_samples = 300
# cluster_method = 'xi'
# metric = 'minkowski'
#
# # Compute OPTICS
# db = OPTICS( min_samples=min_samples, cluster_method=cluster_method, metric=metric).fit(X)
# labels = db.labels_
#
# no_clusters = len(np.unique(labels))
# no_noise = np.sum(np.array(labels) == -1, axis=0)
#
# print('Estimated no. of clusters: %d' % no_clusters)
# print('Estimated no. of noise points: %d' % no_noise)
# # Generate scatter plot for training data
# colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426', labels))
# plt.scatter(X[:, 0], X[:, 1], c=colors, marker="o", picker=True)
# plt.title(f'OPTICS clustering')
# plt.xlabel('Axis X[0]')
# plt.ylabel('Axis X[1]')
# plt.show()
#
# # Generate reachability plot
# reachability = db.reachability_[db.ordering_]
# plt.plot(reachability)
# plt.title('Reachability plot')
# plt.show()


from pymoo.factory import get_reference_directions

n_dim = [3, 5, 8, 10]
n_points = [len(get_reference_directions("das-dennis", d, n_partitions=5)) for d in n_dim]
pass
