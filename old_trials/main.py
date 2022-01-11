# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# %matplotlib inline
import numpy as np
from fcmeans import FCM  # pip install fuzzy-c-means

from matplotlib import pyplot as plt


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    n_samples = 250

    C1 = [-5, -2] + 0.8 * np.random.randn(n_samples, 2)
    C2 = [4, -1] + 0.1 * np.random.randn(n_samples, 2)
    C3 = [1, -2] + 0.2 * np.random.randn(n_samples, 2)
    C4 = [-2, 3] + 0.3 * np.random.randn(n_samples, 2)
    C5 = [3, -2] + 1.6 * np.random.randn(n_samples, 2)
    C6 = [5, 6] + 2 * np.random.randn(n_samples, 2)
    X = np.vstack((C1, C2, C3, C4, C5, C6))

    fcm = FCM(n_clusters=2)
    fcm.fit(X)
    # outputs
    fcm_centers = fcm.centers
    fcm_labels = fcm.predict(X)

    # plot result
    f, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].scatter(X[:, 0], X[:, 1], alpha=.1)
    axes[1].scatter(X[:, 0], X[:, 1], c=fcm_labels, alpha=.1)
    axes[1].scatter(fcm_centers[:, 0], fcm_centers[:, 1], marker="+", s=200, c='black')
    plt.savefig('basic-clustering-output.jpg')
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
