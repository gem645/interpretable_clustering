import numpy as np
import random
from clustering_algorithms.outer_hull import make_outer_hull
from feature_importance.general_feature_importance_implementation import calculate_feature_importance


def k_means_clustering(data, no_dim=1, feature_importance=True, no_centroids=2, centroids=None, max_iter=100,
                       tol=0.0000001):
    """Perform basic K-Means Clustering.
    This is a basic K-Means clustering and hasn't been optimised. It only works on the Euclidean distance metric.

    :param ndarray data: list of data points
    :param int no_dim: number of dimensions of data points
    :param bool feature_importance: if feature importance does or doesn't want to be calculated
    :param int no_centroids: number of centroids of cluster K-Means must find
    :param ndarray centroids: list of initial centroids (optional)
    :param int max_iter: maximum number of iterations (optional)
    :param float tol: tolerance for difference in movement of clusters between last and second last iteration (optional)
    :return: float MSE: minimum sum of square error or inertia of final iteration,
             ndarray clusters: list of clusters labels,
             ndarray centroidslst: list of centroids at each iteration
             ndarray centroids: list of centroids at final iteration
             ndarray initial_centroids: list of centroids at first iteration
             ndarray importance_profiles: feature importance profiles at each iteration
    """
    if data is None:
        print("These are the required inputs:"
              " (data, no_dim, feature_importance, no_centroids, centroids, max_iter, tol)")
        return

    iterations = 0
    if no_centroids > len(data):
        raise Exception("More clusters then data instances")

    # determine initial centroids
    if centroids is None:
        centroids = np.array(random.sample(list(data), no_centroids))
    initial_centroids = centroids

    # create importance profiles of initial clusters
    convex_outer_domain, min_data, max_data = None, None, None
    importance_profiles = None
    if feature_importance:
        # determine the range of the data
        convex_outer_domain, min_data, max_data = make_outer_hull(data)
        importance_profiles = [calculate_feature_importance(centroids, convex_outer_domain, min_data, max_data)[0]]

    updated_centroids = np.zeros((no_centroids, no_dim))
    clusters = np.zeros(len(data))
    centroidslst = [centroids]

    while True:
        # print(f"Iteration: {iterations}")
        for index, data_pt in enumerate(data):
            distance_to_centroids = np.linalg.norm(np.abs(centroids - data_pt), axis=1)
            clusters[index] = np.argmin(distance_to_centroids)
        for i in range(no_centroids):
            if len(data[clusters == i]) == 0:
                raise Exception("error occurred in assigning singleton clusters")
            updated_centroids[i] = np.mean(data[clusters == i], axis=0)

        # calculate feature importance of new centroids
        if feature_importance:
            importance_profiles.append(
                calculate_feature_importance(updated_centroids, convex_outer_domain, min_data, max_data)[0])

        iterations += 1

        # check if it's within the required tolerance
        if np.linalg.norm(updated_centroids - centroids) < tol:
            break
        # check it has run above the maximum number of iterations
        elif max_iter < iterations:
            break
        else:
            centroids = updated_centroids
            updated_centroids = np.zeros((no_centroids, no_dim))
            centroidslst.append(centroids)

    centroids = updated_centroids
    MSE = 0
    for index, data_pt in enumerate(data):
        MSE += np.sum(np.linalg.norm(np.abs(centroids[int(clusters[index])] - data_pt)) ** 2)
    # Optional to know the number of iterations
    # print(f"Total number of iterations: {iterations}")
    return MSE, clusters, centroidslst, centroids, initial_centroids, importance_profiles
