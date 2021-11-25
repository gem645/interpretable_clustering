import numpy as np
from math import cos, sin


def normalise_data_and_centroids(dataset, centroids):
    """Normalise dataset and centroids to 0-1 range.

    :param ndarray dataset: dataset of clusters
    :param ndarray centroids: centers of each cluster
    :return: ndarray normalised dataset, ndarray normalised clusters
    """

    min_data, max_data = np.min(dataset, axis=0), np.max(dataset, axis=0)
    normalised_dataset = (dataset - min_data) / (max_data - min_data)
    normalised_centroids = (centroids - min_data) / (max_data - min_data)
    return normalised_dataset, normalised_centroids


def make_grid(no_clusters_per_feature, cluster_size, return_means=False):
    """Make a grid like structure of spherical clusters

    :param ndarray no_clusters_per_feature: number of rows of cluster per feature
    :param int cluster_size: number of data points in each cluster
    :param bool return_means: if centroids of each cluster should be returned
    :return: ndarray dataset of grid and ndarray of centroids (if return_means = True)
    """
    no_features = len(no_clusters_per_feature)
    # generate centroids of each grid
    feature_centres = []
    for cluster_in_feature in no_clusters_per_feature:
        feature_centres.append([np.mean([i/cluster_in_feature, (i+1)/cluster_in_feature]) for i in range(cluster_in_feature)])
    means = []
    for x_vals in feature_centres:
        if len(means) == 0:
            for x in x_vals:
                means.append([x])
        else:
            cur_len = len(means)
            means = means * len(x_vals)
            for index, x in enumerate(x_vals):
                for i in range(cur_len * index, cur_len * index + cur_len):
                    means[i] = means[i] + [x]
    std = [1/(10 * no_cluster) for no_cluster in no_clusters_per_feature]

    grid_dataset = None
    for index, mean in enumerate(means):
        # generate samples
        cluster = np.random.normal(mean, std, size=[cluster_size, no_features])
        if grid_dataset is None:
            grid_dataset = cluster
        else:
            grid_dataset = np.vstack((grid_dataset, cluster))

    # Normalise to the 0-1 range
    grid_dataset, means = normalise_data_and_centroids(dataset=grid_dataset, centroids=np.array(means))
    if return_means:
        return grid_dataset, means
    return grid_dataset


def make_highly_correlated(cluster_size, return_means=False, theta=None):
    """Make highly correlated synthetic dataset to test.
    Second and third feature generated from the same centroids.

    :param int cluster_size: number of data points in each cluster
    :param bool return_means: if centroids of each cluster should be returned
    :param float rotation: amount of rotation in degrees of second and third feature
                           (default is 45 degrees if not specified)
    :return: dataset
    :rtype: ndarray
    """
    means = np.array([[.25, .25, .25],
                      [.75, .25, .25],
                      [.25, .75, .75],
                      [.75, .75, .75]])

    std = [1 / (10 * 2) for _ in range(3)]

    dataset = None
    for index, mean in enumerate(means):
        # generate dataset
        cluster = np.random.normal(mean, std, size=[cluster_size, 3])
        if dataset is None:
            dataset = cluster
        else:
            dataset = np.vstack((dataset, cluster))
    if theta is not None:
        dataset, means = rotate_highly_correlated(dataset=dataset, means=means, theta=theta-45)
    # Normalise to the 0-1 range
    dataset, means = normalise_data_and_centroids(dataset=dataset, centroids=means)
    if return_means:
        return dataset, means
    return dataset


def rotate_2d(vector, theta):
    """Rotate a given vector by degree theta (in degrees).

    :param ndarray vector: 2-dimensional data point
    :param float theta: angle in degrees
    :return: vector rotated by theta.
    :rtype: ndarray
    """
    theta = np.deg2rad(theta)
    rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    return np.dot(rot, vector)


def rotate_highly_correlated(dataset, means, theta):
    """Rotate the highly correlated dataset.

    :param ndarray dataset: a highly correlated dataset
    :param ndarray means: centroids of each cluster in the highly correlated dataset
    :param float theta: angle of rotation (in degrees)
    :return: ndarray of rotated highly correlated dataset and ndarray of rotated means
    """
    # recenter each data point and rotate
    mid_pt = np.array([.5, .5])
    for index, value in enumerate(dataset):
        dataset[index][1:] = np.array(rotate_2d(vector=dataset[index][1:] - mid_pt, theta=theta)) + mid_pt
    # recenter each mean and rotate
    for index, value in enumerate(means):
        means[index][1:] = np.array(rotate_2d(vector=means[index][1:] - mid_pt, theta=theta)) + mid_pt
    return dataset, means
