import numpy as np
from feature_importance.general_feature_importance_implementation import calculate_feature_importance


def calculate_agglomerative_feature_importance(data, labels, outer_shape, min_data, max_data, specific_layer_of_hierarchy=None):
    """Calculate feature importance for agglomerative clustering.
    More information on how feature importance is calculated is in the read me file.

    :param ndarray data: list of data points
    :param ndarray labels: list of cluster labels at each point in the hierarchy of agglomerative clustering.
    :param ndarray outer_shape: outer convex hull containing the dataset on the interior.
    :param ndarray min_data: minimum data coordinate wise
    :param ndarray max_data: maximum data coordinate wise
    :param ndarray specific_layer_of_hierachy: list if feature importance of only specific layers of the hierarchy is required
    :returns: feature importance of layer in hierarchy
    :rtype: ndarray
    """
    # calculate general feature impotance of all data points
    initial_feature_importance, voronoi_faces = calculate_feature_importance(sites=data,
                                                                             outer_shape=outer_shape,
                                                                             min_data=min_data,
                                                                             max_data=max_data)
    feature_importances = np.zeros((len(data), len(data[0])))
    # then remove each Voronoi Face from weighted averge in hierarchy as clusters merge
    for p1, p2 in voronoi_faces:
        for index, label in enumerate(labels):
            if label[p1] == label[p2]:
                importance, volume = voronoi_faces[tuple((p1, p2))]
                feature_importances[:index] += importance * volume
                break
    feature_importances = feature_importances[:-1]/np.sum(feature_importances[:-1], axis=1)[:, np.newaxis]
    feature_importances = np.r_[feature_importances, [np.zeros(len(data[0]))]]
    # verify this worked as this first layer of feature importance should be equal to the feature importance of
    # general implementation
    assert feature_importances[0] == initial_feature_importance
    if specific_layer_of_hierarchy is None:
        return feature_importances
    else:
        return feature_importances[specific_layer_of_hierarchy]