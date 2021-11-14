import numpy as np
from feature_importance.general_feature_importance_implementation import calculate_feature_importance


def calculate_agglomerative_feature_importance(data, labels, outer_shape, min_data, max_data, specific_layer_of_hierarchy=None):
    """
    calculates agglomerative clustering feature importance.

    """
    initial_feature_importance, voronoi_faces = calculate_feature_importance(sites=data,
                                                                             outer_shape=outer_shape,
                                                                             min_data=min_data,
                                                                             max_data=max_data)
    feature_importances = np.zeros((len(data), len(data[0])))
    for p1, p2 in voronoi_faces:
        for index, label in enumerate(labels):
            if label[p1] == label[p2]:
                importance, volume = voronoi_faces[tuple((p1, p2))]
                feature_importances[:index] += importance * volume
                break
    feature_importances = feature_importances[:-1]/np.sum(feature_importances[:-1], axis=1)[:, np.newaxis]
    feature_importances = np.r_[feature_importances, [np.zeros(len(data[0]))]]
    assert feature_importances[0] == initial_feature_importance
    if specific_layer_of_hierarchy is None:
        return feature_importances
    else:
        return feature_importances[specific_layer_of_hierarchy]