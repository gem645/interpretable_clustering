from scipy.spatial import Voronoi
import numpy as np
from feature_importance.volume_calculation import calculate_volume


def calculate_feature_importance(sites, outer_shape, min_data, max_data):
    """Calculate feature importance over a set of clusters.
    More information on how feature importance is calculated is in the read me file.

    :param ndarray sites: lists of sites where each sites is a representative point for a cluster.
    :param ndarray outer_shape: outer convex hull containing the dataset on the interior.
    :param ndarray min_data: minimum data coordinate wise
    :param ndarray max_data: maximum data coordinate wise
    :returns: feature importance of each dimension,
              dictionary of each voronoi faces of form {(points defining voronoi face): (feature importance, volume) }
    :rtype: ndarray, dict
    """
    site_pairs = []
    feature_importance = np.zeros(len(sites[0]))
    # remove the 1d case
    if len(sites[0]) == 1:
        return np.array([1]), []
    # remove the 1 cluster case
    elif len(sites) == 1:
        return np.zeros(len(sites)), []
    # remove the 2 cluster case
    elif len(sites) == 2:
        return np.abs(sites[0] - sites[1]) / np.sum(np.abs(sites[0] - sites[1])), []
    else:
        # calculate Voronoi Diagram
        merge = np.concatenate((sites, outer_shape), axis=0)
        vor = Voronoi(merge)
        pairs = vor.ridge_points
        # determine all Voronoi Faces between sites and not outer shape points
        for pair in pairs:
            if pair[0] < len(sites) and pair[1] < len(sites):
                site_pairs.append(pair)
    voronoi_faces = {}
    # calculate weighted average of feature importance of each Voronoi Face
    for i, j in site_pairs:
        vertices_indexes = vor.ridge_dict[(i, j)]
        vertices = np.array([vor.vertices[index] for index in vertices_indexes])
        points = np.array([vor.points[i], vor.points[j]])
        min_max = np.array([min_data, max_data])
        volume = calculate_volume(vertices, points, min_max)
        if volume > 0:
            vector = np.abs(sites[i] - sites[j])
            voronoi_faces[tuple((i, j))] = (vector / np.sum(vector), volume)
            feature_importance += vector * volume / np.sum(vector)
        else:
            voronoi_faces[tuple((i, j))] = (0, 0)
    # average over all boundaries
    if np.sum(feature_importance) != 0:
        feature_importance = feature_importance / np.sum(feature_importance)
    return feature_importance, voronoi_faces
