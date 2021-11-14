from scipy.spatial import Voronoi
import numpy as np
from feature_importance.volume_calculation import calculate_volume

def calculate_feature_importance(sites, outer_shape, min_data, max_data):
    """
    Output:
    feature_importance = (f_1,...,f_d)
    voronoi_faces = {} of the form:
    [(point 1, point 2)] = (feature importance of Voronoi face between point 1 and 2,
                            volume of Voronoi Face)
    """
    site_pairs = []
    feature_importance = np.zeros(len(sites[0]))
    if len(sites[0]) == 1:
        return np.array([1]), []
    elif len(sites) ==1:
        return np.zeros(len(sites)), []
    elif len(sites) == 2:
        return np.abs(sites[0] - sites[1])/ np.sum(np.abs(sites[0] - sites[1])), []
    else:
        merge = np.concatenate((sites, outer_shape), axis=0)
        vor = Voronoi(merge)
        pairs = vor.ridge_points
        for pair in pairs:
            if pair[0] < len(sites) and pair[1] < len(sites):
                site_pairs.append(pair)
    voronoi_faces = {}
    for i, j in site_pairs:
        vertices_indexes = vor.ridge_dict[(i, j)]
        vertices = np.array([vor.vertices[index] for index in vertices_indexes])
        points = np.array([vor.points[i], vor.points[j]])
        min_max = np.array([min_data, max_data])
        volume = calculate_volume(vertices, points, min_max)
        if volume > 0:
            vector = np.abs(sites[i] - sites[j])
            voronoi_faces[tuple((i, j))] = (vector/np.sum(vector), volume)
            feature_importance += vector * volume / np.sum(vector)
        else:
            voronoi_faces[tuple((i, j))] = (0, 0)
    # average over all boundaries
    if np.sum(feature_importance) != 0:
        feature_importance = feature_importance/np.sum(feature_importance)
    return feature_importance, voronoi_faces