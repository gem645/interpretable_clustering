import numpy as np
import itertools

def make_outer_hull(data):
    """Calculate the vertices of an outer convex hull for the given dataset.

    :param ndarray data: list of data points
    :returns: points: out convex hull, min_data: lower bound of data points, max_data: upper bound of data points
    :rtype: ndarray, ndarray, ndarray
    """
    min_data = np.min(data, axis=0)
    max_data = np.max(data, axis=0)
    dis = max_data - min_data
    lower_lim = min_data - dis
    upper_lim = max_data + dis
    outer_hull = list(itertools.product(*zip(lower_lim, upper_lim)))
    outer_hull = [np.array(i) for i in outer_hull]
    return outer_hull, min_data, max_data
