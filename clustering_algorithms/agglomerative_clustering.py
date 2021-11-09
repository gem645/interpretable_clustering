import heapq
import numpy as np
from sklearn.metrics import pairwise_distances
from feature_importance_development.feature_importance_hierarchical_higher_dimensions import feature_importance_weight_by_volume_agglomerative
import itertools


"""
Making change to see if it works
"""


def domain(data):
    """
    Make the outer convex hull.
    Points = out convex hull
    min_data = lower bound of point
    max_data= upper bound of data points
    """
    min_data = np.min(data, axis=0)
    max_data = np.max(data, axis=0)
    dis = max_data - min_data
    lower_lim = min_data - dis
    upper_lim = max_data + dis
    outer_hull = list(itertools.product(*zip(lower_lim, upper_lim)))
    outer_hull = [np.array(i) for i in outer_hull]
    return outer_hull, min_data, max_data

# Single link
def agglomerative_clustering_single_link(data):
    # single linkage = minimum
    deleted_labels = set()
    initial_distance_matrix = pairwise_distances(data)
    heap = []
    distances = []
    cluster_labels = []
    cluster_id = {}
    current_cluster_labels = []

    # declare the initial all clusters are their own cluster
    for i in range(len(data)):
        cluster_id[i] = i
        current_cluster_labels.append((i))
    cluster_labels.append(current_cluster_labels)

    # establish heap of pairwise distances!
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            heap.append((initial_distance_matrix[i][j], f"{i} {j}"))
    # heap ify
    heapq.heapify(heap)

    no_clusters = len(data)
    joined_clusters = []
    while no_clusters > 1:
        # extract min
        smallest_dist = heapq.heappop(heap)
        merge_pt_1, merge_pt_2 = int(smallest_dist[1].split(" ")[0]), int(smallest_dist[1].split(" ")[1])
        # check it's not already in the same cluster
        if cluster_id[merge_pt_1] == cluster_id[merge_pt_2]:
            continue

        else:
            distances.append(smallest_dist[0])
        joined_clusters.append([cluster_id[merge_pt_1], cluster_id[merge_pt_2]])
        # decide to reassign all of one cluster to the smaller cluster id
        if cluster_id[merge_pt_1] > cluster_id[merge_pt_2]:
            max_cluster_id = cluster_id[merge_pt_1]
            min_cluster_id = cluster_id[merge_pt_2]
        else:
            max_cluster_id = cluster_id[merge_pt_2]
            min_cluster_id = cluster_id[merge_pt_1]

        # redefine cluster labels
        for index in range(len(data)):
            if cluster_id[index] == max_cluster_id:
                cluster_id[index] = min_cluster_id

            elif cluster_id[index] > max_cluster_id:

                cluster_id[index] = cluster_id[index] - 1

        # update current clusters
        update_current_cluster_labels = []

        for i in range(len(data)):
            update_current_cluster_labels.append(cluster_id[i])
        cluster_labels.append(update_current_cluster_labels)
        current_cluster_labels = update_current_cluster_labels


        no_clusters = no_clusters - 1
    points, min_data, max_data = domain(data)
    feature_importance = feature_importance_weight_by_volume_agglomerative(data, np.array(cluster_labels), points, min_data, max_data)
    print("calculated agglomerative")
    return np.array(cluster_labels), np.array(distances), joined_clusters, feature_importance


###########################
# Complete link

def update_proximity_matrix(proximity_matrix, min_id, max_id):
    val1 = list(proximity_matrix[:, min_id][:min_id]) + list(proximity_matrix[:, min_id][min_id+1:max_id]) + list(proximity_matrix[:, min_id][max_id+1:])
    val2 = list(proximity_matrix[:, max_id][:min_id]) + list(proximity_matrix[:, max_id][min_id+1:max_id]) + list(proximity_matrix[:, max_id][max_id+1:])
    # find updated distances to new cluster
    update_distances = np.max(np.array([val1, val2]), axis=0)
    # remove old distances
    # delete the bigger before the smaller so its index doesn't chage
    proximity_matrix = np.delete(proximity_matrix, max_id, 0)
    proximity_matrix = np.delete(proximity_matrix, max_id, 1)
    proximity_matrix[min_id] = np.array(list(update_distances)[:min_id] + [0]+list(update_distances[min_id:]))
    proximity_matrix[:, min_id] = np.array(list(update_distances)[:min_id] + [0]+list(update_distances[min_id:]))
    return proximity_matrix

def update_heap(proximity_matrix):
    heap = []
    for row in range(len(proximity_matrix)):
        for col in range(row + 1, len(proximity_matrix)):
            heap.append((proximity_matrix[row][col], f"{row} {col}"))
    heapq.heapify(heap)
    return heap

def agglomerative_clustering_complete_link(data, feature_importance=True):
    # single linkage = minimum
    proximity_matrix = pairwise_distances(data)
    heap = []
    distances = []
    cluster_labels = []
    cluster_id = {}
    current_cluster_labels = []

    # declare the initial all clusters are their own cluster
    for i in range(len(data)):
        cluster_id[i] = i
        current_cluster_labels.append(i)
    cluster_labels.append(current_cluster_labels)

    # establish heap of pairwise distances!
    no_clusters = len(data)
    heap = update_heap(proximity_matrix)
    joined_clusters = []
    while no_clusters > 1:
        # heapq.heapify(heap)
        # extract min
        smallest_dist = heapq.heappop(heap)
        merge_c_1, merge_c_2 = int(smallest_dist[1].split(" ")[0]), int(smallest_dist[1].split(" ")[1])
        joined_clusters.append([merge_c_1, merge_c_2])
        # append to distances
        distances.append(smallest_dist[0])
        # decide to reassign all of one cluster to the smaller cluster id
        if merge_c_1 > merge_c_2:
            max_cluster_id, min_cluster_id = merge_c_1, merge_c_2
        else:
            max_cluster_id, min_cluster_id = merge_c_2, merge_c_1
        # update cluster labels
        current_cluster_labels = []
        for data_pt in range(len(data)):
            if cluster_id[data_pt] == max_cluster_id:
                cluster_id[data_pt] = min_cluster_id
            elif cluster_id[data_pt] > max_cluster_id:
                cluster_id[data_pt] = cluster_id[data_pt] - 1
            current_cluster_labels.append(cluster_id[data_pt])
        cluster_labels.append(current_cluster_labels)

        # update proximity matrix
        proximity_matrix = update_proximity_matrix(proximity_matrix, min_cluster_id, max_cluster_id)
        # update heap
        heap = update_heap(proximity_matrix)

        no_clusters = no_clusters - 1

    #print("calculated agglomerative")
    points, min_data, max_data = domain(data)

    if feature_importance:
        feature_importance = feature_importance_weight_by_volume_agglomerative(data, np.array(cluster_labels), points, min_data, max_data)
        return np.array(cluster_labels), np.array(distances), joined_clusters, feature_importance
    else:
        return np.array(cluster_labels), np.array(distances), joined_clusters, []


#####################
# Average



def agglomerative_clustering(data, threshold=None):
    """
    :param data: input data
    :param threshold: smallest number of clusters
    :return: n x n matrix where row i is the ith layer of the dendogram
    # an the jth column if the label of the jth data point

    There are two ways to implement to find the minimum value of the
    """
    # create distance matrix






