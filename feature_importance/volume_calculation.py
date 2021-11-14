import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog
from sklearn.metrics import pairwise_distances

#### DO NOT TOUCH THIS ####
def fully_contained(vertices, min_max):
    """
    Determine if all vertices lie in the bounding box.
    """
    min_array = min_max[0].reshape((len(vertices[0]), 1))
    max_array = min_max[1].reshape((len(vertices[0]), 1))
    if np.sum(vertices.T >= min_array) != len(vertices) * len(vertices[0]):
        return False
    if np.sum(vertices.T <= max_array) != len(vertices) * len(vertices[0]):
        return False
    else:
        return True


def gram_schmidt(input_vector, no_dim):
    u_1 = input_vector / np.linalg.norm(input_vector)
    basis_vectors = []
    for i in range(no_dim):
        v = np.zeros(no_dim)
        v[i] = 1
        proj = np.dot(v, u_1) * u_1 + np.sum([np.dot(v, vec) * vec for vec in basis_vectors], axis=0)
        y = v - proj
        if not np.allclose(y, np.zeros(no_dim)):
            u = y / np.linalg.norm(y)
            basis_vectors.append(u)
        if len(basis_vectors) == no_dim - 1:
            break
    basis_vectors.append(u_1)
    return np.array(basis_vectors)


def generate_change_of_basis(opposing_points):
    # then calculate the basis of the n-1 dimensional plane
    basis = gram_schmidt(opposing_points[1], len(opposing_points[0]))
    assert len(basis) == len(basis[0])
    change_of_basis_matrix = np.linalg.inv(basis.T)
    return change_of_basis_matrix, basis.T

def apply_change_of_basis(change_of_basis_matrix, vertices):
    updated_vertices = change_of_basis_matrix.dot(vertices.T).T
    assert np.allclose(pairwise_distances(vertices), pairwise_distances(updated_vertices))
    updated_vertices = np.array(updated_vertices)
    updated_vertices = np.delete(updated_vertices, -1, 1)
    return updated_vertices

def basis_conversion(data_points, basis):
    new_data_points = []
    for data_point in data_points:
        new_data_points.append(basis.dot(data_point))
    return np.array(new_data_points)


def reduce_halfspaces(halfspaces):
    reduced_halfspaces = set()
    for halfspace in halfspaces:
        reduced_halfspaces.add(tuple(halfspace))
    return np.array([list(i) for i in reduced_halfspaces])


def step_one(min_max, vertices, points):
    """
    Identifying the points of the bounding box on the plane of the voronoi face.
    """
    # halfspaces of the bounding box
    halfspaces_bounding_box = []
    for index, value in enumerate(min_max[0]):
        hyperplane = np.zeros(len(min_max[0]) + 1)
        hyperplane[index] = -1
        hyperplane[-1] = value
        halfspaces_bounding_box.append(hyperplane)
    for index, value in enumerate(min_max[1]):
        hyperplane = np.zeros(len(min_max[1]) + 1)
        hyperplane[index] = 1
        hyperplane[-1] = -1 * value
        halfspaces_bounding_box.append(hyperplane)

    halfplane_voronoi = np.array(list(points[0] / np.sum(np.abs(points[0]))) + [0])

    outputs = np.zeros((len(vertices), len(vertices[0]) + 1))
    outputs.T[: -1] = vertices.T

    # combine bounding a both halfspaces into a series into  of
    halfspaces = np.vstack((halfspaces_bounding_box, halfplane_voronoi))
    norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1),
                             (halfspaces.shape[0], 1))

    c = np.zeros((halfspaces.shape[1],))
    c[-1] = -1
    A = np.hstack((halfspaces[:, :-1], norm_vector))
    b = - halfspaces[:, -1:]
    optim = linprog(c, A_ub=A, b_ub=b,  bounds=(None, None), method='interior-point')
    feasible_point = optim.x[:-1]

    try:
        hs = HalfspaceIntersection(halfspaces, feasible_point)
    except:
        print("didn't work")



    # then look for all intersection points return those which lie on the plane
    # first add a final column of one
    intersecting = np.ones((len(hs.intersections), len(halfplane_voronoi)))
    intersecting.T[:-1] = hs.intersections.T
    summation = np.sum(intersecting * halfplane_voronoi, axis=1)
    summation = [round(intersect, 10) == 0 for intersect in summation]
    return hs.intersections[np.where(summation)[0]]


def calculate_1D_volume(vor_vertices, bound_vertices):
    # sort the pts
    if vor_vertices[0] < vor_vertices[1]:
        vor_a = vor_vertices[0]
        vor_b = vor_vertices[1]
    else:
        vor_a = vor_vertices[1]
        vor_b = vor_vertices[0]
    if bound_vertices[0] < bound_vertices[1]:
        bound_a = bound_vertices[0]
        bound_b = bound_vertices[1]
    else:
        bound_a = bound_vertices[1]
        bound_b = bound_vertices[0]
    if bound_a > vor_b or vor_a > bound_b:
        # no overlap
        return -1
    else:
        return min(bound_b, vor_b) - max(bound_a, vor_a)


def step_three(vor_vertices, bound_vertices):
    """
    Find vertices of intersection between bounding box and voronoi face.
    """
    # first identifying if a single feasible point exists using scipy.optimize.linprog.
    # Again if no solution exists it implies theres no feasible point and no overlap
    # threfore no volume bounding box

    # create the convex hull of both these two there is repetition in this code
    if len(vor_vertices[0]) == 1:
        return calculate_1D_volume(vor_vertices, bound_vertices)
    try:
        conv_vor = ConvexHull(vor_vertices, qhull_options='Q12' )  # , qhull_options='Q14 QbB Q12 Pp')
    except:
        return -1
    conv_bound = ConvexHull(bound_vertices)  # , qhull_options='Q14 QbB Q12 Pp')
    # combine the equations into a matrix and solve
    halfspaces = np.vstack((conv_vor.equations, conv_bound.equations))
    halfspaces = reduce_halfspaces(halfspaces)
    norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1),
                             (halfspaces.shape[0], 1))

    c = np.zeros((halfspaces.shape[1],))
    c[-1] = -1
    A = np.hstack((halfspaces[:, :-1], norm_vector))
    b = - halfspaces[:, -1:]
    try:
        optim = linprog(c, A_ub=A, b_ub=b, bounds=(None, None), method='interior-point') # , options={'maxiter': len(vor_vertices[0])})
    except:
        return -1

    if optim.status == 2:
        # no volume found
        return -1
    elif optim.status != 1 and optim.status != 0:
        print("Error occured problem finding chebyshev")
        return -1
    # now we have a feasible point so will use halfspaces
    x = optim.x[:-1]

    try:
        hs = HalfspaceIntersection(halfspaces, x)
    except:
        return -1
    vertices = hs.intersections
    try:
        conv_hull = ConvexHull(vertices)  # , qhull_options='QbB Q12')
        return conv_hull.volume
    except:
        print("in here itme")
        return -1



def calculate_volume(vertices, points, min_max):
    contained = fully_contained(vertices, min_max)
    if contained and len(vertices[0]) == 1:
        return np.abs(vertices[0] - vertices[1])
    mid_point = np.mean(points, axis=0)
    vertices = vertices - mid_point
    points = points - mid_point
    min_max = min_max - mid_point

    change_of_basis_matrix, basis = generate_change_of_basis(points)

    updated_vertices = apply_change_of_basis(change_of_basis_matrix, vertices)
    basis = np.delete(basis, -1, 1)
    if contained:
        # check for 1 array
        if len(updated_vertices[0]) == 1:
            return np.abs(updated_vertices[0] - updated_vertices[1])
        else:
            conv = ConvexHull(updated_vertices,
                              qhull_options='QJ')  # , qhull_options='QbB Q12')#ConvexHull(updated_vertices)
            return conv.volume

    bound_vertices = step_one(min_max=min_max, vertices=vertices, points=points)
    if len(bound_vertices) <= 1:
        return -1
    updated_bound_vertices = apply_change_of_basis(change_of_basis_matrix, bound_vertices)
    #ow assert np.allclose(basis.dot(updated_bound_vertices.T).T, bound_vertices)
    vol = step_three(updated_vertices, updated_bound_vertices)
    return vol
