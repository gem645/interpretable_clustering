import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog
from sklearn.metrics import pairwise_distances


def fully_contained(vertices, min_max):
    """Determine if all vertices lie in the bounding box.

    :param ndarray vertices: array of vertices defining a convex polytope
    :param ndarray min_max: minimum and maximum points defining a bounding box
    :returns: if all vertices are in bounding box defined by min_max
    :rtype: bool
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
    """Perform Gram-Schmidt process over {input_vector, e_1,..,e_d}.

    :param ndarray input_vector: vector (normal to hyperplane)
    :param int no_dim: number of dimensions
    :returns: array of length no_dim of basis vectors
    :rtype: ndarray
    """
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
    """Calculate change of basis matrix.

    :param ndarray opposing_points: two vectors each normal to hyperplane
    :returns: change of basis matrix,  array of basis vectors
    :rtype: ndarray, ndarray
    """
    basis = gram_schmidt(opposing_points[1], len(opposing_points[0]))
    assert len(basis) == len(basis[0])
    change_of_basis_matrix = np.linalg.inv(basis.T)
    return change_of_basis_matrix, basis.T


def apply_change_of_basis(change_of_basis_matrix, vertices):
    """Apply change of basis matrix to a list of vertices.

    :param ndarray change_of_basis_matrix: change of basis matrix
    :param ndarray vertices: points to apply change of basis to
    :returns: updated vertices on the d-1-dimensional hyperplane
    :rtype: ndarray
    """
    updated_vertices = change_of_basis_matrix.dot(vertices.T).T
    # check relative distances were preserved
    assert np.allclose(pairwise_distances(vertices), pairwise_distances(updated_vertices))
    updated_vertices = np.array(updated_vertices)
    updated_vertices = np.delete(updated_vertices, -1, 1)
    return updated_vertices


def reduce_halfspaces(halfspaces):
    """Remove repetition in halfspace.
     In the scipy.spatial.ConvexHull equations there is often repetition in hyperplane equations.

    :param ndarray halfspaces: list of halfspaces
    :returns: distinct (non-repeating) list of halfspaces
    :rtype: ndarray
    """
    reduced_halfspaces = set()
    for halfspace in halfspaces:
        reduced_halfspaces.add(tuple(halfspace))
    return np.array([list(i) for i in reduced_halfspaces])


def calculate_bounding_box_points_on_voronoi_face_plane(min_max, vertices, points):
    """Identify  points of the bounding box that lie on hyperplane containing the voronoi face.

    :param ndarray min_max: minimum and maximum points defining a bounding box
    :param ndarray vertices: array of Voronoi vertices defining the Voronoi Face
    :param ndarray points: pair of points that share the Voronoi Face
    :returns: list of vertices that lie on the hyperplane containing the Voronoi Face
    :rtype: ndarray
    """
    # Calculate halfspaces of the bounding box
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

    # Calculate the hyperplane containing the Voronoi Face
    halfplane_voronoi = np.array(list(points[0] / np.sum(np.abs(points[0]))) + [0])

    outputs = np.zeros((len(vertices), len(vertices[0]) + 1))
    outputs.T[: -1] = vertices.T

    # combine halfspaces of bounding box and Voronoi Face
    halfspaces = np.vstack((halfspaces_bounding_box, halfplane_voronoi))
    norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1),
                             (halfspaces.shape[0], 1))

    c = np.zeros((halfspaces.shape[1],))
    c[-1] = -1
    A = np.hstack((halfspaces[:, :-1], norm_vector))
    b = - halfspaces[:, -1:]

    # try to find a feasible point in halfspace intersection
    # if one isn't found can determine one doesn't exist
    try:
        optim = linprog(c, A_ub=A, b_ub=b, bounds=(None, None), method='interior-point')
    except:
        return -1

    if optim.status == 2:
        return -1
    feasible_point = optim.x[:-1]

    # Obtain halfspace intersection
    try:
        hs = HalfspaceIntersection(halfspaces, feasible_point)
    except:
        print("didn't work")
        return -1

    # Obtain all points of intersection that lie on the hyperplane containing the Voronoi Face
    intersecting = np.ones((len(hs.intersections), len(halfplane_voronoi)))
    intersecting.T[:-1] = hs.intersections.T
    summation = np.sum(intersecting * halfplane_voronoi, axis=1)
    summation = [round(intersect, 10) == 0 for intersect in summation]
    return hs.intersections[np.where(summation)[0]]


def calculate_1D_volume(vor_vertices, bound_vertices):
    """Identify length of interval in 1D case.
    scipy.spatial functions used require at least 2 dimensions.

    :param ndarray vor_vertices: Vertices defining Voronoi Face
    :param ndarray bound_vertices: Vertices defining the intersection of bounding box
    :returns: length of interval intersecting bounding box and 1D Voronoi Face
    :rtype: float
    """
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


def calculate_intersection_volume(vor_vertices, bound_vertices):
    """Calculate the volume contained in the intersection of two convex polytopes.
     Each convex polytope is defined by vor_vertices and bound_vertices.

    :param ndarray vor_vertices: Vertices defining Voronoi Face
    :param ndarray bound_vertices: Vertices defining the intersection of bounding box
    :returns: volume of intersecting bounding box and Voronoi Face
    :rtype: float
    """
    # Check it's not the 1D case
    if len(vor_vertices[0]) == 1:
        return calculate_1D_volume(vor_vertices, bound_vertices)

    # Obtain convex structure of Voronoi Face (if one doesn't exist it's because the object is too obsecure)
    try:
        conv_vor = ConvexHull(vor_vertices, qhull_options='Q12')
    except:
        return -1

    # Obtain convex structure of Voronoi Face (if one doesn't exist it's because the object is too obsecure)
    try:
        conv_bound = ConvexHull(bound_vertices)
    except:
        return -1
    # Combine the equations of each convex polytopes together and solve for halfspace intersection
    halfspaces = np.vstack((conv_vor.equations, conv_bound.equations))
    halfspaces = reduce_halfspaces(halfspaces)
    norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1),
                             (halfspaces.shape[0], 1))

    c = np.zeros((halfspaces.shape[1],))
    c[-1] = -1
    A = np.hstack((halfspaces[:, :-1], norm_vector))
    b = - halfspaces[:, -1:]
    # Identify a feasible point
    try:
        optim = linprog(c, A_ub=A, b_ub=b, bounds=(None, None),
                        method='interior-point')
    except:
        return -1

    if optim.status == 2:
        # no volume found
        return -1
    x = optim.x[:-1]

    # Find the vertices of halfsapce intersection
    try:
        hs = HalfspaceIntersection(halfspaces, x)
    except:
        return -1
    vertices = hs.intersections

    # Calculate the volume of halfspace intersection
    try:
        conv_hull = ConvexHull(vertices)
        return conv_hull.volume
    except:
        # if one doesn't exist because the intersection was too small
        return -1


def calculate_volume(vertices, points, min_max):
    """Calculate the volume of Voronoi Face contained in bounding box.

    :param ndarray vertices: Vertices defining Voronoi Face
    :param ndarray points: Vertices defining the intersection of bounding box
    :param ndarray min_max:  minimum and maximum points defining a bounding box
    :returns: volume of intersecting bounding box and Voronoi Face or -1 if a volume doesn't exist
    :rtype: float
    """
    contained = fully_contained(vertices=vertices, min_max=min_max)
    if contained and len(vertices[0]) == 1:
        return np.abs(vertices[0] - vertices[1])
    mid_point = np.mean(points, axis=0)
    vertices = vertices - mid_point
    points = points - mid_point
    min_max = min_max - mid_point

    change_of_basis_matrix, basis = generate_change_of_basis(opposing_points=points)

    updated_vertices = apply_change_of_basis(change_of_basis_matrix=change_of_basis_matrix, vertices=vertices)
    basis = np.delete(basis, -1, 1)
    # One check to ensure the change of basis project onto only the hyperplane
    assert np.allclose(basis.dot(updated_vertices.T).T, vertices, rtol=1e-07)
    if contained:
        # check for 1-dimension array
        if len(updated_vertices[0]) == 1:
            return np.abs(updated_vertices[0] - updated_vertices[1])
        else:
            conv = ConvexHull(updated_vertices,
                              qhull_options='QJ')
            return conv.volume

    bound_vertices = calculate_bounding_box_points_on_voronoi_face_plane(min_max=min_max, vertices=vertices,
                                                                         points=points)
    if len(bound_vertices) <= 1:
        return -1
    updated_bound_vertices = apply_change_of_basis(change_of_basis_matrix=change_of_basis_matrix, vertices=bound_vertices)
    assert np.allclose(basis.dot(updated_bound_vertices.T).T, bound_vertices, rtol=1e-07)
    vol = calculate_intersection_volume(vor_vertices=updated_vertices, bound_vertices=updated_bound_vertices)
    return vol
