import trimesh
import numpy as np

def sample_points_on_mesh(mesh, num_points):
    """
    Sample random points uniformly distributed on the surface of the mesh.

    :param mesh: A trimesh object.
    :param num_points: The number of points to sample.
    :return: An array of sampled points of shape (num_points, 3).
    """
    # Calculate the area of each triangle
    areas = mesh.area_faces
    total_area = areas.sum()

    # Compute the cumulative distribution function (CDF)
    cdf = np.cumsum(areas) / total_area

    # Sample triangles based on the CDF
    random_vals = np.random.rand(num_points)
    triangle_indices = np.searchsorted(cdf, random_vals)

    # Get vertices of the selected triangles
    triangles = mesh.triangles[triangle_indices]
    print(triangles.shape)
    # Sample random points in each triangle using barycentric coordinates
    u = np.random.rand(num_points, 1)
    v = np.random.rand(num_points, 1)
    is_valid = u + v <= 1
    u[~is_valid] = 1 - u[~is_valid]
    v[~is_valid] = 1 - v[~is_valid]
    w = 1 - u - v

    # Calculate the points
    points = (triangles[:, 0] * u + triangles[:, 1] * v + triangles[:, 2] * w)

    return points