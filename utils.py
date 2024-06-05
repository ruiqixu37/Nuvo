import trimesh
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd


def sample_uv_points(num_points):
    """
    Sample random UV points uniformly distributed in the unit square.

    :param num_points: The number of points to sample.
    :return: An array of sampled UV points of shape (num_points, 2).
    """
    points = np.random.rand(num_points, 2)
    return points


def sample_points_on_mesh(mesh, num_points):
    """
    Sample random points uniformly distributed on the surface of the mesh.

    :param mesh: A trimesh object.
    :param num_points: The number of points to sample.
    :return: 1. An array of sampled points of shape (num_points, 3).
        2. An array of vertex normals at the sampled points of shape (num_points, 3).
        Note: For simplicity, the vertex normals are equal to face normals,
        because the sampled points are not necessarily on the vertices most of the time.
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
    # Sample random points in each triangle using barycentric coordinates
    u = np.random.rand(num_points, 1)
    v = np.random.rand(num_points, 1)
    is_valid = u + v <= 1
    u[~is_valid] = 1 - u[~is_valid]
    v[~is_valid] = 1 - v[~is_valid]
    w = 1 - u - v

    # Calculate the points
    points = triangles[:, 0] * u + triangles[:, 1] * v + triangles[:, 2] * w

    normals = mesh.face_normals[triangle_indices]

    return points, normals


def random_tangent_vectors(normals):
    """
    Generate random orthogonal tangent vectors given a normal vector.

    :param normal: The normal vector of shape (num_points, 3)
    :return: Two lists orthogonal tangent vectors of shape (num_points) each.
    """
    num_points = normals.shape[0]

    # Normalize the normal vectors
    normals = normals / torch.norm(normals, dim=1, keepdim=True)

    # Generate random vectors and ensure they are orthogonal to the normal vectors
    tangent1 = torch.rand((num_points, 3), device=normals.device)
    tangent1 -= (tangent1 * normals).sum(dim=1, keepdim=True) * normals
    tangent1 /= torch.norm(tangent1, dim=1, keepdim=True)

    # Compute the second tangent vector using the cross product
    tangent2 = torch.linalg.cross(normals, tangent1)
    tangent2 /= torch.norm(tangent2, dim=1, keepdim=True)

    return tangent1, tangent2


def compute_jacobian(mlp, chart_idx, points):
    """
    Compute the Jacobian matrix of the MLP at point x using automatic differentiation.

    :param mlp: The MLP representing the texture coordinate mapping.
    :param points: The input point of shape (num_points, 3).
    :return: The Jacobian matrix of shape (num_points, 3, 2).
    """
    # Compute the gradient with respect to the input
    points.requires_grad_(True)
    y = mlp(points, chart_idx)

    jacobian = []
    for i in range(y.shape[1]):
        grads = autograd.grad(
            outputs=y[:, i],
            inputs=points,
            grad_outputs=torch.ones_like(y[:, i]),
            retain_graph=True,
            create_graph=True,
        )[0]
        jacobian.append(grads)

    jacobian = torch.stack(jacobian, dim=-1)
    return jacobian


def compute_uv_vectors(mlp, points, normals, chart_idx, epsilon=1e-2):
    """
    Compute the UV vectors for given point x using the MLP and perturbations.

    :param mlp: The MLP representing the texture coordinate mapping.
    :param points: The input points of shape (num_points, 3).
    :param normal: The normal vector at point x of shape (num_points, 3).
    :param epsilon: Small perturbation value.
    :return: The UV vectors Dti(εpx) and Dti(εqx).
    """
    tangent1, tangent2 = random_tangent_vectors(normals)

    jacobian = compute_jacobian(mlp, chart_idx, points)

    # Transform the tangent vectors using the Jacobian matrix
    Dti_pxs = torch.bmm(
        jacobian.transpose(1, 2), epsilon * tangent1.unsqueeze(-1)
    ).squeeze(-1)
    Dti_qxs = torch.bmm(
        jacobian.transpose(1, 2), epsilon * tangent2.unsqueeze(-1)
    ).squeeze(-1)

    return Dti_pxs, Dti_qxs


def bilinear_interpolation(grid, uvs):
    """
    Perform bilinear interpolation on a 2D grid given UV coordinates.

    :param grid: A 2D grid of shape (H, W, C).
    :param uv: UV coordinates of shape (num_points, 2).
    :return: Interpolated values of shape (N, C).
    """
    H, W, C = grid.shape
    u, v = uvs[:, 0], uvs[:, 1]

    u = u * (W - 1)
    v = v * (H - 1)

    u0 = torch.floor(u).long()
    v0 = torch.floor(v).long()
    u1 = u0 + 1
    v1 = v0 + 1

    u0 = torch.clamp(u0, 0, W - 1)
    v0 = torch.clamp(v0, 0, H - 1)
    u1 = torch.clamp(u1, 0, W - 1)
    v1 = torch.clamp(v1, 0, H - 1)

    Ia = grid[v0, u0]
    Ib = grid[v1, u0]
    Ic = grid[v0, u1]
    Id = grid[v1, u1]

    wa = (u1 - u) * (v1 - v)
    wb = (u1 - u) * (v - v0)
    wc = (u - u0) * (v1 - v)
    wd = (u - u0) * (v - v0)

    interpolated = (
        wa.unsqueeze(-1) * Ia
        + wb.unsqueeze(-1) * Ib
        + wc.unsqueeze(-1) * Ic
        + wd.unsqueeze(-1) * Id
    )
    return interpolated
