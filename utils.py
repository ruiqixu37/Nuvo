import trimesh
import numpy as np
import matplotlib
import random
import torch
import torch.nn as nn
import torch.autograd as autograd

from wandb import Object3D
from network import Nuvo


def set_all_seeds(seed):
    """
    Set all seeds for reproducibility.

    :param seed: The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_mesh(mesh):
    """
    Normalize the mesh to fit within the unit cube.

    :param mesh: The trimesh object.
    :return: The normalized mesh.
    """
    vertices = mesh.vertices
    min_bound = np.min(vertices, axis=0)
    max_bound = np.max(vertices, axis=0)
    scale = np.max(max_bound - min_bound)
    center = (max_bound + min_bound) / 2
    mesh.vertices = (vertices - center) / scale
    return mesh


def create_wandb_object(mesh, device, model: Nuvo):
    """
    Create a wandb object for visualization.

    :param mesh: The trimesh object.
    :param model: The Nuvo model.
    :return: A wandb object for visualization.
    """
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32).to(device)

    chart_indices = model.chart_assignment_mlp(vertices).argmax(dim=1)
    chart_indices = chart_indices.detach().cpu().numpy()
    # assign colors to vertices based on chart indices. Create distinct colors for each chart.
    hsv_colors = [(i / model.num_charts, 0.5, 0.5) for i in range(model.num_charts)]
    rgb_colors = [
        (255 * matplotlib.colors.hsv_to_rgb(hsv)).astype(int) for hsv in hsv_colors
    ]
    colors = [rgb_colors[chart_idx] for chart_idx in chart_indices]

    xyz_rgb = np.concatenate([mesh.vertices, colors], axis=1)
    wandb_obj = Object3D.from_numpy(xyz_rgb)

    # create mesh as well for debugging
    new_mesh = trimesh.Trimesh(
        vertices=mesh.vertices, faces=mesh.faces, vertex_colors=colors
    )

    return wandb_obj, new_mesh


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
    r1 = np.random.rand(num_points, 1)
    r2 = np.random.rand(num_points, 1)

    u = 1 - np.sqrt(r1)
    v = np.sqrt(r1) * (1 - r2)
    w = np.sqrt(r1) * r2


    # Calculate the points
    points = triangles[:, 0] * u + triangles[:, 1] * v + triangles[:, 2] * w

    normals = mesh.face_normals[triangle_indices]

    return points, normals


def create_rgb_maps(texture_map_res, mesh, device, model: Nuvo):
    """
    Given a mesh and a model that predicts uv coordinates, create RGB maps.

    :param texture_map_res: The resolution of the texture maps.
    :param mesh: The trimesh object.
    :param device: The device to use.
    :param model: The Nuvo model.
    :return: A list of RGB maps for each chart. Shape (num_charts, resolution, resolution, 3).
    """

    texture_maps = torch.zeros(
        (model.num_charts, texture_map_res, texture_map_res, 3), device=device
    )
    colors = torch.tensor(mesh.visual.vertex_colors, dtype=torch.float32, device=device)
    points = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    chart_indices = model.chart_assignment_mlp(points).argmax(dim=1)
    uvs = model.texture_coordinate_mlp(points, chart_indices).cpu().detach().numpy()
    uvs = (uvs * texture_map_res).astype(int)
    texture_maps[chart_indices, uvs[:, 0], uvs[:, 1], :] = colors[:, :3]

    return texture_maps


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
