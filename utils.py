import trimesh
import numpy as np
import matplotlib
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

from wandb import Object3D
from network import Nuvo
from PIL import Image
import os


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


def create_checkerboard_textures(size, num_squares, dark_colors):
    """
    Create a list of checkerboard textures as numpy arrays.

    :param size: The size of the texture (width, height).
    :param num_squares: The number of squares along one dimension of the checkerboard.
    :param dark_colors: A list of numpy arrays to be used as the dark colors in the checkerboard.
    :return: A list of numpy arrays representing the checkerboard textures.
    """
    textures = []
    width, height = size
    square_size = width // num_squares

    for dark_color in dark_colors:
        # Create a lighter color by increasing the brightness
        light_color = np.clip(dark_color * 1.2, 0, 255).astype(np.uint8)
        
        texture = np.zeros((height, width, 3), dtype=np.uint8)
        
        for i in range(num_squares):
            for j in range(num_squares):
                color = dark_color if (i + j) % 2 == 0 else light_color
                texture[i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size] = color
        
        textures.append(texture)

    return textures

      
def create_wandb_object(mesh, device, model: Nuvo):
    """
    Create a wandb object for visualization.

    :param mesh: The trimesh object.
    :param model: The Nuvo model.
    :return: A wandb object for visualization.
    """
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32).to(device)

    chart_indices = model.chart_assignment_mlp(vertices).argmax(dim=1)
    uvs = model.texture_coordinate_mlp(vertices, chart_indices)
    chart_indices_numpy = chart_indices.detach().cpu().numpy()

    # assign colors to vertices based on chart indices. Create distinct colors for each chart.
    hsv_colors = [(i / model.num_charts, 0.5, 0.5) for i in range(model.num_charts)]
    rgb_colors = [
        (255 * matplotlib.colors.hsv_to_rgb(hsv)).astype(int) for hsv in hsv_colors
    ]
    colors = [rgb_colors[chart_idx] for chart_idx in chart_indices_numpy]

    xyz_rgb = np.concatenate([mesh.vertices, colors], axis=1)
    wandb_obj = Object3D.from_numpy(xyz_rgb)

    textures = create_checkerboard_textures((256, 256), 16, rgb_colors)

    # save the generated checkerboard texture for debugging
    output_dir = "checkerboard_textures"
    os.makedirs(output_dir, exist_ok=True)
    for index, texture in enumerate(textures):
        Image.fromarray(texture).save(os.path.join(output_dir, f'checkerboard_texture_{index + 1}.png'))

    chart_colors = torch.zeros((vertices.shape[0], 3), dtype=torch.uint8).to(device)

    # Compute the per-vertex colors
    for chart_idx in range(model.num_charts):
        mask = chart_indices == chart_idx
        texture = torch.tensor(textures[chart_idx], dtype=torch.float32).to(device)
        uvs_masked = uvs[mask]
        vertices_colors = bilinear_interpolation(texture, uvs_masked)
        chart_colors[mask] = vertices_colors.to(torch.uint8)

    chart_colors = chart_colors.cpu().detach().numpy()

    new_mesh = trimesh.Trimesh(
        vertices=mesh.vertices, faces=mesh.faces, vertex_colors=chart_colors
    )

    return wandb_obj, new_mesh

def create_uv_mesh(mesh, device, model: Nuvo, conf, output_path, checkerboard_res=256, squares=16):
    """
    Create a UV mesh for the given 3D mesh.

    :param mesh: The trimesh object.
    :param device: The device to use.
    :param model: The Nuvo model.
    :param conf: The configuration object.
    :param output_dir: The output directory for saving textures.
    :param checkerboard_res: The resolution of the checkerboard textures.
    :param squares: The number of squares in the checkerboard.
    :return: A UV mesh for the given 3D mesh.
    """
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32).to(device)
    chart_indices = model.chart_assignment_mlp(vertices).argmax(dim=1)
    uvs = model.texture_coordinate_mlp(vertices, chart_indices)
    num_charts = int(conf.model.num_charts)

    # tile the uvs
    uvs = uvs.detach().cpu().numpy()
    # add the chart_indices to the uvs, and then normalize the uvs
    chart_indices = chart_indices.detach().cpu().numpy()
    uvs[:, 0] = uvs[:, 0] + chart_indices
    uvs[:, 0] = uvs[:, 0] / num_charts

    atlas_w = checkerboard_res * max(1, num_charts)
    atlas_h = checkerboard_res

    # Build a list of base (dark) colors (numpy arrays) using an HSV sweep
    dark_colors = [
        (255 * matplotlib.colors.hsv_to_rgb((c / max(1, num_charts), 0.6, 0.6))).astype(np.uint8)
        for c in range(num_charts)
    ]

    # Create per-chart checkerboard textures using the helper
    textures = create_checkerboard_textures((checkerboard_res, checkerboard_res), squares, dark_colors)

    atlas = Image.new("RGB", (atlas_w, atlas_h), (200, 200, 200))
    for c, tex in enumerate(textures):
        # tex is a numpy array HxWx3 (uint8)
        tile_img = Image.fromarray(tex)
        atlas.paste(tile_img, (c * checkerboard_res, 0))
    texvisuals = trimesh.visual.TextureVisuals(uv=uvs, image=atlas)
    mesh = trimesh.Trimesh(vertices=vertices.cpu().numpy(), faces=mesh.faces, visual=texvisuals)
    
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    mesh.export(output_path)

    return mesh

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
    triangle_vertex_indices = mesh.faces[triangle_indices]
    vertex_normals = mesh.vertex_normals[triangle_vertex_indices]

    # Sample random points in each triangle using barycentric coordinates
    r1 = np.random.rand(num_points, 1)
    r2 = np.random.rand(num_points, 1)

    u = 1 - np.sqrt(r1)
    v = np.sqrt(r1) * (1 - r2)
    w = np.sqrt(r1) * r2


    # Calculate the points
    points = triangles[:, 0] * u + triangles[:, 1] * v + triangles[:, 2] * w

    normals = (
        vertex_normals[:, 0] * u +
        vertex_normals[:, 1] * v +
        vertex_normals[:, 2] * w
    )
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

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
    Perform bilinear interpolation using PyTorch's grid_sample.

    :param grid: A 2D grid of shape (H, W, C).
    :param uvs: UV coordinates of shape (N, 2), where each row contains (u, v) in normalized [0, 1].
    :return: Interpolated values of shape (N, C).
    """
    H, W, C = grid.shape
    grid_tensor = grid.permute(2, 0, 1).unsqueeze(0)

    uvs_normalized = uvs * 2 - 1
    uvs_tensor = uvs_normalized.unsqueeze(0).unsqueeze(0)

    interpolated = F.grid_sample(grid_tensor, uvs_tensor, align_corners=True, mode='bilinear')
    return interpolated.permute(3, 1, 2, 0).squeeze(-1).squeeze(-1)  # Shape: (N, C)
