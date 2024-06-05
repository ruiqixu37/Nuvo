import torch
import torch.nn as nn
import torch.nn.functional as F
from network import Nuvo
from utils import compute_uv_vectors, bilinear_interpolation


def compute_loss(conf, points, normals, uvs, model, sigma, normal_maps):
    three_two_three = three_two_three_loss(points, model)
    two_three_two = two_three_two_loss(uvs, model)
    entropy = entropy_loss(uvs, model)
    surface = surface_loss(uvs, points, model)
    cluster = cluster_loss(points, model)
    conformal = conformal_loss(points, normals, model)
    stretch = stretch_loss(points, normals, sigma, model)
    texture = texture_loss(points, normal_maps, normals, model)

    loss = (
        conf.loss.three_two_three * three_two_three
        + conf.loss.two_three_two * two_three_two
        + conf.loss.entropy * entropy
        + conf.loss.surface * surface
        + conf.loss.cluster * cluster
        + conf.loss.conformal * conformal
        + conf.loss.stretch * stretch
        + conf.loss.texture * texture
    )

    loss_dict = {
        "three_two_three": three_two_three * conf.loss.three_two_three,
        "two_three_two": two_three_two * conf.loss.two_three_two,
        "entropy": entropy * conf.loss.entropy,
        "surface": surface * conf.loss.surface,
        "cluster": cluster * conf.loss.cluster,
        "conformal": conformal * conf.loss.conformal,
        "stretch": stretch * conf.loss.stretch,
        "texture": texture * conf.loss.texture,
        "loss_combined": loss,
    }

    return loss_dict


def three_two_three_loss(points, model: Nuvo):
    loss = 0
    chart_probs = model.chart_assignment_mlp(points)
    for chart_idx in range(model.num_charts):
        pred_uv = model.texture_coordinate_mlp(points, chart_idx)
        reconstructed_p = model.surface_coordinate_mlp(pred_uv, chart_idx)
        loss += (
            chart_probs[:, chart_idx] * ((points - reconstructed_p).norm(dim=1).pow(2))
        ).mean()
    return loss


def two_three_two_loss(uvs, model: Nuvo):
    loss = 0
    for chart_idx in range(model.num_charts):
        pred_p = model.surface_coordinate_mlp(uvs, chart_idx)
        reconstructed_uv = model.texture_coordinate_mlp(pred_p, chart_idx)
        loss += (uvs - reconstructed_uv).norm(dim=1).pow(2).mean()
    return loss


def entropy_loss(uvs, model: Nuvo):
    T = len(uvs)
    loss = 0
    for chart_idx in range(model.num_charts):
        pred_p = model.surface_coordinate_mlp(uvs, chart_idx)
        chart_probs = model.chart_assignment_mlp(pred_p)
        loss += -torch.sum(torch.log(chart_probs + 1e-6))
    loss /= T
    return loss


def surface_loss(uvs, points, model: Nuvo):
    loss = 0

    reconstructed_p = torch.tensor([]).to(points.device)
    for chart_idx in range(model.num_charts):
        pred_p = model.surface_coordinate_mlp(uvs, chart_idx)
        reconstructed_p = torch.cat((reconstructed_p, pred_p), dim=0)

    squared_dists = torch.cdist(points, reconstructed_p).pow(2)
    min_squared_dists = torch.min(squared_dists, dim=1)[0]
    loss += min_squared_dists.mean()
    min_squared_dists_reconstructed = torch.min(squared_dists, dim=0)[0]
    loss += min_squared_dists_reconstructed.mean()
    return loss


def cluster_loss(points, model: Nuvo):
    chart_probs = model.chart_assignment_mlp(points)
    numerators = torch.matmul(chart_probs.t(), points)
    denominators = chart_probs.sum(dim=0)
    centroids = numerators / denominators[:, None]
    squared_cidsts = torch.cdist(points, centroids).pow(2)
    loss = (squared_cidsts * chart_probs).mean()

    return loss


def conformal_loss(points, normals, model: Nuvo):
    loss = 0
    chart_probs = model.chart_assignment_mlp(points)
    for chart_idx in range(model.num_charts):
        Dti_pxs, Dti_qxs = compute_uv_vectors(
            model.texture_coordinate_mlp, points, normals, chart_idx
        )
        cosine_similarity = torch.sum(Dti_pxs * Dti_qxs, dim=1) / (
            torch.norm(Dti_pxs, dim=1) * torch.norm(Dti_qxs, dim=1)
        )
        loss += (chart_probs[:, chart_idx] * (cosine_similarity**2)).mean()
    return loss


def stretch_loss(points, normals, sigma: nn.Parameter, model: Nuvo):
    loss = 0
    chart_probs = model.chart_assignment_mlp(points)
    for chart_idx in range(model.num_charts):
        Dti_pxs, Dti_qxs = compute_uv_vectors(
            model.texture_coordinate_mlp, points, normals, chart_idx
        )
        # pad with zeros to make the cross product work
        Dti_pxs = torch.cat(
            (Dti_pxs, torch.zeros(Dti_pxs.shape[0], 1, device=Dti_pxs.device)), 1
        )
        Dti_qxs = torch.cat(
            (Dti_qxs, torch.zeros(Dti_qxs.shape[0], 1, device=Dti_qxs.device)), 1
        )
        area = torch.norm(torch.linalg.cross(Dti_pxs, Dti_qxs), dim=1)
        loss += (chart_probs[:, chart_idx] * ((area - sigma) ** 2)).mean()
    return loss


def texture_loss(points, normal_grids, normals, model: Nuvo):
    """
    Compute the texture loss according to the given formula.
    :param points: A list of sampled 3D points from the scene. Shape (G, 3).
    :param normal_grids: A list of 2D grids representing normal maps for each chart. Shape (num_charts, H, W, 3).
    :param normals: A list of surface normals for each point. Shape (G, 3).
    :param model: The Nuvo model.
    :return: The texture loss.
    """
    loss = 0
    chart_probs = model.chart_assignment_mlp(points)
    for chart_idx in range(model.num_charts):
        uvs = model.texture_coordinate_mlp(points, chart_idx)
        normal_map = normal_grids[chart_idx]
        pred_normals = bilinear_interpolation(normal_map, uvs)
        loss += (
            chart_probs[:, chart_idx] * (pred_normals - normals).norm(dim=1).pow(2)
        ).mean()
    return loss
