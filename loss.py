import torch
import torch.nn as nn
import torch.nn.functional as F
from network import Nuvo


def three_two_three_loss(points, model: Nuvo):
    G = len(points)
    loss = 0
    for p in points:
        chart_probs = model.chart_assignment_mlp(p)
        for chart_idx in range(model.n_charts):
            pred_uv = model.texture_coordinate_mlp(p, chart_idx)
            reconstructed_p = model.surface_coordinate_mlp(pred_uv, chart_idx)
            loss += chart_probs[chart_idx] * F.mse_loss(p, reconstructed_p)
    loss /= G
    return loss


def two_three_two_loss(uvs, model: Nuvo):
    T = len(uvs)
    loss = 0
    for uv in uvs:
        for chart_idx in range(model.n_charts):
            pred_p = model.surface_coordinate_mlp(uv, chart_idx)
            reconstructed_uv = model.texture_coordinate_mlp(pred_p, chart_idx)
            loss += F.mse_loss(uv, reconstructed_uv)
    loss /= T
    return loss


def entropy_loss(uvs, model: Nuvo):
    T = len(uvs)
    loss = 0
    for uv in uvs:
        for chart_idx in range(model.n_charts):
            pred_p = model.surface_coordinate_mlp(uv, chart_idx)
            chart_probs = model.chart_assignment_mlp(pred_p)
            loss += -torch.sum(torch.log(chart_probs + 1e-6))
    loss /= T
    return loss


def surface_loss(uvs, points, model: Nuvo):
    G = len(points)
    loss = 0

    reconstructed_p = []
    for uv in uvs:
        for chart_idx in range(model.n_charts):
            pred_p = model.surface_coordinate_mlp(uv, chart_idx)
            reconstructed_p.append(pred_p)
    reconstructed_p = torch.stack(reconstructed_p)
    # get rid of possible duplicate reconstructed 3D points
    reconstructed_p = torch.unique(reconstructed_p, dim=0)
    T = len(reconstructed_p)

    for p in points:
        min_distance = float("inf")
        for reconstructed_point in reconstructed_p:
            distance = F.mse_loss(p, reconstructed_point)
            min_distance = min(min_distance, distance)
        loss += min_distance / G

    for reconstructed_point in reconstructed_p:
        min_distance = float("inf")
        for p in points:
            distance = F.mse_loss(p, reconstructed_point)
            min_distance = min(min_distance, distance)
        loss += min_distance / T
    return loss


def cluster_loss(points, model: Nuvo):
    G = len(points)
    loss = 0
    for p in points:
        chart_probs = model.chart_assignment_mlp(p)
        for chart_idx in range(model.n_charts):
            numerator, denominator = 0, 0
            for p_prime in points:
                chart_probs_prime = model.chart_assignment_mlp(p_prime)
                numerator += chart_probs_prime[chart_idx] * p_prime
                denominator += chart_probs_prime[chart_idx]
            centroid = numerator / denominator
            loss += chart_probs[chart_idx] * F.mse_loss(p, centroid)
    loss /= G
    return loss
