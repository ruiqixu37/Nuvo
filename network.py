import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def positional_encoding(x, degree=1):
    """
    Apply positional encoding to the input tensor x.

    :param x: Input tensor of shape (batch_size, 3).
    :param degree: Degree of positional encoding.
    :return: Positional encoded tensor.
    """
    if degree < 1:
        return x

    pe = [x]
    for d in range(1, degree + 1):
        for fn in [torch.sin, torch.cos]:
            pe.append(fn(2.0**d * math.pi * x))
    return torch.cat(pe, dim=-1)


class ChartAssignmentMLP(nn.Module):
    def __init__(
        self, input_dim=3, output_dim=8, hidden_dim=256, num_layers=8, degree=1
    ):
        """
        Initialize the MLP with positional encoding.

        :param input_dim: Dimension of the input, default is 3 for a 3D point.
        :param output_dim: Dimension of the output, which is the number of charts n.
        :param hidden_dim: Number of hidden units in each layer.
        :param num_layers: Number of fully-connected layers.
        :param degree: Degree of positional encoding.
        """
        super(ChartAssignmentMLP, self).__init__()

        self.degree = degree
        self.input_dim = input_dim * (
            2 * degree + 1
        )  # Adjust input_dim based on positional encoding

        layers = []
        layers.append(nn.Linear(self.input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the MLP.

        :param x: Input tensor of shape (batch_size, 3).
        :return: Output tensor of shape (batch_size, output_dim) with probabilities.
        """
        x = positional_encoding(x, self.degree)
        logits = self.model(x)
        probabilities = F.softmax(logits, dim=1)
        return probabilities


class TextureCoordinateMLP(nn.Module):
    def __init__(
        self,
        input_dim=3,
        output_dim=2,
        hidden_dim=256,
        num_layers=8,
        degree=4,
        n_charts=8,
    ):
        """
        Initialize the MLP with positional encoding for texture coordinate mapping.

        :param input_dim: Dimension of the input, default is 3 for a 3D point.
        :param output_dim: Dimension of the output, which is 2 for UV coordinates.
        :param hidden_dim: Number of hidden units in each layer.
        :param num_layers: Number of fully-connected layers.
        :param degree: Degree of positional encoding.
        :param n_charts: Number of charts (i.e., number of MLPs).
        """
        super(TextureCoordinateMLP, self).__init__()

        self.degree = degree
        self.input_dim = input_dim * (
            2 * degree + 1
        )  # Adjust input_dim based on positional encoding
        self.n_charts = n_charts

        self.model = nn.ModuleList()
        for _ in range(n_charts):
            layers = [nn.Linear(self.input_dim, hidden_dim), nn.ReLU()]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, output_dim))
            layers.append(nn.Sigmoid())  # To ensure output is between 0 and 1
            self.model.append(nn.Sequential(*layers))

    def forward(self, x):
        """
        Forward pass of the MLP.

        :param x: Input tensor of shape (batch_size, 3).
        :return: List of output tensors for each chart, each of shape (batch_size, 2).
        """
        x = positional_encoding(x, self.degree)
        outputs = [mlp(x) for mlp in self.mlps]
        return outputs


class SurfaceCoordinateMLP(nn.Module):
    def __init__(
        self, input_dim=2, output_dim=3, hidden_dim=256, num_layers=8, n_charts=8
    ):
        """
        Initialize the MLP for surface coordinate mapping.

        :param input_dim: Dimension of the input, default is 2 for a 2D texture coordinate.
        :param output_dim: Dimension of the output, which is 3 for 3D point coordinates.
        :param hidden_dim: Number of hidden units in each layer.
        :param num_layers: Number of fully-connected layers.
        :param n_charts: Number of charts (i.e., number of MLPs).
        """
        super(SurfaceCoordinateMLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_charts = n_charts

        self.mlps = nn.ModuleList()
        for _ in range(n_charts):
            layers = [nn.Linear(self.input_dim, hidden_dim), nn.ReLU()]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.mlps.append(nn.Sequential(*layers))

    def forward(self, x):
        """
        Forward pass of the MLP.

        :param x: Input tensor of shape (batch_size, 2).
        :return: List of output tensors for each chart, each of shape (batch_size, 3).
        """
        outputs = [mlp(x) for mlp in self.mlps]
        return outputs
