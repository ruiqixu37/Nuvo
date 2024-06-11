import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


def positional_encoding(x, degree=1):
    """
    Apply positional encoding to the input tensor x.

    :param x: Input tensor of shape (batch_size, 2 | 3).
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

        self.procedural_parameters = None
        self.degree = degree
        self.input_dim = input_dim * (
            2 * degree + 1
        ) + 2 # Adjust input_dim based on positional encoding

        layers = []
        layers.append(nn.Linear(self.input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def update_procedural_parameters(self, parameters):
        self.procedural_parameters = parameters

    def forward(self, x):
        """
        Forward pass of the MLP.

        :param x: Input tensor of shape (batch_size, 3).
        :return: Output tensor of shape (batch_size, output_dim) with probabilities.
        """
        x = positional_encoding(x, self.degree)
        # concatenate procedural parameters with x
        repeated_procedural_parameters = self.procedural_parameters.repeat(x.shape[0], 1)
        x = torch.cat([x, repeated_procedural_parameters], dim=-1)
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
        num_charts=8,
    ):
        """
        Initialize the MLP with positional encoding for texture coordinate mapping.

        :param input_dim: Dimension of the input, default is 3 for a 3D point.
        :param output_dim: Dimension of the output, which is 2 for UV coordinates.
        :param hidden_dim: Number of hidden units in each layer.
        :param num_layers: Number of fully-connected layers.
        :param degree: Degree of positional encoding.
        :param num_charts: Number of charts (i.e., number of MLPs).
        """
        super(TextureCoordinateMLP, self).__init__()
        self.procedural_parameters = None
        self.degree = degree
        self.input_dim = input_dim * (
            2 * degree + 1
        ) + 2  # Adjust input_dim based on positional encoding
        self.num_charts = num_charts

        self.mlps = nn.ModuleList()
        for _ in range(num_charts):
            layers = [nn.Linear(self.input_dim, hidden_dim), nn.ReLU()]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, output_dim))
            layers.append(nn.Sigmoid())  # To ensure output is between 0 and 1
            self.mlps.append(nn.Sequential(*layers))

    def update_procedural_parameters(self, parameters):
        self.procedural_parameters = parameters

    def forward(self, x, mlp_idx: Union[int, torch.Tensor] = None):
        """
        Forward pass of the MLP.

        :param x: Input tensor of shape (batch_size, 3).
        :return: outputs tensor from the MLP at index mlp_idx, of shape (batch_size, 2).
        """
        x = positional_encoding(x, self.degree)
        # concatenate procedural parameters with x
        repeated_procedural_parameters = self.procedural_parameters.repeat(x.shape[0], 1)
        x = torch.cat([x, repeated_procedural_parameters], dim=-1)
        if isinstance(mlp_idx, int):
            output = self.mlps[mlp_idx](x)
        else:
            output = torch.stack(
                [self.mlps[idx](sample) for idx, sample in zip(mlp_idx, x)]
            )
        return output


class SurfaceCoordinateMLP(nn.Module):
    def __init__(
        self,
        input_dim=2,
        output_dim=3,
        hidden_dim=256,
        num_layers=8,
        degree=4,
        num_charts=8,
    ):
        """
        Initialize the MLP for surface coordinate mapping.

        :param input_dim: Dimension of the input, default is 2 for a 2D texture coordinate.
        :param output_dim: Dimension of the output, which is 3 for 3D point coordinates.
        :param hidden_dim: Number of hidden units in each layer.
        :param num_layers: Number of fully-connected layers.
        :param num_charts: Number of charts (i.e., number of MLPs).
        """
        super(SurfaceCoordinateMLP, self).__init__()
        self.procedural_parameters = None
        self.degree = degree
        self.input_dim = input_dim * (
            2 * degree + 1
        ) + 2  # Adjust input_dim based on positional encoding
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_charts = num_charts

        self.mlps = nn.ModuleList()
        for _ in range(num_charts):
            layers = [nn.Linear(self.input_dim, hidden_dim), nn.ReLU()]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.mlps.append(nn.Sequential(*layers))

    def update_procedural_parameters(self, parameters):
        self.procedural_parameters = parameters

    def forward(self, x, mlp_idx: int = None):
        """
        Forward pass of the MLP.

        :param x: Input tensor of shape (batch_size, 2).
        :return: outputs tensor from the MLP at index mlp_idx, of shape (batch_size, 3).
        """
        x = positional_encoding(x, self.degree)
        # concatenate procedural parameters with x
        repeated_procedural_parameters = self.procedural_parameters.repeat(x.shape[0], 1)
        x = torch.cat([x, repeated_procedural_parameters], dim=-1)
        output = self.mlps[mlp_idx](x)
        return output


class Nuvo(nn.Module):
    def __init__(
        self,
        input_dim=3,
        hidden_dim=256,
        num_layers=8,
        c_pe_degree=1,
        t_pe_degree=4,
        s_pe_degree=4,
        num_charts=8,
    ):
        super(Nuvo, self).__init__()
        self.procedural_parameters = None
        self.num_charts = num_charts

        self.chart_assignment_mlp = ChartAssignmentMLP(
            input_dim=input_dim,
            output_dim=num_charts,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            degree=c_pe_degree,
        )

        self.texture_coordinate_mlp = TextureCoordinateMLP(
            input_dim=input_dim,
            output_dim=2,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            degree=t_pe_degree,
            num_charts=num_charts,
        )

        self.surface_coordinate_mlp = SurfaceCoordinateMLP(
            input_dim=2,
            output_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            degree=s_pe_degree,
            num_charts=num_charts,
        )

        self.reset_weights()

    def reset_weights(self):
        for mlp in [
            self.chart_assignment_mlp,
            self.texture_coordinate_mlp,
            self.surface_coordinate_mlp,
        ]:
            for layer in mlp.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def update_procedural_parameters(self, parameters):
        self.procedural_parameters = parameters
        self.chart_assignment_mlp.update_procedural_parameters(parameters)
        self.texture_coordinate_mlp.update_procedural_parameters(parameters)
        self.surface_coordinate_mlp.update_procedural_parameters(parameters)

    def forward(self, x):
        pass
