"""Generator and discriminator used in MetricGAN

Authors:
* Szu-Wei Fu 2020
"""
import torch
import speechbrain as sb
from torch import nn
from torch.nn.utils import spectral_norm


def xavier_init_layer(
    in_size, out_size=None, spec_norm=True, layer_type=nn.Linear, **kwargs
):
    "Create a layer with spectral norm, xavier uniform init and zero bias"
    if out_size is None:
        out_size = in_size

    layer = layer_type(in_size, out_size, **kwargs)
    if spec_norm:
        layer = spectral_norm(layer)

    # Perform initialization
    nn.init.xavier_uniform_(layer.weight, gain=1.0)
    nn.init.zeros_(layer.bias)

    return layer


def shifted_sigmoid(x):
    "Computes the shifted sigmoid."
    return 1.2 / (1 + torch.exp(-(1 / 1.6) * x))


class Learnable_sigmoid(nn.Module):
    """Implementation of a leanable sigmoid.

    Arguments
    ---------
    in_features : int
        Input dimensionality
    """

    def __init__(self, in_features=257):
        super().__init__()
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True  # set requiresGrad to true!

        # self.scale = nn.Parameter(torch.ones(1))
        # self.scale.requiresGrad = True # set requiresGrad to true!

    def forward(self, x):
        """ Processes the input tensor x and returns an output tensor."""
        return 1.2 * torch.sigmoid(self.slope * x)


class EnhancementGenerator(nn.Module):
    """Simple LSTM for enhancement with custom initialization.

    Arguments
    ---------
    input_size : int
        Size of the input tensor's last dimension.
    hidden_size : int
        Number of neurons to use in the LSTM layers.
    num_layers : int
        Number of layers to use in the LSTM.
    dropout : int
        Fraction of neurons to drop during training.
    """

    def __init__(
        self, input_size=257, hidden_size=200, num_layers=2, dropout=0,
    ):
        super().__init__()
        self.activation = nn.LeakyReLU(negative_slope=0.3)

        self.blstm = sb.nnet.RNN.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
        )
        """
        Use orthogonal init for recurrent layers, xavier uniform for input layers
        Bias is 0
        """
        for name, param in self.blstm.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)

        self.linear1 = xavier_init_layer(400, 300, spec_norm=False)
        self.linear2 = xavier_init_layer(300, 257, spec_norm=False)

        self.Learnable_sigmoid = Learnable_sigmoid()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        """ Processes the input tensor x and returns an output tensor."""
        out, _ = self.blstm(x, lengths=lengths)

        out = self.linear1(out)
        out = self.activation(out)

        out = self.linear2(out)
        out = self.Learnable_sigmoid(out)

        return out


class MetricDiscriminator(nn.Module):
    """Metric estimator for enhancement training.

    Consists of:
     * four 2d conv layers
     * channel averaging
     * three linear layers

    Arguments
    ---------
    kernel_size : tuple
        The dimensions of the 2-d kernel used for convolution.
    base_channels : int
        Number of channels used in each conv layer.
    """

    def __init__(
        self, kernel_size=(5, 5), base_channels=15, activation=nn.LeakyReLU,
    ):
        super().__init__()

        self.activation = activation(negative_slope=0.3)

        self.BN = nn.BatchNorm2d(num_features=2, momentum=0.01)

        self.conv1 = xavier_init_layer(
            2, base_channels, layer_type=nn.Conv2d, kernel_size=kernel_size
        )
        self.conv2 = xavier_init_layer(
            base_channels, layer_type=nn.Conv2d, kernel_size=kernel_size
        )
        self.conv3 = xavier_init_layer(
            base_channels, layer_type=nn.Conv2d, kernel_size=kernel_size
        )
        self.conv4 = xavier_init_layer(
            base_channels, layer_type=nn.Conv2d, kernel_size=kernel_size
        )

        self.Linear1 = xavier_init_layer(base_channels, out_size=50)
        self.Linear2 = xavier_init_layer(in_size=50, out_size=10)
        self.Linear3 = xavier_init_layer(in_size=10, out_size=1)

    def forward(self, x):
        """ Processes the input tensor x and returns an output tensor."""
        out = self.BN(x)

        out = self.conv1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.activation(out)

        out = self.conv4(out)
        out = self.activation(out)

        out = torch.mean(out, (2, 3))

        out = self.Linear1(out)
        out = self.activation(out)

        out = self.Linear2(out)
        out = self.activation(out)

        out = self.Linear3(out)

        return out
