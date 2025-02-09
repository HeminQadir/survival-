from __future__ import annotations

import warnings
from collections.abc import Sequence

import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
#from .convoluions import Convolution
from monai.networks.layers.factories import Act, Norm
#from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import alias, export
import inspect


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        This class defines a generic one layer feed-forward neural network for embedding input data of
        dimensionality input_dim to an embedding space of dimensionality emb_dim.
        '''
        self.input_dim = input_dim
        
        # define the layers for the network
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        
        # create a PyTorch sequential model consisting of the defined layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # flatten the input tensor
        x = x.view(-1, self.input_dim)
        # apply the model layers to the flattened tensor
        return self.model(x)
    

@export("monai.networks.nets")
@alias("VAE_GAN")
class VAE_GAN(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Sequence[int] | int = 3,
        up_kernel_size: Sequence[int] | int = 3,
        num_res_units: int = 0,
        act: tuple | str = Act.PRELU,
        norm: tuple | str = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
    ) -> None:
        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if isinstance(kernel_size, Sequence) and len(kernel_size) != spatial_dims:
            raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence) and len(up_kernel_size) != spatial_dims:
            raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
    

        # Encoder with 5 convolutional layers
        self.encoder = nn.Sequential(
            Convolution(self.dimensions, in_channels=in_channels, out_channels=channels[0], strides=strides[0],
                        kernel_size=self.up_kernel_size,act=self.act,norm=self.norm, dropout=self.dropout,
                        bias=self.bias, conv_only=False and self.num_res_units == 0, is_transposed=False,adn_ordering=self.adn_ordering),

            Convolution(self.dimensions, in_channels=channels[0], out_channels=channels[1], strides=strides[1],
                        kernel_size=self.up_kernel_size,act=self.act,norm=self.norm, dropout=self.dropout,
                        bias=self.bias, conv_only=False and self.num_res_units == 0, is_transposed=False,adn_ordering=self.adn_ordering),

            Convolution(self.dimensions, in_channels=channels[1], out_channels=channels[2], strides=strides[2],
                        kernel_size=self.up_kernel_size,act=self.act,norm=self.norm, dropout=self.dropout,
                        bias=self.bias, conv_only=False and self.num_res_units == 0, is_transposed=False,adn_ordering=self.adn_ordering),

            Convolution(self.dimensions, in_channels=channels[2], out_channels=channels[3], strides=strides[3],
                        kernel_size=self.up_kernel_size,act=self.act,norm=self.norm, dropout=self.dropout,
                        bias=self.bias, conv_only=False and self.num_res_units == 0, is_transposed=False,adn_ordering=self.adn_ordering),

            Convolution(self.dimensions, in_channels=channels[3], out_channels=channels[4], strides=strides[4],
                        kernel_size=self.up_kernel_size,act=self.act,norm=self.norm, dropout=self.dropout,
                        bias=self.bias, conv_only=False and self.num_res_units == 0, is_transposed=False,adn_ordering=self.adn_ordering)
                        )
        

        self.conv_mu = Convolution(self.dimensions, in_channels=channels[-1], out_channels=channels[-2],strides=strides[-1],
                        kernel_size=self.up_kernel_size,act=self.act,norm=self.norm,dropout=self.dropout,bias=self.bias,
                        conv_only=True and self.num_res_units == 0, is_transposed=True, adn_ordering=self.adn_ordering)

        self.conv_logvar = Convolution(self.dimensions, in_channels=channels[-1], out_channels=channels[-2],strides=strides[-1],
                        kernel_size=self.up_kernel_size,act=self.act,norm=self.norm,dropout=self.dropout,bias=self.bias,
                        conv_only=True and self.num_res_units == 0, is_transposed=True, adn_ordering=self.adn_ordering)
        
        # Decoder with 5 convolutional layers
        self.decoder = nn.Sequential(
            Convolution(self.dimensions, in_channels=channels[-2], out_channels=channels[-3],strides=strides[-2],
                        kernel_size=self.up_kernel_size,act=self.act,norm=self.norm,dropout=self.dropout,bias=self.bias,
                        conv_only=False and self.num_res_units == 0, is_transposed=True, adn_ordering=self.adn_ordering),

            Convolution(self.dimensions, in_channels=channels[-3], out_channels=channels[-4],strides=strides[-3],
                        kernel_size=self.up_kernel_size,act=self.act,norm=self.norm,dropout=self.dropout,bias=self.bias,
                        conv_only=False and self.num_res_units == 0, is_transposed=True, adn_ordering=self.adn_ordering),

            Convolution(self.dimensions, in_channels=channels[-4], out_channels=channels[-5],strides=strides[-4],
                        kernel_size=self.up_kernel_size,act=self.act,norm=self.norm,dropout=self.dropout,bias=self.bias,
                        conv_only=False and self.num_res_units == 0, is_transposed=True, adn_ordering=self.adn_ordering),

            Convolution(self.dimensions, in_channels=channels[-5], out_channels=out_channels,strides=strides[-5],
                        kernel_size=self.up_kernel_size,act=self.act,norm=self.norm,dropout=self.dropout,bias=self.bias,
                        conv_only=False and self.num_res_units == 0, is_transposed=True, adn_ordering=self.adn_ordering)                                                            
           )
        
        # Discriminator with 4 Conv3D layers
        self.discriminator = nn.Sequential(
            nn.Conv3d(out_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        log_var = torch.clamp(log_var, min=-10, max=10)
        std = torch.exp(0.5 * log_var) + 1e-6
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        x_encoded = self.encoder(x)
        mu = self.conv_mu(torch.cat([x_encoded], dim=1))
        logvar = self.conv_logvar(torch.cat([x_encoded], dim=1))
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        disc_pred = self.discriminator(x_reconstructed)
        return x_reconstructed, mu, logvar, disc_pred



