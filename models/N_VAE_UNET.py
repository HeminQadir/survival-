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
@alias("VAE_UNET")
class VAE_UNET(nn.Module):
    """
    Enhanced version of UNet which has residual units implemented with the ResidualUnit class.
    The residual part uses a convolution to change the input dimensions to match the output dimensions
    if this is necessary but will use nn.Identity if not.
    Refer to: https://link.springer.com/chapter/10.1007/978-3-030-12029-0_40.

    Each layer of the network has a encode and decode path with a skip connection between them. Data in the encode path
    is downsampled using strided convolutions (if `strides` is given values greater than 1) and in the decode path
    upsampled using strided transpose convolutions. These down or up sampling operations occur at the beginning of each
    block rather than afterwards as is typical in UNet implementations.

    To further explain this consider the first example network given below. This network has 3 layers with strides
    of 2 for each of the middle layers (the last layer is the bottom connection which does not down/up sample). Input
    data to this network is immediately reduced in the spatial dimensions by a factor of 2 by the first convolution of
    the residual unit defining the first layer of the encode part. The last layer of the decode part will upsample its
    input (data from the previous layer concatenated with data from the skip connection) in the first convolution. this
    ensures the final output of the network has the same shape as the input.

    Padding values for the convolutions are chosen to ensure output sizes are even divisors/multiples of the input
    sizes if the `strides` value for a layer is a factor of the input sizes. A typical case is to use `strides` values
    of 2 and inputs that are multiples of powers of 2. An input can thus be downsampled evenly however many times its
    dimensions can be divided by 2, so for the example network inputs would have to have dimensions that are multiples
    of 4. In the second example network given below the input to the bottom layer will have shape (1, 64, 15, 15) for
    an input of shape (1, 1, 240, 240) demonstrating the input being reduced in size spatially by 2**4.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        channels: sequence of channels. Top block first. The length of `channels` should be no less than 2.
        strides: sequence of convolution strides. The length of `stride` should equal to `len(channels) - 1`.
        kernel_size: convolution kernel size, the value(s) should be odd. If sequence,
            its length should equal to dimensions. Defaults to 3.
        up_kernel_size: upsampling convolution kernel size, the value(s) should be odd. If sequence,
            its length should equal to dimensions. Defaults to 3.
        num_res_units: number of residual units. Defaults to 0.
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        bias: whether to have a bias term in convolution blocks. Defaults to True.
            According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
            if a conv layer is directly followed by a batch norm layer, bias should be False.
        adn_ordering: a string representing the ordering of activation (A), normalization (N), and dropout (D).
            Defaults to "NDA". See also: :py:class:`monai.networks.blocks.ADN`.

    Examples::

        from monai.networks.nets import UNet

        # 3 layer network with down/upsampling by a factor of 2 at each layer with 2-convolution residual units
        net = VAE_UNET(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(4, 8, 16),
            strides=(2, 2, 1),
            num_res_units=2
        )

        # 5 layer network with simple convolution/normalization/dropout/activation blocks defining the layers
        net = VAE_UNET(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(4, 8, 16, 32, 64),
            strides=(2, 2, 2, 2, 1),
        )

    .. deprecated:: 0.6.0
        ``dimensions`` is deprecated, use ``spatial_dims`` instead.

    Note: The acceptable spatial size of input data depends on the parameters of the network,
        to set appropriate spatial size, please check the tutorial for more details:
        https://github.com/Project-MONAI/tutorials/blob/master/modules/UNet_input_size_constrains.ipynb.
        Typically, when using a stride of 2 in down / up sampling, the output dimensions are either half of the
        input when downsampling, or twice when upsampling. In this case with N numbers of layers in the network,
        the inputs must have spatial dimensions that are all multiples of 2^N.
        Usually, applying `resize`, `pad` or `crop` transforms can help adjust the spatial size of input data.

    """

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
    
        self.resolution = 12

        # The encoder path 
        self.conv_0 = Convolution(
            self.dimensions,
            in_channels=in_channels,
            out_channels=channels[0],
            strides=strides[0],
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=False and self.num_res_units == 0,
            is_transposed=False,
            adn_ordering=self.adn_ordering,
        )

        self.conv_1 = Convolution(
            self.dimensions,
            in_channels=channels[0],
            out_channels=channels[1],
            strides=strides[1],
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=False and self.num_res_units == 0,
            is_transposed=False,
            adn_ordering=self.adn_ordering,
        )    


        self.conv_2 = Convolution(
            self.dimensions,
            in_channels=channels[1],
            out_channels=channels[2],
            strides=strides[2],
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=False and self.num_res_units == 0,
            is_transposed=False,
            adn_ordering=self.adn_ordering,
        )   


        self.conv_3 = Convolution(
            self.dimensions,
            in_channels=channels[2],
            out_channels=channels[3],
            strides=strides[3],
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=False and self.num_res_units == 0,
            is_transposed=False,
            adn_ordering=self.adn_ordering,
        )   


        self.conv_4 = Convolution(
            self.dimensions,
            in_channels=channels[3],
            out_channels=channels[4],
            strides=strides[4],
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=False and self.num_res_units == 0,
            is_transposed=False,
            adn_ordering=self.adn_ordering,
        )  


        # The decoder path     
        self.conv_transpose_1 = Convolution(
            self.dimensions,
            in_channels=channels[-2], #+channels[-1],
            out_channels=channels[-3],
            strides=strides[-2],
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=False and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
            )
        

        self.conv_transpose_2 = Convolution(
            self.dimensions,
            in_channels=channels[-3], #*2,
            out_channels=channels[-4],
            strides=strides[-3],
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=False and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
            )


        self.conv_transpose_3 = Convolution(
            self.dimensions,
            in_channels=channels[-4], #*2,
            out_channels=channels[-5],
            strides=strides[-4],
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=False and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
            )
    

        self.conv_transpose_4 = Convolution(
            self.dimensions,
            in_channels=channels[-5], #*2,
            out_channels=out_channels,
            strides=strides[-5],
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=True and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
            )
        
        self.conv_mu = Convolution(
            self.dimensions,
            in_channels=channels[-1],
            out_channels=channels[-2],
            strides=strides[-1],
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=True and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
            )
        

        self.conv_logvar = Convolution(
            self.dimensions, 
            in_channels=channels[-1],
            out_channels=channels[-2],
            strides=strides[-1],
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=True and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
            )
        
        # Time Decoder (New)
        self.time_decoder_1 = Convolution(spatial_dims, channels[-2], channels[-3], 1, kernel_size, act, norm, dropout, bias)
        self.time_decoder_2 = Convolution(spatial_dims, channels[-3], channels[-4], 1, kernel_size, act, norm, dropout, bias)
        self.time_decoder_3 = Convolution(spatial_dims, channels[-4], channels[-5], 1, kernel_size, act, norm, dropout, bias)

        # Flatten and Fully Connected Layer to output exactly 2 values
        self.time_fc = nn.Linear(channels[-5], 2)  # Output exactly 2 neurons

        # **Softplus Activation for Final Time Output**
        self.softplus = nn.Softplus()

        

    # def reparameterize(self, mu, log_var):
    #     std = torch.exp(0.5 * log_var)
    #     eps = torch.randn_like(std)
    #     return mu + eps * std


    def reparameterize(self, mu, log_var):
        # Clamping log_var to avoid numerical instability
        log_var = torch.clamp(log_var, min=-10, max=10)
        std = torch.exp(0.5 * log_var) + 1e-6
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Iterate through the convolutional layers and apply them sequentially
        #print"I am in the model")
        # Encoder Part 
        x0 = self.conv_0(x)
        #printx0.shape)
        # torch.Size([4, 16, 48, 48, 48])
        x1 = self.conv_1(x0)
        #printx1.shape)
        # torch.Size([4, 32, 24, 24, 24])
        x2 = self.conv_2(x1)
        #printx2.shape)
        # torch.Size([4, 64, 12, 12, 12]) 
        x3 = self.conv_3(x2)
        #printx3.shape)
        # torch.Size([4, 128, 6, 6, 6])
        x4 = self.conv_4(x3)
        #printx4.shape)
        # torch.Size([4, 256, 6, 6, 6])
        
        # Latent Space 
        #printtext_proj.shape)
        # torch.Size([4, 256, 6, 6, 6])
        mu = self.conv_mu(torch.cat([x4], dim=1))
        logvar = self.conv_logvar(torch.cat([x4], dim=1))
        y = self.reparameterize(mu, logvar) 
        #printy.shape)
        #torch.Size([4, 128, 6, 6, 6])

        y_decoded = self.conv_transpose_1(y)
        y_decoded = self.conv_transpose_2(y_decoded)
        y_decoded = self.conv_transpose_3(y_decoded)
        y_decoded = self.conv_transpose_4(y_decoded)
        #printy.shape)

        # Time Decoder
        time_output = self.time_decoder_1(y)
        time_output = self.time_decoder_2(time_output)
        time_output = self.time_decoder_3(time_output)
        #[batch_size, channels[-5], H, W, D],
        # Global Average Pooling to remove spatial dimensions (H, W, D)
        time_output = torch.mean(time_output, dim=[2, 3, 4]) #[batch_size, channels[-5]]

        # Fully connected layer to get exactly 2 values
        time_output = self.time_fc(time_output)

        # Apply Softplus to ensure positive values
        weibull_params = self.softplus(time_output)
        print(f"weibill_params:{weibull_params}")
        
        #print"The end of the model")

        return y_decoded, mu, logvar, weibull_params
    


# Training 
# input size torch.Size([4, 1, 96, 96, 96])
# embed size torch.Size([4, 1, 1, 512])
# torch.Size([4, 16, 48, 48, 48])
# torch.Size([4, 32, 24, 24, 24])
# torch.Size([4, 64, 12, 12, 12])
# torch.Size([4, 128, 6, 6, 6])
# torch.Size([4, 256, 6, 6, 6])
# torch.Size([4, 256, 6, 6, 6])
# torch.Size([4, 128, 6, 6, 6])
# torch.Size([4, 64, 12, 12, 12])
# torch.Size([4, 32, 24, 24, 24])
# torch.Size([4, 16, 48, 48, 48])
# torch.Size([4, 2, 96, 96, 96])

# Validation 
# input size torch.Size([4, 1, 96, 96, 96])
# embed size torch.Size([4, 1, 1, 512])
# torch.Size([4, 16, 48, 48, 48])
# torch.Size([4, 32, 24, 24, 24])
# torch.Size([4, 64, 12, 12, 12])
# torch.Size([4, 128, 6, 6, 6])
# torch.Size([4, 256, 6, 6, 6])
# torch.Size([1, 256, 6, 6, 6])
# error 


