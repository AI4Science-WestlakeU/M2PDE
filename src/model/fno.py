import torch
from torch import nn, einsum
from torch import Tensor
import torch.nn.functional as F
from functools import partial
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import math
from einops import rearrange, reduce


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Conv1dFCLayer(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_fn=nn.Identity,
        weight_norm: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_features
        self.out_channels = out_features
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=1, bias=True)
        # self.conv = nn.Conv1d(in_features, out_features, kernel_size=3,padding=1, bias=True)
        self.reset_parameters()
        self.activation_fn = activation_fn
        if weight_norm:
            logger.warn("Weight norm not supported for Conv FC layers")

    def reset_parameters(self) -> None:
        nn.init.constant_(self.conv.bias, 0)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.activation_fn is not nn.Identity:
            x = self.activation_fn(x)
        return x


class Conv2dFCLayer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_fn=nn.Identity,
        activation_par: Optional[nn.Parameter] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.reset_parameters()
        self.activation_fn = activation_fn

    def reset_parameters(self) -> None:
        nn.init.constant_(self.conv.bias, 0)
        self.conv.bias.requires_grad = False
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.activation_fn is not nn.Identity:
            x = self.activation_fn(x)
        return x


class Conv3dFCLayer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_fn=nn.Identity,
        activation_par: Optional[nn.Parameter] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True)
        self.reset_parameters()
        self.activation_fn = activation_fn

    def reset_parameters(self) -> None:
        nn.init.constant_(self.conv.bias, 0)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.activation_fn is not nn.Identity:
            x = self.activation_fn(x)
        return x


class Freq1dLinear(nn.Module):
    def __init__(self, in_channel, modes1):
        super().__init__()
        self.modes1 = modes1
        scale = 1 / (in_channel + 2 * modes1)
        self.weights = nn.Parameter(scale * torch.randn(in_channel, 2 * modes1, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(1, 2 * modes1, dtype=torch.float32))

    def forward(self, x):
        B = x.shape[0]
        h = torch.einsum("tc,cm->tm", x, self.weights) + self.bias
        h = h.reshape(B, self.modes1, 1, 2)
        return torch.view_as_complex(h)


class Freq2dLinear(nn.Module):
    def __init__(self, in_channel, modes1, modes2):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channel + 4 * modes1 * modes2)
        self.weights = nn.Parameter(scale * torch.randn(in_channel, 4 * modes1 * modes2, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(1, 4 * modes1 * modes2, dtype=torch.float32))

    def forward(self, x):
        B = x.shape[0]
        h = torch.einsum("tc,cm->tm", x, self.weights) + self.bias
        h = h.reshape(B, self.modes1, self.modes2, 2, 2)
        return torch.view_as_complex(h)


class Freq3dLinear(nn.Module):
    def __init__(self, in_channel, modes1, modes2, modes3):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        scale = 1 / (in_channel + 8 * modes1 * modes2 * modes3)
        self.weights = nn.Parameter(scale * torch.randn(in_channel, 8 * modes1 * modes2 * modes3, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(1, 8 * modes1 * modes2 * modes3, dtype=torch.float32))

    def forward(self, x):
        B = x.shape[0]
        h = torch.einsum("tc,cm->tm", x, self.weights) + self.bias
        h = h.reshape(B, self.modes1, self.modes2, self.modes3, 4, 2)
        return torch.view_as_complex(h)


class SpectralConv1d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, modes1: int, time_emb_dim=None):
        super().__init__()
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = 1 / (in_channels * out_channels)  #  ֤scale
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, 2, dtype=torch.float32)
        )
        self.cond_emb = Freq1dLinear(time_emb_dim, self.modes1)

    # Complex multiplication
    #     input  weight
    def compl_mul1d(self, input: Tensor, weights: Tensor, emb=None) -> Tensor:
        if emb is not None:
            input = input * emb.unsqueeze(1)
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        cweights = torch.view_as_complex(weights)
        return torch.einsum("bix,iox->box", input, cweights)

    def forward(self, x: Tensor, t=None) -> Tensor:
        if t is not None and self.cond_emb is not None:
            emb = self.cond_emb(t)[..., 0]

        bsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            bsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1] = self.compl_mul1d(x_ft[:, :, : self.modes1], self.weights1, emb)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class SpectralConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, modes1, modes2, time_emb_dim=None):
        super().__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2, dtype=torch.float32)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2, dtype=torch.float32)
        )
        self.cond_emb = Freq2dLinear(time_emb_dim, self.modes1, self.modes2) if time_emb_dim is not None else None

    # Complex multiplication
    def compl_mul2d(self, input: Tensor, weights: Tensor, emb=None) -> Tensor:
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        if emb is not None:
            input = input * emb.unsqueeze(1)
        cweights = torch.view_as_complex(weights)
        return torch.einsum("bixy,ioxy->boxy", input, cweights)

    def forward(self, x: Tensor, t=None) -> Tensor:
        if t is not None and self.cond_emb is not None:
            emb12 = self.cond_emb(t)
            emb1, emb2 = emb12[..., 0], emb12[..., 1]
        else:
            emb1, emb2 = None, None

        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1, emb1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2, emb2
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

        return x


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, time_emb_dim=None):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2, dtype=torch.float32)
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2, dtype=torch.float32)
        )
        self.weights3 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2, dtype=torch.float32)
        )
        self.weights4 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2, dtype=torch.float32)
        )
        self.cond_emb = (
            Freq3dLinear(time_emb_dim, self.modes1, self.modes2, self.modes3) if time_emb_dim is not None else None
        )

    # Complex multiplication
    def compl_mul3d(self, input, weights, emb=None):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        if emb is not None:
            input = input * emb.unsqueeze(1)
        cweights = torch.view_as_complex(weights)
        return torch.einsum("bixyz,ioxyz->boxyz", input, cweights)

    def forward(self, x, t=None):
        if t is not None and self.cond_emb is not None:
            emb1234 = self.cond_emb(t)
            emb1, emb2 = emb1234[..., 0], emb1234[..., 1]
            emb3, emb4 = emb1234[..., 2], emb1234[..., 3]
        else:
            emb1, emb2, emb3, emb4 = None, None, None, None
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], self.weights1, emb1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3], self.weights2, emb2
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3], self.weights3, emb3
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3], self.weights4, emb4
        )

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class Projection(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=None,
        n_dim=2,
        non_linearity=F.gelu,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.non_linearity = non_linearity
        # self.dropout = nn.Dropout2d(dropout)
        Conv = getattr(nn, f"Conv{n_dim}d")
        self.fc1 = Conv(in_channels, self.hidden_channels, 1)
        # self.fc1 = Conv(in_channels, self.hidden_channels,kernel_size=3,padding=1)
        self.fc2 = Conv(self.hidden_channels, out_channels, 1)
        # self.fc2 = Conv(self.hidden_channels, out_channels,kernel_size=3,padding=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.non_linearity(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        return x


class FNO1D(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        nr_fno_layers: int = 4,
        fno_layer_size: int = 32,
        fno_modes: Union[int, list] = 16,
        padding: Union[int, list] = 8,
        padding_type: str = "constant",
        coord_features: bool = False,
        activation_fn=F.gelu,
        cond_emb=None,
        time_input=False,
        sinusoidal_pos_emb_theta=10000,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.nr_fno_layers = nr_fno_layers
        self.fno_width = fno_layer_size
        self.coord_features = coord_features
        # Spectral modes to have weights
        if isinstance(fno_modes, int):
            fno_modes = [fno_modes]
        # Add relative coordinate feature
        if self.coord_features:
            self.in_channels = self.in_channels + 1
        self.activation_fn = activation_fn

        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        # Initial lift layer
        self.lift_layer = Conv1dFCLayer(self.in_channels, self.fno_width)
        self.decoder = Projection(in_channels=self.fno_width, out_channels=out_channels, n_dim=1)

        self.cond_emb = cond_emb
        # time embeddings
        if time_input:
            time_dim = self.fno_width * 4
            fourier_dim = self.fno_width
            sinu_pos_emb = SinusoidalPosEmb(fourier_dim, theta=sinusoidal_pos_emb_theta)
            self.time_fc = nn.Sequential(
                sinu_pos_emb, nn.Linear(fourier_dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim)
            )
        else:
            time_dim = None

        # Build Neural Fourier Operators
        for _ in range(self.nr_fno_layers):
            self.spconv_layers.append(
                SpectralConv1d(self.fno_width, self.fno_width, fno_modes[0], time_emb_dim=time_dim)
            )
            self.conv_layers.append(nn.Conv1d(self.fno_width, self.fno_width, 1))
            self.norm_layers.append(nn.GroupNorm(num_groups=4, num_channels=self.fno_width))
            # self.conv_layers.append(nn.Conv1d(self.fno_width,self.fno_width,kernel_size=3,padding=1))

        # Padding values for spectral conv
        if isinstance(padding, int):
            padding = [padding]
        self.pad = padding[:1]
        self.ipad = [-pad if pad > 0 else None for pad in self.pad]
        self.padding_type = padding_type

    def forward(self, x: Tensor, t=None, cond=None, *args) -> Tensor:
        # diffusion
        if isinstance(x, list) == False:
            if cond is not None:
                for i, field in enumerate(cond):
                    emb_field = self.cond_emb[i](field)
                    x = torch.cat((x, emb_field), dim=1)
        # surrogate
        else:
            x_new = self.cond_emb[0](x[0])
            for i, field in enumerate(x):
                if i == 0:
                    continue
                emb_field = self.cond_emb[i](field)
                x_new = torch.cat((x_new, emb_field), dim=1)
            x = x_new

        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.lift_layer(x)
        # (b,c,x)->b,c1,x
        x = F.pad(x, (0, self.pad[0]), mode=self.padding_type)

        t_emb = self.time_fc(t) if t is not None else None
        # Spectral layers
        for k, conv_w_n in enumerate(zip(self.spconv_layers, self.conv_layers, self.norm_layers)):
            conv, w, norm = conv_w_n
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(norm(conv(x, t_emb)) + w(x))  # Spectral Conv + GELU causes JIT issue!
            else:
                x = norm(conv(x, t_emb)) + w(x)
        x = x[..., : self.ipad[0]]
        x = self.decoder(x)
        return x

    def meshgrid(self, shape: List[int], device: torch.device):
        bsize, size_x = shape[0], shape[2]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1)
        return grid_x


class FNO2D(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        nr_fno_layers: int = 4,
        fno_layer_size: int = 32,
        fno_modes: Union[int, List[int]] = 16,
        padding: Union[int, List[int]] = 8,
        padding_type: str = "constant",
        activation_fn=F.gelu,
        coord_features: bool = False,
        cond_emb=None,
        time_input=False,
        sinusoidal_pos_emb_theta=10000,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.nr_fno_layers = nr_fno_layers
        self.fno_width = fno_layer_size
        self.coord_features = coord_features
        # Spectral modes to have weights
        if isinstance(fno_modes, int):
            fno_modes = [fno_modes, fno_modes]
        # Add relative coordinate feature
        if self.coord_features:
            self.in_channels = self.in_channels + 2
        self.activation_fn = activation_fn

        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        # Initial lift layer
        self.lift_layer = Conv2dFCLayer(self.in_channels, self.fno_width)
        self.decoder = Projection(in_channels=self.fno_width, out_channels=out_channels, n_dim=2)

        self.cond_emb = cond_emb
        # time embeddings
        if time_input:
            time_dim = self.fno_width * 4
            fourier_dim = self.fno_width
            sinu_pos_emb = SinusoidalPosEmb(fourier_dim, theta=sinusoidal_pos_emb_theta)
            self.time_fc = nn.Sequential(
                sinu_pos_emb, nn.Linear(fourier_dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim)
            )
        else:
            time_dim = None
        # Build Neural Fourier Operators
        for _ in range(self.nr_fno_layers):
            self.spconv_layers.append(
                SpectralConv2d(self.fno_width, self.fno_width, fno_modes[0], fno_modes[1], time_emb_dim=time_dim)
            )
            self.conv_layers.append(nn.Conv2d(self.fno_width, self.fno_width, kernel_size=1))
            self.norm_layers.append(nn.GroupNorm(num_groups=4, num_channels=self.fno_width))
            # self.conv_layers.append(nn.Conv2d(self.fno_width,self.fno_width,kernel_size=3,padding=1))

        # Padding values for spectral conv
        if isinstance(padding, int):
            padding = [padding, padding]
        padding = padding + [0, 0]
        self.pad = padding[:2]
        self.ipad = [-pad if pad > 0 else None for pad in self.pad]
        self.padding_type = padding_type

    def forward(self, x: Tensor, t=None, cond=None, *args) -> Tensor:
        # diffusion
        if isinstance(x, list) == False:
            if cond is not None:
                for i, field in enumerate(cond):
                    emb_field = self.cond_emb[i](field)
                    x = torch.cat((x, emb_field), dim=1)
        # surrogate
        else:
            x_new = self.cond_emb[0](x[0])
            for i, field in enumerate(x):
                if i == 0:
                    continue
                emb_field = self.cond_emb[i](field)
                x_new = torch.cat((x_new, emb_field), dim=1)
            x = x_new

        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.lift_layer(x)
        # (left, right, top, bottom)
        x = F.pad(x, (0, self.pad[0], 0, self.pad[1]), mode=self.padding_type)

        t_emb = self.time_fc(t) if t is not None else None
        # Spectral layers
        for k, conv_w_n in enumerate(zip(self.spconv_layers, self.conv_layers, self.norm_layers)):
            conv, w, norm = conv_w_n
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(norm(conv(x, t_emb)) + w(x))  # Spectral Conv + GELU causes JIT issue!
                # x = self.dropout(x)
            else:
                x = norm(conv(x, t_emb)) + w(x)

        # remove padding
        x = x[..., : self.ipad[1], : self.ipad[0]]
        x = self.decoder(x)
        return x

    def meshgrid(self, shape: List[int], device: torch.device):
        bsize, size_x, size_y = shape[0], shape[2], shape[3]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing="ij")
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1)
        return torch.cat((grid_x, grid_y), dim=1)


class FNO3D(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        nr_fno_layers: int = 4,
        fno_layer_size: int = 32,
        fno_modes: Union[int, List[int]] = 16,
        padding: Union[int, List[int]] = 8,
        padding_type: str = "constant",
        coord_features: bool = False,
        activation_fn=F.gelu,
        cond_emb=None,
        time_input=False,
        sinusoidal_pos_emb_theta=10000,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.nr_fno_layers = nr_fno_layers
        self.fno_width = fno_layer_size
        self.coord_features = coord_features
        # Spectral modes to have weights
        if isinstance(fno_modes, int):
            fno_modes = [fno_modes, fno_modes, fno_modes]
        # Add relative coordinate feature
        if self.coord_features:
            self.in_channels = self.in_channels + 3
        self.activation_fn = activation_fn

        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        # Initial lift layer
        self.lift_layer = Conv3dFCLayer(self.in_channels, self.fno_width)
        self.decoder = Projection(in_channels=self.fno_width, out_channels=out_channels, n_dim=3)

        self.cond_emb = cond_emb
        # time embeddings
        if time_input:
            time_dim = self.fno_width * 4
            fourier_dim = self.fno_width * 2
            sinu_pos_emb = SinusoidalPosEmb(fourier_dim, theta=sinusoidal_pos_emb_theta)
            self.time_fc = nn.Sequential(
                sinu_pos_emb, nn.Linear(fourier_dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim)
            )
        else:
            time_dim = None
        # Build Neural Fourier Operators
        for _ in range(self.nr_fno_layers):
            self.spconv_layers.append(
                SpectralConv3d(
                    self.fno_width, self.fno_width, fno_modes[0], fno_modes[1], fno_modes[2], time_emb_dim=time_dim
                )
            )
            self.conv_layers.append(nn.Conv3d(self.fno_width, self.fno_width, 1))
            self.norm_layers.append(nn.GroupNorm(num_groups=4, num_channels=self.fno_width))

        # Padding values for spectral conv
        if isinstance(padding, int):
            padding = [padding, padding, padding]
        padding = padding + [0, 0, 0]  # Pad with zeros for smaller lists
        self.pad = padding[:3]
        self.ipad = [-pad if pad > 0 else None for pad in self.pad]
        self.padding_type = padding_type

    def forward(self, x: Tensor, t=None, cond=None, *args) -> Tensor:
        # diffusion
        if isinstance(x, list) == False:
            if cond is not None:
                for i, field in enumerate(cond):
                    emb_field = self.cond_emb[i](field)
                    x = torch.cat((x, emb_field), dim=1)
        # surrogate
        else:
            x_new = self.cond_emb[0](x[0])
            for i, field in enumerate(x):
                if i == 0:
                    continue
                emb_field = self.cond_emb[i](field)
                x_new = torch.cat((x_new, emb_field), dim=1)
            x = x_new

        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.lift_layer(x)
        # (left, right, top, bottom, front, back)
        x = F.pad(
            x,
            (0, self.pad[0], 0, self.pad[1], 0, self.pad[2]),
            mode=self.padding_type,
        )

        t_emb = self.time_fc(t) if t is not None else None
        # Spectral layers
        for k, conv_w_n in enumerate(zip(self.spconv_layers, self.conv_layers, self.norm_layers)):
            conv, w, norm = conv_w_n
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(norm(conv(x, t_emb)) + w(x))  # Spectral Conv + GELU causes JIT issue!
            else:
                x = norm(conv(x, t_emb)) + w(x)

        x = x[..., : self.ipad[2], : self.ipad[1], : self.ipad[0]]
        x = self.decoder(x)
        return x

    def meshgrid(self, shape: List[int], device: torch.device):
        bsize, size_x, size_y, size_z = shape[0], shape[2], shape[3], shape[4]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        grid_z = torch.linspace(0, 1, size_z, dtype=torch.float32, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
        grid_z = grid_z.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
        return torch.cat((grid_x, grid_y, grid_z), dim=1)
