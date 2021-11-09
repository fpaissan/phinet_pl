from phinet_pl.model_utils import DepthwiseConv2d, SEBlock, ReLUMax, HSwish, correct_pad

import torch.nn as nn
import torch


class PhiNetConvBlock(nn.Module):
    """Implements PhiNet's convolutional block"""
    def __init__(self,
                 in_shape,
                 expansion,
                 stride,
                 filters,
                 has_se,
                 block_id=None,
                 res=True,
                 h_swish=True,
                 k_size=3,
                 dp_rate=0,
                 sd_p=1):
        """Defines the structure of the PhiNet conv block

        Args:
            in_shape ([Tuple]): [Input shape, as returned by Tensor.shape]
            expansion ([Int]): [Expansion coefficient]
            stride ([Int]): [Stride for conv block]
            filters ([Int]): [description]
            block_id ([Int]): [description]
            has_se (bool): [description]
            res (bool, optional): [description]. Defaults to True.
            h_swish (bool, optional): [description]. Defaults to True.
            k_size (int, optional): [description]. Defaults to 3.
            sd_p (float, optional): [stochastic depth probability]. Defaults to 0.
        """
        super(PhiNetConvBlock, self).__init__()
        self.skip_conn = False
        self.pdf = torch.distributions.bernoulli.Bernoulli(torch.Tensor([sd_p]))

        self._layers = torch.nn.ModuleList()
        in_channels = in_shape[0]
        # Define activation function
        if h_swish:
            activation = HSwish()
        else:
            activation = ReLUMax(6)

        if block_id:
            # Expand
            conv1 = nn.Conv2d(
                in_channels, int(expansion * in_channels),
                kernel_size=1,
                padding="same",
                bias=False,
            )

            bn1 = nn.BatchNorm2d(
                int(expansion * in_channels),
                eps=1e-3,
                momentum=0.999,
            )

            self._layers.append(conv1)
            self._layers.append(bn1)
            self._layers.append(activation)

        if stride == 2:
            pad = nn.ZeroPad2d(
                padding=correct_pad(in_shape, 3),
            )

            self._layers.append(pad)
        
        self._layers.append(nn.Dropout2d(dp_rate))

        d_mul = 1
        in_channels_dw = int(expansion * in_channels) if block_id else in_channels
        out_channels_dw = in_channels_dw * d_mul
        dw1 = DepthwiseConv2d(
            in_channels=in_channels_dw,
            depth_multiplier=d_mul,
            kernel_size=k_size,
            stride=stride,
            bias=False,
            padding="same" if stride == 1 else "valid",
            # name=prefix + 'depthwise'
        )

        bn_dw1 = nn.BatchNorm2d(
            out_channels_dw,
            eps=1e-3,
            momentum=0.999,
        )

        self._layers.append(dw1)
        self._layers.append(bn_dw1)
        self._layers.append(activation)

        if has_se:
            num_reduced_filters = max(1, int(in_channels * 0.25))
            se_block = SEBlock(int(expansion * in_channels), num_reduced_filters, h_swish=h_swish)
            self._layers.append(se_block)

        conv2 = nn.Conv2d(
            in_channels=int(expansion * in_channels),
            out_channels=filters,
            kernel_size=1,
            padding="same",
            bias=False,
        )

        bn2 = nn.BatchNorm2d(
            filters,
            eps=1e-3,
            momentum=0.999,
        )

        self._layers.append(conv2)
        self._layers.append(bn2)

        if res and in_channels == filters and stride == 1:
            self.skip_conn = True

    def forward(self, x):
        """Executes PhiNet's convolutional block

        Args:
            x ([Tensor]): [Conv block input]

        Returns:
            [Tensor]: [Output of convolutional block]
        """
        if self.skip_conn:
            inp = x

        for l in self._layers:
            # print(l, l(x).shape)
            x = l(x)

        if self.skip_conn:
            if self.training:
                if torch.equal(self.pdf.sample(), torch.ones(1)):
                    return x + inp
                else:
                    return inp
            else:
                return x + inp  # can also be x*self.sd_p + inp
        else:
            return x