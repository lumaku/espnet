import logging

import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict

from espnet.nets.pytorch_backend.nets_utils import to_device

class SincConv(torch.nn.Module):
    """Sinc Convolution

    :param int in_channels: number of input channels (currently the same filters are applied to all input channels)
    :param int out_channels: number of output channels (i.e. number of filters)
    :param int kernel_size: kernel size (i.e. length of each filter)
    :param int stride: see torch.nn.functional.conv1d
    :param int padding: see torch.nn.functional.conv1d
    :param int dilation: see torch.nn.functional.conv1d
    :param int window_func: window function to use on the filter. Possible values are 'hamming', 'none'
    :param int sample_rate: sample rate of the input data
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, window_func='hamming', sample_rate=16000):
        super(SincConv, self).__init__()

        window_funcs = {
        'none' : self.none_window,
        'hamming' : self.hamming_window,
        }
        if window_func not in window_funcs:
            logging.error("SincConv error: window function has to be one of %s"
                ", using hamming instead" % str(list( window_funcs.keys() )) )
            window_func = 'hamming'
        self.window_func = window_funcs[window_func]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.sample_rate = sample_rate

        if self.kernel_size%2 == 0:
            logging.warn('SincConv: Even kernel size, adding one to make it odd')
            self.kernel_size += 1


        N = self.kernel_size//2
        self._x = 2 * np.pi * torch.linspace(1, N, N)
        self._window = self.window_func(torch.linspace(1, N, N))

        #init gets overwritten by E2E network, but is still required to calculate output dim
        self.init_filters()

    @staticmethod
    def sinc(x):
        x2 = x+1e-6
        return torch.sin(x2)/x2

    @staticmethod
    def none_window(x):
        return torch.ones_like(x)

    @staticmethod
    def hamming_window(x):
        L = 2*x.size(0)+1
        x = x.flip(0)
        return 0.54 - 0.46*torch.cos(2*np.pi*x / L)

    def init_filters(self):
        def mel(x): return 1125*np.log(1 + x/700)
        def hz(x): return 700*(np.exp(x/1125) -1)

        # mel filter bank
        fs = torch.linspace(mel(30), mel(self.sample_rate*0.5), self.out_channels+2)
        fs = hz(fs) / self.sample_rate
        f1,f2 = fs[:-2],fs[2:]
        self.f = torch.nn.Parameter(torch.stack([f1,f2], dim=1), requires_grad=True)

    def _create_filters(self, device):
        f_mins = torch.abs(self.f[:,0])
        f_maxs = torch.abs(self.f[:,0]) + torch.abs(self.f[:,1] - self.f[:,0])

        # Faster implementation. Heavily inspired by Ravanelli et al.
        # https://github.com/mravanelli/SincNet
        self._x = self._x.to(device)
        self._window = self._window.to(device)

        f_mins_x = torch.matmul(f_mins.view(-1,1), self._x.view(1,-1))
        f_maxs_x = torch.matmul(f_maxs.view(-1,1), self._x.view(1,-1))

        kernel = ( torch.sin(f_maxs_x) - torch.sin(f_mins_x) ) / (0.5*self._x)
        kernel = kernel * self._window

        kernel_left = kernel.flip(1)
        kernel_center = (2*f_maxs - 2*f_mins).unsqueeze(1)
        kernel_right = kernel
        filters = torch.cat([kernel_left, kernel_center, kernel], dim=1)

        filters = filters.view(filters.size(0), 1, filters.size(1))
        self.sinc_filters = filters

    def forward(self, xs):
        """SincConv forward

        :param torch.Tensor xs: batch of input data (B, C_in, D_in)
        :return: batch of output data (B, C_out, D_out)
        :rtype: torch.Tensor
        """
        #logging.info(self.__class__.__name__ + ' current filters: ' + str(self.f))

        self._create_filters(xs.device)
        xs = F.conv1d(
        	xs,
        	self.sinc_filters,
        	padding=self.padding,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.in_channels
        )
        return xs

    def get_odim(self, idim):
        D_out = idim+2*self.padding-self.dilation*(self.kernel_size-1) -1
        D_out = int(np.floor(D_out/self.stride)) +1
        return D_out


class LogCompression(torch.nn.Module):
    """Log Compression Activation
    Activation function log(abs(x) + 1)
    """
    def __init__(self):
        super(LogCompression, self).__init__()

    def forward(self, x):
        return torch.log(torch.abs(x) + 1)


class LightweightSincConvs(torch.nn.Module):
    """Lightweight Sinc Convolutions.

    https://arxiv.org/abs/2010.07597

    :param in_channels: number of output channels (i.e. number of filters), default: 1
    """
    def __init__(self, in_channels=1):
        super(LightweightSincConvs, self).__init__()
        self.pointwise_convolution = True
        self.pointwise_convolution = False
        self.pointwise_grouping = [0, 0, 0]
        self._create_sinc_convs(in_channels)

    def _create_sinc_convs(self, in_channels):
        self.in_channels = in_channels
        blocks = []

        #SincConvBlock
        out_channels = 128
        self.sinc_filters = SincConv(in_channels, out_channels, 101, 1)
        block = torch.nn.Sequential(
                self.sinc_filters,
                LogCompression(),
                #torch.nn.ReLU(),
                torch.nn.BatchNorm1d(out_channels, affine=True),
                torch.nn.AvgPool1d(2),
        )
        blocks.append(block)
        in_channels = out_channels #in for next block

        # First convolutional block, connects the sinc output to the front-end "body"
        out_channels = 128
        block = OrderedDict([
            ('depthwise', torch.nn.Conv1d(in_channels, out_channels, 25, 2, groups=in_channels)),
            ('pointwise', torch.nn.Conv1d(out_channels, out_channels, 1, 1, groups=1)),
            ('activation', torch.nn.LeakyReLU()),
            ('batchnorm', torch.nn.BatchNorm1d(out_channels, affine=True)),
            ('avgpool', torch.nn.AvgPool1d(2)),
            ('dropout', torch.nn.Dropout(0.1)),
        ])
        if not self.pointwise_convolution:
            del block['pointwise']
        blocks.append(torch.nn.Sequential(block))
        in_channels = out_channels

        # Second convolutional block, multiple convolutional layers
        out_channels = 256
        for grouping in self.pointwise_grouping:
            block = OrderedDict()
            block['depthwise'] = torch.nn.Conv1d(in_channels, out_channels, 9, 1, groups=in_channels)
            if self.pointwise_convolution:
                block['pointwise'] = torch.nn.Conv1d(in_channels, out_channels, 1, 1, groups=grouping)
            block['activation'] = torch.nn.LeakyReLU()
            block['batchnorm'] = torch.nn.BatchNorm1d(out_channels, affine=True)
            block['dropout'] = torch.nn.Dropout(0.15)
            blocks.append(torch.nn.Sequential(block))
            in_channels = out_channels

        # Third Convolutional block, acts as coupling to encoder
        out_channels = 256
        block = OrderedDict([
            ('depthwise', torch.nn.Conv1d(in_channels, out_channels, 7, 1, groups=in_channels)),
            ('pointwise', torch.nn.Conv1d(out_channels, out_channels, 1, 1, groups=1)),
            ('activation', torch.nn.LeakyReLU()),
            ('batchnorm', torch.nn.BatchNorm1d(out_channels, affine=True)),
        ])
        if not self.pointwise_convolution:
            del block['pointwise']
        blocks.append(torch.nn.Sequential(block))

        self.blocks = torch.nn.ModuleList(blocks)
        self.out_channels = out_channels

    def init_sinc_convs(self):
        self.sinc_filters.init_filters()
        for block in self.blocks:
            if type(block) is not torch.nn.Sequential:
                block = [block]
            for layer in block:
                if type(layer) == torch.nn.BatchNorm1d and layer.affine:
                    layer.weight.data[:] = 1.
                    layer.bias.data[:] = 0.

    def forward(self, xs_pad, ilens, **kwargs):
        """SincConv forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, C_in*D_in)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: batch of padded hidden state sequences (B, Tmax, C_out*D_out)
        :rtype: torch.Tensor
        """
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        #abbreviations for dims:
        #B - batch, T - timesteps, C_in/out - in/out-channels, D_in/out - in/out_features

        #---TRANSFORM: (B, T, C_in*D_in) -> (B*T, C_in, D_in)
        B, T, CD_in = xs_pad.size()
        xs_pad = xs_pad.view(B*T, self.in_channels, CD_in//self.in_channels)

        #---FORWARD
        #logging.info(self.__class__.__name__ + ' input lengths: ' + str(xs_pad.size()))
        for block in self.blocks:
            xs_pad = block.forward(xs_pad)
            #logging.info(self.__class__.__name__ + ' input lengths: ' + str(xs_pad.size()))

        #---TRANSFORM: (B*T, C_out, D_out) -> (B, T, C_out*D_out)
        _,C_out,D_out = xs_pad.size()
        xs_pad = xs_pad.view(B, T, C_out*D_out)
        return xs_pad, ilens, None  # no state in this layer

    def get_odim(self, idim):
        #test dim here is (1,T,idim), T set to idim without any special reason
        in_test = torch.zeros((1,idim,idim))
        out,_,_ = self.forward(in_test, [idim])
        logging.info(self.__class__.__name__ + ' output dimensions: ' + str(out.size(2)))
        return out.size(2)


