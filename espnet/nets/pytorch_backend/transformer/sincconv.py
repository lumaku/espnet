import logging

import numpy as np
import torch
import torch.nn.functional as F


from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class SincConvs(torch.nn.Module):
    """Sinc Convolution for the transformer network

    :param int idim: input dim
    :param int odim: output dim
    :param float dropout_rate: dropout rate
    """

    def __init__(self, idim, attention_dim, dropout_rate, pos_enc_class, positional_dropout_rate):
        super(SincConvs, self).__init__()
        from espnet.nets.pytorch_backend.sincconv import LightweightSincConvs
        self.sinc_convs = LightweightSincConvs(in_channels=1)

        idim = self.sinc_convs.get_odim(idim)
        self.embed = torch.nn.Sequential(
            torch.nn.Linear(idim, attention_dim),
            torch.nn.LayerNorm(attention_dim),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU(),
            pos_enc_class(attention_dim, positional_dropout_rate)
        )

    def init_sinc_convs(self):
        self.sinc_convs.init_filters()

    def forward(self, xs_pad, ilens):
        xs_pad,ilens,_ = self.sinc_convs(xs_pad, ilens)
        new_masks = (~make_pad_mask(ilens.tolist())).to(xs_pad.device).unsqueeze(-2)
        return self.embed(xs_pad), new_masks

