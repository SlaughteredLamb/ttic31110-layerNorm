from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import torch.nn as nn

class LayerNormGRUCell(nn.GRUCell):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LayerNormGRUCell, self).__init__(input_size, hidden_size, bias)

        self.gamma_ih = nn.Parameter(torch.ones(3 * self.hidden_size))
        self.gamma_hh = nn.Parameter(torch.ones(3 * self.hidden_size))
        self.eps = 0

    def _layer_norm_x(self, x, g, b):
        mean = x.mean(1).expand_as(x)
        std = x.std(1).expand_as(x)
        return g.expand_as(x) * ((x - mean) / (std + self.eps)) + b.expand_as(x)

    def _layer_norm_h(self, x, g, b):
        mean = x.mean(1).expand_as(x)
        return g.expand_as(x) * (x - mean) + b.expand_as(x)

    def forward(self, x, h):
        for j in range(x.size(0)):

            #print('inGRUcell',x.size(),x[j].size(),self.weight_ih.narrow(0, 0, 2 * self.hidden_size).transpose(0, 1).size())
            ih_rz = self._layer_norm_x(
                torch.mm(x[j].clone().unsqueeze(0), self.weight_ih.narrow(0, 0, 2 * self.hidden_size).transpose(0, 1)),
                self.gamma_ih.narrow(0, 0, 2 * self.hidden_size),
                self.bias_ih.narrow(0, 0, 2 * self.hidden_size))

            #print(hb.size(), self.weight_hh.narrow(0, 0, 2 * self.hidden_size).transpose(0, 1).size())
            hh_rz = self._layer_norm_h(
                torch.mm(h[j].clone().unsqueeze(0), self.weight_hh.narrow(0, 0, 2 * self.hidden_size).transpose(0, 1)),
                self.gamma_hh.narrow(0, 0, 2 * self.hidden_size),
                self.bias_hh.narrow(0, 0, 2 * self.hidden_size))

            rz = torch.sigmoid(ih_rz + hh_rz)
            r = rz.narrow(1, 0, self.hidden_size)
            z = rz.narrow(1, self.hidden_size, self.hidden_size)

            ih_n = self._layer_norm_x(
                torch.mm(x[j].clone().unsqueeze(0), self.weight_ih.narrow(0, 2 * self.hidden_size, self.hidden_size).transpose(0, 1)),
                self.gamma_ih.narrow(0, 2 * self.hidden_size, self.hidden_size),
                self.bias_ih.narrow(0, 2 * self.hidden_size, self.hidden_size))

            hh_n = self._layer_norm_h(
                torch.mm(h[j].clone().unsqueeze(0), self.weight_hh.narrow(0, 2 * self.hidden_size, self.hidden_size).transpose(0, 1)),
                self.gamma_hh.narrow(0, 2 * self.hidden_size, self.hidden_size),
                self.bias_hh.narrow(0, 2 * self.hidden_size, self.hidden_size))

            n = torch.tanh(ih_n + r * hh_n)
            h[j] = (1 - z) * n + z * h[j].clone()
        return h

class LayerNormGRU(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LayerNormGRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size


    def forward(self, xs, h = None):
        if type(h) == type(None):
            h = torch.zeros(xs.size(0),self.hidden_size, dtype=xs.dtype, device=xs.device)
            self.cell = LayerNormGRUCell(self.input_size, self.hidden_size)
            self.weight_ih_l0 = self.cell.weight_ih
            self.weight_hh_l0 = self.cell.weight_hh
            self.bias_ih_l0 = self.cell.bias_ih
            self.bias_hh_l0 = self.cell.bias_hh
        else:
            self.cell = LayerNormGRUCell(self.hidden_size, self.hidden_size)
        y = torch.zeros(xs.size(0),xs.size(1),h.size(1))
        for j in range(xs.size(0)):
            h[j]=h[j].clone().squeeze(0)
            for i in range(xs[j].size(0)):
                x = xs[j].clone().narrow(0, i, 1).squeeze(0)
                h[j] = self.cell(x.unsqueeze(0), h[j].clone().unsqueeze(0))
                y[j][i] = h[j].clone()
        return y, h