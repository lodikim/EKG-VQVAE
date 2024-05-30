import torch
from torch import nn
from torch.nn import functional as F

import distributed as dist_fn


# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        input = input.permute(0, 2, 1)
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

    def extract_codebook(self):
        return self.embed

'''
class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out
'''

class Encoder(nn.Module):
    def __init__(self, in_channel, channel, out_channel, stride):
        super().__init__()

        if stride == 8:
            blocks = [
                nn.Conv1d(in_channel, channel, 4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope = 0.2),
                nn.Conv1d(channel, channel, 4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope = 0.2),
                nn.Conv1d(channel, channel, 4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope = 0.2),
                #nn.Conv1d(channel, out_channel, 3, padding=1),
            ]

        elif stride == 4:
            blocks = [
                nn.Conv1d(in_channel, channel, 4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope = 0.2),
                nn.Conv1d(channel, channel, 4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope = 0.2),
                #nn.Conv1d(channel, out_channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv1d(in_channel, channel, 4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope = 0.2),
                #nn.Conv1d(channel, out_channel, 3, padding=1),
            ]

        blocks.append(nn.Conv1d(channel, out_channel, 3, padding=1))

        #blocks.append(nn.LeakyReLU(negative_slope = 0.2))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        output = self.blocks(input)
        return output


class Decoder(nn.Module):
    def __init__(
        self, in_channel, channel, out_channel, stride
    ):
        super().__init__()

        blocks = []
        blocks.append(nn.Conv1d(out_channel, channel, 3, padding=1))

        if stride == 8:
            blocks.extend(
                [
                    nn.LeakyReLU(negative_slope = 0.2),
                    nn.ConvTranspose1d(channel, channel, 4, stride=2, padding=1),
                    nn.LeakyReLU(negative_slope = 0.2),
                    nn.ConvTranspose1d(channel, channel, 4, stride=2, padding=1),
                    nn.LeakyReLU(negative_slope = 0.2),
                    nn.ConvTranspose1d(channel, in_channel, 4, stride=2, padding=1),
                ]
            )

        elif stride == 4:
            blocks.extend(
                [
                    nn.LeakyReLU(negative_slope = 0.2),
                    nn.ConvTranspose1d(channel, channel, 4, stride=2, padding=1),
                    nn.LeakyReLU(negative_slope = 0.2),
                    nn.ConvTranspose1d(channel, in_channel, 4, stride=2, padding=1),
                ]
            )

        elif stride == 2:
            blocks.extend(
                [
                    nn.LeakyReLU(negative_slope = 0.2),
                    nn.ConvTranspose1d(channel, in_channel, 4, stride=2, padding=1)
                ]
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        output = self.blocks(input)
        return output


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=1,
        channel=32,
        out_channel=64,
        embed_dim=64,
        n_embed=2048,
    ):
        super().__init__()

        self.enc = Encoder(in_channel, channel, out_channel, stride=8)
        self.quantize = Quantize(embed_dim, n_embed)
        self.dec = Decoder(in_channel, channel, out_channel, stride=8)

    def forward(self, input):
        input = input.permute(0, 2, 1)
        quant, diff, _ = self.encode(input)
        quant = quant.permute(0, 2, 1)
        dec = self.decode(quant)
        dec = dec.permute(0, 2, 1)

        return dec, diff

    def encode(self, input):
        enc = self.enc(input)
        quant, diff, idx = self.quantize(enc)
        return quant, diff, idx

    def decode(self, quant):
        dec = self.dec(quant)    
        return dec

    def decode_code(self, code):
        quant = self.quantize.embed_code(code)
        dec = self.decode(quant)
        return dec
