from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from fast_attention import Attention as FAttention

MAPDEPTH = 17
ENTITY_FEATURE = 60  # same with env.freeze_wrapper


def dense_with_layernorm(in_dim, out_dim):
    return nn.Sequential(
        OrderedDict(
            [
                ("dense", nn.Linear(in_dim, out_dim, True)),
                ("activation", nn.ReLU()),
                ("normalize", nn.LayerNorm(out_dim)),
            ]
        )
    )


def image_trunk():
    return nn.Sequential(
        OrderedDict(
            [
                ("conv1", nn.Conv2d(MAPDEPTH, 32, 5, stride=2)),
                ("activation1", nn.ReLU()),
                ("conv2", nn.Conv2d(32, 32, 5, stride=2)),
                ("activation2", nn.ReLU()),
            ]
        )
    )


def decode_map(localmap, globalmap):
    # get self vector
    onehot_local = F.one_hot(torch.reshape(localmap, (-1, 21, 21)).long(), MAPDEPTH)
    onehot_global = F.one_hot(torch.reshape(globalmap, (-1, 23, 35)).long(), MAPDEPTH)

    return onehot_local, onehot_global


class Net(nn.Module):
    def __init__(self, units):
        super(Net, self).__init__()
        self.units = units
        self.label_num = 18
        self.entities_emb = dense_with_layernorm(
            in_dim=ENTITY_FEATURE, out_dim=self.units
        )
        self.entities_attention = FAttention(hidden_size=units, num_heads=4)
        self.localmap1 = image_trunk()
        self.localmap2 = dense_with_layernorm(in_dim=288, out_dim=self.units)
        self.globalmap1 = image_trunk()
        self.globalmap2 = dense_with_layernorm(in_dim=576, out_dim=self.units)
        self.preout = dense_with_layernorm(
            in_dim=self.units * 3, out_dim=self.units * 2
        )
        self.out = nn.Linear(self.units * 2, self.label_num + 1, True)

    def forward(self, entities, localmap, globalmap):
        onehot_local, onehot_global = decode_map(localmap, globalmap)
        entities = torch.reshape(entities, (-1, 60, 60))
        entities_emb = self.entities_emb(entities)
        entities_attention = self.entities_attention(
            entities_emb, entities_emb, entities_emb
        ).permute(0, 2, 1)
        N_ENTITIES = 60
        pooled_att = nn.MaxPool1d(N_ENTITIES)(entities_attention).squeeze(dim=2)
        local_feature1 = self.localmap1(onehot_local.permute(0, 3, 1, 2).float())
        local_feature2 = self.localmap2(local_feature1.flatten(1))
        global_feature1 = self.globalmap1(onehot_global.permute(0, 3, 1, 2).float())
        global_feature2 = self.globalmap2(global_feature1.flatten(1))
        common = torch.cat([local_feature2, global_feature2, pooled_att], dim=1)
        preout = self.preout(common)
        out = self.out(preout)
        return out
