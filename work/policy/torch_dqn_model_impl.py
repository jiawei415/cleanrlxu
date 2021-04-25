#!/usr/bin/env python
# encoding: utf-8

from alpha.base import state
from plf.base import registry, torch_helper
from plf.policy.model.impl.torch.torch_model_impl import _TorchModelImpl
from collections import OrderedDict
from fast_attention import Attention as FAttention

import os
import sys
import glog
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F

MAPDEPTH = 17
ENTITY_FEATURE = 60  # same with env.freeze_wrapper
MISC_DIR = str(pathlib.Path(__file__).parent.parent.absolute().joinpath("env/misc/"))
sys.path.append(MISC_DIR)
sys.path.append(os.path.dirname(__file__))


def print_total_parameters(net):
    for params in net.named_parameters():
        print(
            params[0],
            ": shape={} params={}".format(tuple(params[1].shape), params[1].numel()),
        )
    total_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("total_parameters: " + str(total_parameters))


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
    def __init__(self, units=128, label_num=18, dueling=False):
        super(Net, self).__init__()
        self.units = units
        self.label_num = label_num
        self.dueling = dueling
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
        if self.dueling:
            self.value = nn.Linear(self.units * 2, 1, True)
            self.adv = nn.Linear(self.units * 2, self.label_num, True)
        else:
            self.out = nn.Linear(self.units * 2, self.label_num, True)

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
        if self.dueling:
            value = self.value(preout)
            advs = self.adv(preout)
            out = value + (advs - advs.mean(1).unsqueeze(1))
        else:
            out = self.out(preout)
        return out


@registry.register("torch_dqn_asym_model")
class _TorchAsymModelImplV3(_TorchModelImpl):
    def __init__(
        self,
        pretrain_model_path=None,
        embedding_size=256,
        label_num=18,
        n_step=0,
        double=False,
        dueling=False,
        **kwargs,
    ):
        self.embedding_size = embedding_size
        self._label_num = label_num
        self.pretrain_model_path = None
        self.model_key = None
        if state.has("learner_init_model_key"):
            self.model_key = state.learner_init_model_key()
        if pretrain_model_path is not None and pretrain_model_path != "None":
            self.pretrain_model_path = pretrain_model_path
        self.n_step = int(n_step)
        self.double = bool(double)
        self.dueling = bool(dueling)
        super().__init__(**kwargs)
        self.target_net = self._build_net().to(torch_helper.device)
        self.policy_net = self._net
        self._update_target_net()

        self.flag = False

    def _build_net(self):
        net = Net(
            units=self.embedding_size, label_num=self._label_num, dueling=self.dueling
        )
        if self.pretrain_model_path is not None and self.model_key is not None:
            model_path = os.path.join(
                self.pretrain_model_path, self.model_key, "model.pt"
            )
            net.load_state_dict(
                torch.load(model_path, map_location=torch_helper.device)
            )
            glog.info(f"{self.model_key}: load pretrain {self.pretrain_model_path}")
        return net

    def _update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        glog.info("update target model")

    def forward(self, input_dict):
        """forward function for batch select method"""

        if "update_target_net" in input_dict and input_dict["update_target_net"]:
            self._update_target_net()

        agent_num = len(input_dict["agents"])
        for i in range(agent_num):
            p = input_dict["agents"][i]
            entities = p["obs"]["entities"]
            localmap = p["obs"]["localmap"]
            globalmap = p["obs"]["globalmap"]
            p["qs"] = self.policy_net(entities, localmap, globalmap)

            if "train" in input_dict and input_dict["train"]:
                if not self.n_step:
                    next_entities = p["next_obs"]["entities"]
                    next_localmap = p["next_obs"]["localmap"]
                    next_globalmap = p["next_obs"]["globalmap"]
                    with torch.no_grad():
                        p["next_target_qs"] = self.target_net(
                            next_entities, next_localmap, next_globalmap
                        )
                        if self.double:
                            p["next_qs"] = self.policy_net(
                                next_entities, next_localmap, next_globalmap
                            )
                else:
                    n_step_entities = p["n_step_obs"]["entities"]
                    n_step_localmap = p["n_step_obs"]["localmap"]
                    n_step_globalmap = p["n_step_obs"]["globalmap"]
                    with torch.no_grad():
                        p["n_step_target_qs"] = self.target_net(
                            n_step_entities, n_step_localmap, n_step_globalmap
                        )
                        if self.double:
                            p["n_step_qs"] = self.policy_net(
                                n_step_entities, n_step_localmap, n_step_globalmap
                            )

        if not self.flag:
            print_total_parameters(self._net)
            self.flag = True
        return input_dict

    def save(self, model_path):
        r"""
        Saves model. Overrides. Save .pt and .onnx.

        Args:
            model_path (str): Path to save model.
        """
        os.system("mkdir -p {}".format(model_path))
        import torch

        torch.save(self._net.state_dict(), "{}/model.pt".format(model_path))

        def convert_onnx(model, onnx_name):
            import copy

            dummy_globalmap = torch.randint(
                low=0, high=17, size=(1, 805), dtype=torch.int32, device="cpu"
            ).float()
            dummy_localmap = torch.randint(
                low=0, high=17, size=(1, 441), dtype=torch.int32, device="cpu"
            ).float()
            dummy_entities = torch.randn(1, 60 * 60, device="cpu")

            input_names = ["entities", "localmap", "globalmap"]
            output_names = ["output"]
            _tmp_model = copy.deepcopy(model).to("cpu")
            torch.onnx.export(
                _tmp_model,
                (dummy_entities, dummy_localmap, dummy_globalmap),
                onnx_name,
                verbose=False,
                input_names=input_names,
                output_names=output_names,
                opset_version=12,
            )

        convert_onnx(self._net, "{}/model.onnx".format(model_path))
