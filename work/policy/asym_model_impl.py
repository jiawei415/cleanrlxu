from plf.base import registry, torch_helper
from alpha.base import state
from plf.policy.model.impl.torch.torch_model_impl import _TorchModelImpl
import os
import sys
import glog

sys.path.append(os.path.dirname(__file__))


flag = False


def print_total_parameters(net):
    for params in net.named_parameters():
        print(
            params[0],
            ": shape={} params={}".format(tuple(params[1].shape), params[1].numel()),
        )
    total_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("total_parameters:" + str(total_parameters))


@registry.register("torch_asym_model")
class _TorchAsymModelImplV2(_TorchModelImpl):
    def __init__(
        self, label_num, pretrain_model_path=None, embedding_size=256, **kwargs
    ):
        self.embedding_size = embedding_size
        self._label_num = label_num
        self.pretrain_model_path = None
        self.model_key = None
        if state.has("learner_init_model_key"):
            # for learner load model
            self.model_key = state.learner_init_model_key()
        if pretrain_model_path is not None and pretrain_model_path != "None":
            self.pretrain_model_path = pretrain_model_path
        super().__init__(**kwargs)

    def _build_net(self):
        import torch
        from torch_model import Net

        net = Net(units=self.embedding_size)
        if self.pretrain_model_path is not None and self.model_key is not None:
            model_path = os.path.join(
                self.pretrain_model_path, self.model_key, "model.pt"
            )
            net.load_state_dict(
                torch.load(model_path, map_location=torch_helper.device)
            )
            glog.info(f"{self.model_key}: load pretrain {self.pretrain_model_path}")
        return net

    def forward(self, input_dict):
        """forward function for batch select method"""

        global flag
        agent_num = len(input_dict["agents"])

        for i in range(agent_num):
            p = input_dict["agents"][i]
            entities = p["obs"]["entities"]
            localmap = p["obs"]["localmap"]
            globalmap = p["obs"]["globalmap"]
            out = self._net(entities, localmap, globalmap)
            p["logits"] = out[:, 0 : self._label_num]  # noqa E203
            p["value"] = out[:, self._label_num]

        if not flag:
            # print_total_parameters(self._net)
            flag = True

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
