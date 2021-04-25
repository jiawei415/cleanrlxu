#!/usr/bin/env python
# encoding: utf-8

import torch
import numpy as np
from scipy import stats

from plf.base import registry, torch_helper
from plf.policy.algorithm.impl.algorithm_impl import _AlgorithmImpl


@registry.register("torch_dqn_algorithm_with_log")
class _TorchDQNAlgorithmV1(_AlgorithmImpl):
    def __init__(
        self,
        gamma,
        n_step=0,
        double=False,
        distributional_method=None,
        prioritized=False,
        beta=0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.gamma = float(gamma)
        self.n_step = int(n_step)
        self.double = bool(double)
        self.distributional_method = distributional_method
        self.prioritized = bool(prioritized)
        self.beta = float(beta)
        self.ceil = -torch.pow(torch.tensor(10.0), 20.0).to(torch_helper.device)

    def _get_importance_weights(self, probs):
        # importance sampling
        importance_weights = (1.0 / probs.squeeze()) ** self.beta
        importance_weights /= importance_weights.squeeze().max()
        return importance_weights

    def _dqn_loss(self, input_dict):
        p = input_dict["agents"][0]
        action = p["action"].unsqueeze(1)
        qs = p["qs"]
        q = qs.gather(1, action).squeeze()
        dones = input_dict["done"]
        va = p["va"]

        using_model = p["using_model"][0].item()  # HACK: get model from input dict

        with torch.no_grad():
            if not self.n_step:
                next_target_qs = torch.where(va == 1, p["next_target_qs"], self.ceil)
                if self.double:
                    next_qs = torch.where(va == 1, p["next_qs"], self.ceil)
                    next_max_q = p["next_target_qs"].gather(
                        1,
                        next_qs.max(1)[1].unsqueeze(1),
                    )
                else:
                    next_max_q = next_target_qs.max(1)[0]
                next_max_q[dones > 0] = 0
                target_q = next_max_q.squeeze() * self.gamma + p["reward"]
            else:
                n_step_target_qs = torch.where(
                    va == 1, p["n_step_target_qs"], self.ceil
                )
                if self.double:
                    n_step_qs = torch.where(va == 1, p["n_step_qs"], self.ceil)
                    n_step_max_q = n_step_target_qs.gather(
                        1,
                        n_step_qs.max(1)[1].unsqueeze(1),
                    )
                else:
                    n_step_max_q = n_step_target_qs.max(1)[0]
                n_step = p["n_step"].squeeze()
                n_step_max_q[dones > 0] = 0
                n_step_max_q[n_step == 0] = 0
                n_step = n_step.float()
                target_q = (
                    n_step_max_q.squeeze() * (self.gamma ** n_step) + p["n_step_return"]
                )

        loss = (q - target_q).pow(2)

        if self.prioritized:
            ret = {"key_array": input_dict["key_array"].detach().cpu().numpy()}
        else:
            ret = {}
        with torch.no_grad():
            ret["q"] = qs.mean()
        if self.prioritized:
            with torch.no_grad():
                ret["priority"] = (q - target_q).abs().cpu().numpy()
                importance_weights = self._get_importance_weights(input_dict["prob"])
            loss *= importance_weights
        ret["loss"] = loss.mean()
        for action, q_action in enumerate(qs.mean(0)):
            ret.update({f"m{using_model}_action_{action}": q_action})
        most_action = stats.mode(p["action"].to("cpu"))[0][0]
        ret[f"m{using_model}_most_action"] = most_action
        re = p["reward"]
        ret.update(
            {
                f"m{using_model}_total_loss": loss.mean(),
                f"m{using_model}_reward_mean": re.mean(),
                f"m{using_model}_q_mean": q.mean(),
                f"m{using_model}_target_q_mean": target_q.mean(),
                f"m{using_model}_qs_mean": qs.mean(),
            }
        )

        return ret

    def _projection_dist(self, next_dist, rew, done, atoms, n_step=1):
        v_min, v_max, n_atoms = atoms[0], atoms[-1], atoms.shape[0]
        mask = (-(done.to(torch.float32).view(-1, 1) - 1)).abs()
        if isinstance(n_step, int):
            discount = self.gamma ** n_step
        elif isinstance(n_step, torch.Tensor):
            discount = self.gamma ** (n_step.float().squeeze())
            discount = discount.view(-1, 1)
        else:
            assert 0, f"invalid n_step type {type(n_step)}"
        target_atoms = (rew.unsqueeze(-1) + discount * mask * atoms.view(1, -1)).clamp(
            v_min, v_max
        )
        # https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/agent/CategoricalDQN_agent.py#L78
        target_dist = (
            1
            - (target_atoms.unsqueeze(1) - atoms.view(1, -1, 1)).abs()
            / ((v_max - v_min) / (n_atoms - 1))
        ).clamp(0, 1) * next_dist.unsqueeze(1)
        return target_dist.sum(-1)

    def _categorical_dqn_loss(self, input_dict):
        p = input_dict["agents"][0]
        # a batch of atoms (bs, n_atoms)
        atoms = p["atoms"][0]
        batch_indices = np.arange(p["atoms"].shape[0])
        q_dist = p["q_dist"][batch_indices, p["action"], :].squeeze()

        using_model = p["using_model"][0].item()  # HACK: get model from input dict

        with torch.no_grad():
            if not self.n_step:
                if self.double:
                    next_action = p["next_qs"].argmax(1).squeeze()
                else:
                    next_action = p["next_target_qs"].argmax(1).squeeze()
                next_target_q_dist = p["next_target_q_dist"][
                    batch_indices, next_action, :
                ].squeeze()
                target_q_dist = self._projection_dist(
                    next_target_q_dist, p["reward"], input_dict["done"], atoms
                ).squeeze()
            else:
                if self.double:
                    next_action = p["n_step_qs"].argmax(1).squeeze()
                else:
                    next_action = p["n_step_target_qs"].argmax(1).squeeze()
                n_step_target_q_dist = p["n_step_target_q_dist"][
                    batch_indices, next_action, :
                ].squeeze()
                done = input_dict["done"].squeeze()
                n_step = p["n_step"].squeeze()
                n_step_terminal = torch.zeros_like(done)
                n_step_terminal[n_step == 0] = 1
                done += n_step_terminal
                done[done > 0] = 1
                target_q_dist = self._projection_dist(
                    n_step_target_q_dist, p["n_step_return"], done, atoms, n_step
                ).squeeze()

        loss = -(target_q_dist * torch.log(q_dist + 1e-8)).sum(1)

        if self.prioritized:
            ret = {"key_array": input_dict["key_array"].detach().cpu().numpy()}
        else:
            ret = {}
        if self.prioritized:
            with torch.no_grad():
                ret["priority"] = loss.clone().abs().detach().cpu().numpy()
                importance_weights = self._get_importance_weights(input_dict["prob"])
            loss *= importance_weights
        ret["loss"] = loss.mean()
        re = p["reward"]
        ret.update(
            {
                f"m{using_model}_total_loss": loss.mean(),
                f"m{using_model}_reward_mean": re.mean(),
            }
        )

        return ret

    def loss(self, input_dict):
        if not self.distributional_method:
            return self._dqn_loss(input_dict)
        elif self.distributional_method == "categorical":
            return self._categorical_dqn_loss(input_dict)
        else:
            raise RuntimeError(
                f"Unknow distributional_method[{self.distributional_method}]"
            )
