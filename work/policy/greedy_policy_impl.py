#!/usr/bin/env python
# encoding: utf-8

import math
import numpy as np
from plf.base import registry
from plf.base.implementation import expose_to_interface
from plf.policy.impl.policy_impl import _PolicyImpl


@registry.register("greedy_policy")
class _GreedyPolicyImplV2(_PolicyImpl):
    r"""
    Policy implemented for DQN.

    Args:
        update_target_net_gap (int): Steps between updating target net.
        eps_start (float): Parameter for epsilon-greedy.
        eps_end (float): Parameter for epsilon-greedy.
        eps_decay (float): Parameter for epsilon-greedy.
        greedy (bool): Selects actions with maximum probability. Defaults to False.

    .. seealso:: Check out source code of :func:`~_TorchDQNPolicyImpl.sample`
        for how parameters **eps_\*** work.
    """

    def __init__(
        self,
        update_target_net_gap=100,
        eps=None,
        eps_start=None,
        eps_end=None,
        eps_decay=None,
        n_actions=None,
        greedy=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        print(f"eps_start: {eps_start}, eps_end: {eps_end}")
        if not greedy:
            if eps is not None:
                self.eps = float(eps)
            else:
                self.eps_start = float(eps_start)
                self.eps_end = float(eps_end)
                self.eps_decay = float(eps_decay)
            self.n_actions = int(n_actions)
        self.update_target_net_gap = int(update_target_net_gap)
        self.loss_counter = 0
        self.sample_counter = 0
        self.greedy = bool(greedy)

    def sample(self, input_dict):
        o = self.model.infer(input_dict)
        p = o["agents"][0]
        va = p["va"]
        using_model = p["using_model"]
        if using_model == -1:
            p["action"] = self.ppo_sample(va, p["logits"])
        else:
            p["action"] = self.dqn_sample(va, p["qs"])
        return o

    def dqn_sample(self, va, qs):
        self.sample_counter += 1
        counter = self.sample_counter
        if not self.greedy:
            if hasattr(self, "eps"):
                eps = self.eps
            else:
                eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
                    -1 * counter / self.eps_decay
                )
            import random

            if random.random() > eps:
                ceil = math.pow(10.0, 20)
                va_qs = np.where(va, qs, np.full_like(qs, -ceil))
                action = np.argmax(va_qs)
            else:
                va_qs = np.where(qs * va)[0]
                action = va_qs[random.randint(0, len(va_qs) - 1)]
        else:
            action = np.argmax(qs)
        return action

    def ppo_sample(self, va, logits):
        ceil = math.pow(10.0, 20)
        logits = np.where(va, logits, np.full_like(logits, -ceil))
        logits_max = np.amax(logits)
        logits = va * (np.exp(logits - logits_max) + 1e-6)
        sum_logits = np.sum(logits)
        if sum_logits == 0:
            return np.full_like(logits, 1) / len(logits)
        pi = logits / sum_logits
        return np.argmax(pi)

    def loss(self, input_dict):
        r"""Returns loss.

        Args:
            input_dict (dict): A dict of inputs.

        Returns:
            Output from :func:`~plf.policy.algorithm.algorithm.Algorithm.loss`\.
        """
        self.loss_counter += 1
        if self.loss_counter % self.update_target_net_gap == 0:
            input_dict["update_target_net"] = np.array(1)
        input_dict["train"] = np.array(1)

        out = self.model.forward(input_dict)
        out = self.alg.loss(out)

        return out

    @expose_to_interface
    def set_greedy(self, greedy):
        r"""
        set_greedy(greedy)
        Sets attr :attr:`~greedy`\.

        Args:
            greedy (bool): Selects actions with maximum probability.
        """
        self.greedy = greedy
