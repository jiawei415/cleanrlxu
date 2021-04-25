import math
import numpy as np
from plf.base import registry
from plf.policy.impl.policy_impl import _PolicyImpl


def _gen_pi(va, logits, min_prob):
    ceil = math.pow(10.0, 20)

    logits = np.where(va, logits, np.full_like(logits, -ceil))
    logits_max = np.amax(logits)
    logits = va * (np.exp(logits - logits_max) + min_prob)
    sum_logits = np.sum(logits)
    if sum_logits == 0:
        return np.full_like(logits, 1) / len(logits)
    pi = logits / sum_logits
    return pi


def custom_softmax(o):
    for a in o["agents"]:
        logits = a["logits"]
        va = a["va"]
        pi = _gen_pi(va, logits, 1e-6)
        print(f"logits: {logits.shape}, pi: {pi.shape}")
        if int(a["using_model"]) == -1:
            # for fixed model, using model is -1
            action = np.argmax(pi)
        else:
            action = np.random.choice(np.arange(len(pi)), p=pi)
        a["pi"] = pi
        a["action"] = action
        a["logp"] = np.log(pi[action], dtype=np.float32)

    return o


@registry.register("softmax_policy_v2")
class _SoftmaxPolicyImplV2(_PolicyImpl):
    r"""
    Policy selects actions based on softmax probability.
    """

    def __init__(self, mapping=None, **kwargs):
        super().__init__(**kwargs)

        self.mapping = mapping

        self._softmax = custom_softmax

    def sample(self, input_dict):
        r"""
        Samples actions from policy with valid actions.

        Output of :class:`~plf.policy.model.model.Model.forward` should
        have the following structure::

            output = {
                "agents":
                [
                    {
                        "logits": tensorflow.Operation,
                    },
                    ...,
                ]
            }

        input_dict should have following structure::

            input_dict = {
                "agents":
                [
                    {
                        "obs": any type as input
                        "va": numpy.Array (optional, if not given, valid action will be all actions avaliable)
                    }
                ]
            }

        Args:
            input_dict (dict): A dict of inputs.

        Returns:
            A dict with structure::

                output = {
                    "agents":
                    [
                        {
                            "pi": numpy.Array,
                            "action": int,
                            "logp": float,
                            "value": float, (if exist)
                        },
                        ...,
                    ]
                }
        """
        o = self.infer(input_dict)
        o = self._softmax(o)

        return o

    def batch_sample(self, input_dict_list):
        output_dict_list = self.batch_infer(input_dict_list)
        return [self._softmax(output_dict) for output_dict in output_dict_list]
