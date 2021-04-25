from plf.base import registry
from plf.policy.algorithm.impl.algorithm_impl import _AlgorithmImpl
# from plf.policy_tf.algorithm.impl.tf1.tf1_ppo_algorithm_impl import _TF1PPOImpl
from plf.policy.algorithm.impl.torch.torch_ppo_algorithm_impl import _TorchPPOAlgorithm

@registry.register("torch_ppo_algorithm_with_log")
class _TorchPPOAlgorithm(_AlgorithmImpl):
    r"""selfplay ppo for Asym

    Args:
        clip_ratio (float): PPO clip ratio for policy_tf loss and value difference.
        value_coef (float): value loss coefficient.
        entropy_coef (float): entropy loss coefficient.
        do_value_clip (bool): need value clipping or not.
        do_adv_norm (bool): need to normalize advantages or not.

    Attributes:
        clip_ratio (float): PPO clip ratio for policy_tf loss and value difference.
        value_coef (float): value loss coefficient.
        entropy_coef (float): entropy loss coefficient.
        do_value_clip (bool): need value clipping or not.
        do_adv_norm (bool): need to normalize advantages or not.
    """
    def __init__(
        self,
        clip_ratio,
        value_coef=1.0,
        entropy_coef=0.0,
        do_value_clip=False,
        do_adv_norm=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.clip_ratio = float(clip_ratio)
        self.do_value_clip = do_value_clip
        self.do_adv_norm = do_adv_norm

    def loss(self, input_dict):
        import torch
        s_loss = None
        o = {}
        for i, p in enumerate(input_dict["agents"]):
            ret = p["return"]
            value = p["value"]
            action = p["action"]
            logits = p["logits"]
            logp_old = p["logp_old"]
            value_old = p["value_old"]
            is_train = p["is_train"]
            va = p["va"] if "va" in p else torch.ones_like(logits,
                                                           dtype=torch.float32)
            using_model = p["using_model"][0].item() # HACK: get model from input dict

            adv = ret - value_old
            if self.do_adv_norm:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # softmax, masked invalid action logits
            mask_val = (1 - va) * torch.pow(torch.tensor(10.0), 20.0)
            logits_sub_max = torch.clamp(
                logits - torch.max(logits - mask_val, 1, True)[0],
                -torch.pow(torch.tensor(10.0), 20.0), 1)

            sm_numerator = va * torch.exp(logits_sub_max) + 0.00001
            sm_denominator = torch.sum(sm_numerator, 1, keepdim=True)
            pi = 1.0 * sm_numerator / sm_denominator

            # surrogate loss
            logp = torch.log(pi.gather(1,
                                       action.unsqueeze(1).long())).squeeze(1)

            ratio = torch.exp(logp - logp_old)
            clip_adv = torch.clamp(ratio, 1 - self.clip_ratio,
                                   1 + self.clip_ratio) * adv
            policy_loss = -(torch.min(ratio * adv, clip_adv) * is_train).mean()

            # value loss
            value_diff = value - value_old
            clip_value = value_old + torch.clamp(value_diff, -self.clip_ratio,
                                                 self.clip_ratio)
            if self.do_value_clip:
                value_loss = 0.5 * ((torch.max((value - ret)**2,
                                               (clip_value - ret)**2)).mean())
            else:
                value_loss = 0.5 * ((value - ret)**2).mean()

            # crit = torch.nn.SmoothL1Loss(reduction='sum')
            # value_loss = crit(input=value, target=ret)

            # entropy loss
            entropy = -torch.sum(pi * va * torch.log(pi), dim=1)
            entropy_loss = -torch.mean(entropy * is_train, dim=0)

            # loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            # s_loss += loss
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            if s_loss is None:
                s_loss = torch.zeros_like(loss)
            s_loss += loss

            # additional info
            approx_kl = 0.5 * (((logp_old - logp)**2).mean())
            policy_clip_frac = torch.as_tensor(ratio.gt(1 + self.clip_ratio)
                                               | ratio.lt(1 - self.clip_ratio),
                                               dtype=torch.float32).mean()
            value_clip_frac = torch.as_tensor(
                value_diff.gt(self.clip_ratio)
                | value_diff.lt(-self.clip_ratio),
                dtype=torch.float32).mean()
            re = p['reward']

            o.update({
                f"m{using_model}_total_loss": loss,
                f"m{using_model}_value_loss": value_loss,
                f"m{using_model}_policy_loss": policy_loss,
                f"m{using_model}_value_mean": value.mean(),
                f"m{using_model}_return_mean": ret.mean(),
                f"m{using_model}_entropy_loss": entropy_loss,
                f"m{using_model}_approx_kl": approx_kl,
                f"m{using_model}_value_diff": value_diff.mean(),
                f"m{using_model}_policy_clip_frac": policy_clip_frac,
                f"m{using_model}_value_clip_frac": value_clip_frac,
                f"m{using_model}_advantage": adv.mean(),
                f"m{using_model}_reward_mean": re.mean(),
                f"m{using_model}_value_min": torch.min(value),
                f"m{using_model}_value_max": torch.max(value),
            })

        o["loss"] = s_loss
        return o

