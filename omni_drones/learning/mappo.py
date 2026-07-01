# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from functorch import vmap
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.utils import expand_right
try:
    from tensordict.nn import make_functional, TensorDictModule, TensorDictParams
except ImportError:  # tensordict 0.1.x has no TensorDictParams wrapper.
    from tensordict.nn import make_functional, TensorDictModule

    def TensorDictParams(params):
        return _ensure_tensordict_leaf_parameters(params)
from torch.optim import lr_scheduler

from omni_drones.utils.torchrl.specs import (
    BoundedTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    MultiDiscreteTensorSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec as UnboundedTensorSpec,
)

from omni_drones.utils.torchrl.env import AgentSpec

from .utils import valuenorm
from .utils.gae import compute_gae

LR_SCHEDULER = lr_scheduler._LRScheduler
from torchrl.modules import TanhNormal, IndependentNormal


def _ensure_tensordict_leaf_parameters(params):
    params_td = params.to_tensordict() if hasattr(params, "to_tensordict") else params
    for key in list(params_td.keys(include_nested=True, leaves_only=True)):
        value = params_td.get(key)
        if (
            isinstance(value, torch.Tensor)
            and value.requires_grad
            and (not isinstance(value, nn.Parameter) or not value.is_leaf)
        ):
            params_td.set(key, nn.Parameter(value.detach().clone()))
    return params_td


def _iter_tensordict_parameters(params):
    if hasattr(params, "parameters"):
        return params.parameters()
    params_td = params.to_tensordict() if hasattr(params, "to_tensordict") else params
    out = []
    for key in list(params_td.keys(include_nested=True, leaves_only=True)):
        value = params_td.get(key)
        if not isinstance(value, torch.Tensor) or not value.requires_grad:
            continue
        if not isinstance(value, nn.Parameter) or not value.is_leaf:
            value = nn.Parameter(value.detach().clone())
            params_td.set(key, value)
        out.append(value)
    return out


class MAPPOPolicy(object):
    def __init__(
        self,
        cfg,
        agent_spec: AgentSpec,
        device="cuda",
        predictor_net=None,
        predictor_stride: int = 1,
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.agent_spec = agent_spec
        self.device = device

        print(self.agent_spec.observation_spec)
        print(self.agent_spec.action_spec)

        self.clip_param = cfg.clip_param
        self.ppo_epoch = int(cfg.ppo_epochs)
        self.num_minibatches = int(cfg.num_minibatches)
        self.normalize_advantages = cfg.normalize_advantages

        self.entropy_coef = cfg.entropy_coef
        self.gae_gamma = cfg.gamma
        self.gae_lambda = cfg.gae_lambda

        self.act_dim = agent_spec.action_spec.shape[-1]

        if cfg.reward_weights is not None:
            self.reward_weights = torch.as_tensor(cfg.reward_weights, device=device).float()
        else:
            self.reward_weights = torch.ones(
                self.agent_spec.reward_spec.shape, device=device
            )

        self.obs_name = ("agents", "observation")
        self.act_name = ("agents", "action")
        self.prev_act_name = ("agents", "prev_action")
        self.raw_act_name = ("agents", "action_raw")
        self.reward_name = ("agents", "reward")
        self._cached_prev_action = None

        self.make_actor()
        self.make_critic()
        predictor_cfg = self.cfg.get("predictor", None)
        self.predictor_net = predictor_net
        self.use_predictor_net = bool(
            predictor_cfg is not None
            and predictor_cfg.get("enabled", False)
            and predictor_net is not None
        )
        self.predictor_optimizer = None
        self.predictor_loss_fn = None
        self.predictor_epochs = 1
        self.predictor_num_minibatches = 1
        self.predictor_future_stride = max(1, int(predictor_stride))
        self.predictor_horizon = int(getattr(predictor_net, "T", 10)) if predictor_net is not None else 10
        if self.use_predictor_net:
            from omni_drones.learning.predictor.dual_intent_gmm_predictor import StrategicGMMLoss

            self.predictor_optimizer = torch.optim.Adam(
                self.predictor_net.parameters(),
                lr=float(predictor_cfg.get("lr", 3e-4)),
            )
            self.predictor_epochs = int(predictor_cfg.get("epochs", 1))
            self.predictor_num_minibatches = int(
                predictor_cfg.get("num_minibatches", self.num_minibatches)
            )
            self.predictor_loss_fn = StrategicGMMLoss(
                lambda_strategic=float(predictor_cfg.get("lambda_strategic", 1.0)),
                lambda_gmm=float(predictor_cfg.get("lambda_gmm", 1.0)),
                lambda_speed=float(predictor_cfg.get("lambda_speed", 0.02)),
                lambda_accel=float(predictor_cfg.get("lambda_accel", 0.02)),
                lambda_height=float(predictor_cfg.get("lambda_height", 0.0)),
                dt_pred=float(predictor_cfg.get("dt_pred", 0.05)),
                max_speed=float(predictor_cfg.get("max_speed", 3.0)),
                max_accel=float(predictor_cfg.get("max_accel", 3.0)),
            )

        self.train_in_keys = list(
            set(
                self.actor_in_keys
                + self.actor_out_keys
                + self.critic_in_keys
                + self.critic_out_keys
                + [
                    "next",
                    self.act_logps_name,
                    self.raw_act_name,
                    ("reward", self.reward_name),
                    "state_value",
                ]
                + ["progress", ("collector", "traj_ids")]
            )
        )

        self.n_updates = 0
        self.actor_frozen = False
        self.ppo_target_kl: Optional[float] = None
        self.ppo_kl_coef: float = 0.0
        self.ppo_kl_hard_stop: bool = False
        self.anchor_actor_params: Optional[TensorDictParams] = None
        self.anchor_actor_state: Optional[Dict[str, torch.Tensor]] = None
        self.anchor_loss_coef: float = 0.0
        self.anchor_loss_type: str = "mse"

    @property
    def act_logps_name(self):
        return f"{self.agent_spec.name}.action_logp"

    def make_actor(self):
        cfg = self.cfg.actor

        self.actor_in_keys = [self.obs_name, self.act_name]
        prev_action_cfg = cfg.get("prev_action_conditioning", None)
        self.actor_has_prev_action_conditioning = bool(
            prev_action_cfg is not None and prev_action_cfg.get("enabled", False)
        )
        if self.actor_has_prev_action_conditioning:
            self.actor_in_keys = [self.obs_name, self.prev_act_name, self.act_name]
        self.actor_out_keys = [
            self.act_name,
            self.act_logps_name,
            f"{self.agent_spec.name}.action_entropy",
        ]

        if cfg.get("rnn", None):
            self.actor_in_keys.extend(
                [f"{self.agent_spec.name}.actor_rnn_state", "is_init"]
            )
            self.actor_out_keys.append(f"{self.agent_spec.name}.actor_rnn_state")
            self.minibatch_seq_len = self.cfg.actor.rnn.train_seq_len
            assert self.minibatch_seq_len <= self.cfg.train_every

        create_actor_fn = lambda: TensorDictModule(
            make_ppo_actor(
                cfg, self.agent_spec.observation_spec, self.agent_spec.action_spec
            ),
            in_keys=self.actor_in_keys,
            out_keys=self.actor_out_keys
        ).to(self.device)

        if self.cfg.share_actor:
            self.actor = create_actor_fn()
            # tensordict 0.1.x functional calls do not propagate gradients back
            # to the external params reliably. For the shared-actor path used by
            # this project, train the TensorDictModule directly.
            self.actor_params = None
        else:
            actors = nn.ModuleList([create_actor_fn() for _ in range(self.agent_spec.n)])
            self.actor = actors[0]
            stacked_params = torch.stack([make_functional(actor) for actor in actors])
            self.actor_params = TensorDictParams(stacked_params.to_tensordict())
        
        actor_parameters = (
            self.actor.parameters()
            if self.actor_params is None
            else _iter_tensordict_parameters(self.actor_params)
        )
        self.actor_opt = torch.optim.Adam(actor_parameters, lr=cfg.lr)
        actor_scheduler = cfg.get("lr_scheduler", None) if hasattr(cfg, "get") else getattr(cfg, "lr_scheduler", None)
        if actor_scheduler is not None:
            actor_scheduler = eval(actor_scheduler)
            actor_scheduler_kwargs = cfg.get("lr_scheduler_kwargs", {}) if hasattr(cfg, "get") else getattr(cfg, "lr_scheduler_kwargs", {})
            if actor_scheduler_kwargs is None:
                actor_scheduler_kwargs = {}
            self.actor_opt_scheduler: LR_SCHEDULER = actor_scheduler(
                self.actor_opt, **actor_scheduler_kwargs
            )

    def _reset_actor_optimizer(self):
        cfg = self.cfg.actor
        actor_parameters = (
            self.actor.parameters()
            if self.actor_params is None
            else _iter_tensordict_parameters(self.actor_params)
        )
        self.actor_opt = torch.optim.Adam(actor_parameters, lr=cfg.lr)
        actor_scheduler = cfg.get("lr_scheduler", None) if hasattr(cfg, "get") else getattr(cfg, "lr_scheduler", None)
        if actor_scheduler is None:
            if hasattr(self, "actor_opt_scheduler"):
                delattr(self, "actor_opt_scheduler")
            return
        actor_scheduler = eval(actor_scheduler)
        actor_scheduler_kwargs = cfg.get("lr_scheduler_kwargs", {}) if hasattr(cfg, "get") else getattr(cfg, "lr_scheduler_kwargs", {})
        if actor_scheduler_kwargs is None:
            actor_scheduler_kwargs = {}
        self.actor_opt_scheduler = actor_scheduler(self.actor_opt, **actor_scheduler_kwargs)

    def make_critic(self):
        cfg = self.cfg.critic

        if cfg.use_huber_loss:
            self.critic_loss_fn = nn.HuberLoss(delta=cfg.huber_delta)
        else:
            self.critic_loss_fn = nn.MSELoss()

        assert self.cfg.critic_input in ("state", "obs")
        if self.cfg.critic_input == "state" and self.agent_spec.state_spec is not None:
            self.critic_in_keys = [("agents", "state")]
            self.critic_out_keys = ["state_value"]
            if cfg.get("rnn", None):
                self.critic_in_keys.extend([
                    f"{self.agent_spec.name}.critic_rnn_state", "is_init"
                ])
                self.critic_out_keys.append(f"{self.agent_spec.name}.critic_rnn_state")
            reward_spec = self.agent_spec.reward_spec
            reward_spec = reward_spec.expand(self.agent_spec.n, *reward_spec.shape)
            critic = make_critic(cfg, self.agent_spec.state_spec, reward_spec, centralized=True)
            self.critic = TensorDictModule(
                critic,
                in_keys=self.critic_in_keys,
                out_keys=self.critic_out_keys,
            ).to(self.device)
            self.value_func = self.critic
        else:
            self.critic_in_keys = [self.obs_name]
            self.critic_out_keys = ["state_value"]
            if cfg.get("rnn", None):
                self.critic_in_keys.extend([
                    f"{self.agent_spec.name}.critic_rnn_state", "is_init"
                ])
                self.critic_out_keys.append(f"{self.agent_spec.name}.critic_rnn_state")
            critic = make_critic(cfg, self.agent_spec.observation_spec, self.agent_spec.reward_spec, centralized=False)
            self.critic = TensorDictModule(
                critic,
                in_keys=self.critic_in_keys,
                out_keys=self.critic_out_keys,
            ).to(self.device)
            self.value_func = vmap(self.critic, in_dims=1, out_dims=1)

        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = cfg.lr_scheduler
        if scheduler is not None:
            scheduler = eval(scheduler)
            self.critic_opt_scheduler: LR_SCHEDULER = scheduler(
                self.critic_opt, **cfg.lr_scheduler_kwargs
            )

        if hasattr(cfg, "value_norm") and cfg.value_norm is not None:
            # The original MAPPO implementation uses ValueNorm1 with a very large beta,
            # and normalizes advantages at batch level.
            # Tianshou (https://github.com/thu-ml/tianshou) uses ValueNorm2 with subtract_mean=False,
            # and normalizes advantages at mini-batch level.
            # Empirically the performance is similar on most of the tasks.
            cls = getattr(valuenorm, cfg.value_norm["class"])
            self.value_normalizer: valuenorm.Normalizer = cls(
                # input_shape=self.agent_spec.reward_spec.shape[-2:],
                input_shape=self.agent_spec.reward_spec.shape[-1:],
                **cfg.value_norm["kwargs"],
            ).to(self.device)

    def _ensure_prev_action(self, actor_input: TensorDict):
        if not getattr(self, "actor_has_prev_action_conditioning", False):
            return actor_input
        if self.prev_act_name in actor_input.keys(True):
            return actor_input

        prev_action = None
        expected_shape = (*actor_input.batch_size, self.agent_spec.n, self.act_dim)
        if self._cached_prev_action is not None:
            cache = self._cached_prev_action
            if tuple(cache.shape) == expected_shape:
                prev_action = cache.clone()

        if prev_action is None:
            prev_action = torch.zeros(
                *expected_shape,
                device=self.device,
                dtype=torch.float32,
            )

        if "is_init" in actor_input.keys():
            is_init = actor_input["is_init"].to(device=prev_action.device, dtype=torch.bool)
            while is_init.ndim < prev_action.ndim:
                is_init = is_init.unsqueeze(-1)
            prev_action = torch.where(is_init, torch.zeros_like(prev_action), prev_action)

        actor_input[self.prev_act_name] = prev_action
        return actor_input

    def _cache_prev_action(self, action: Optional[torch.Tensor]):
        if not getattr(self, "actor_has_prev_action_conditioning", False):
            return
        if action is None:
            self._cached_prev_action = None
            return
        self._cached_prev_action = action.detach().clone()

    def reset_prev_action_cache(self):
        self._cached_prev_action = None

    def _get_actor_log_std_tensor(self) -> Optional[torch.Tensor]:
        if self.actor_params is not None:
            try:
                return self.actor_params["module"]["act_dist"]["log_std"]
            except KeyError:
                return None
        module = getattr(self.actor, "module", None)
        act_dist = getattr(module, "act_dist", None)
        log_std = getattr(act_dist, "log_std", None)
        return log_std

    def set_actor_frozen(self, frozen: bool):
        self.actor_frozen = bool(frozen)

    def set_actor_lr(self, lr: float):
        lr = float(lr)
        for group in self.actor_opt.param_groups:
            group["lr"] = lr

    def set_critic_lr(self, lr: float):
        lr = float(lr)
        for group in self.critic_opt.param_groups:
            group["lr"] = lr

    def set_kl_reg(self, target_kl: Optional[float], coef: float = 0.0, hard_stop: bool = False):
        self.ppo_target_kl = None if target_kl is None else float(target_kl)
        self.ppo_kl_coef = float(coef)
        self.ppo_kl_hard_stop = bool(hard_stop)

    def capture_anchor_actor(self):
        if self.actor_params is None:
            self.anchor_actor_params = None
            self.anchor_actor_state = {
                key: value.detach().clone()
                for key, value in self.actor.state_dict().items()
            }
            return
        anchor_td = self.actor_params.to(self.device).to_tensordict().clone()
        self.anchor_actor_params = TensorDictParams(anchor_td)

    def clear_anchor_actor(self):
        self.anchor_actor_params = None
        self.anchor_actor_state = None

    def set_anchor_loss(self, coef: float = 0.0, loss_type: str = "mse"):
        self.anchor_loss_coef = float(coef)
        self.anchor_loss_type = str(loss_type)

    def _run_actor(
        self,
        actor_input: TensorDict,
        actor_params,
        deterministic: bool = False,
        eval_action: bool = False,
    ):
        if actor_params is None:
            return self.actor(
                actor_input,
                deterministic=deterministic,
                eval_action=eval_action,
            )
        randomness = "different" if (not deterministic and not eval_action) else "error"
        if hasattr(self, "minibatch_seq_len"):
            agent_dim = len(actor_input.batch_size) - 1
            if self.cfg.share_actor:
                return vmap(
                    self.actor,
                    in_dims=(agent_dim, None),
                    out_dims=agent_dim,
                    randomness=randomness,
                )(
                    actor_input, actor_params, deterministic=deterministic, eval_action=eval_action
                )
            return vmap(
                self.actor,
                in_dims=(agent_dim, 0),
                out_dims=agent_dim,
                randomness=randomness,
            )(
                actor_input, actor_params, deterministic=deterministic, eval_action=eval_action
            )
        if self.cfg.share_actor:
            return self.actor(
                actor_input,
                actor_params,
                deterministic=deterministic,
                eval_action=eval_action,
            )
        return vmap(
            self.actor,
            in_dims=(1, 0),
            out_dims=1,
            randomness=randomness,
        )(
            actor_input,
            actor_params,
            deterministic=deterministic,
            eval_action=eval_action,
        )

    def value_op(self, tensordict: TensorDict) -> TensorDict:
        critic_input = tensordict.select(*self.critic_in_keys, strict=False)
        if "is_init" in critic_input.keys():
            critic_input["is_init"] = expand_right(
            critic_input["is_init"], (*critic_input.batch_size, self.agent_spec.n, 1)
        )
        if self.cfg.critic_input == "obs":
            critic_input.batch_size = [*critic_input.batch_size, self.agent_spec.n]
        elif "is_init" in critic_input.keys() and critic_input["is_init"].shape[-1] != 1:
            critic_input["is_init"] = critic_input["is_init"].all(-1, keepdim=True)
        tensordict = self.value_func(critic_input)
        return tensordict

    def __call__(self, tensordict: TensorDict, deterministic: bool = False):
        actor_input = tensordict.select(*self.actor_in_keys, strict=False)
        actor_input = self._ensure_prev_action(actor_input)
        if "is_init" in actor_input.keys():
            actor_input["is_init"] = expand_right(
            actor_input["is_init"], (*actor_input.batch_size, self.agent_spec.n, 1)
        )
        actor_input.batch_size = [*actor_input.batch_size, self.agent_spec.n] # [env_num, drone_num]
        actor_output = self._run_actor(
            actor_input,
            self.actor_params,
            deterministic=deterministic,
            eval_action=False,
        )

        self._cache_prev_action(actor_output.get(self.act_name, None))
        tensordict.update(actor_output)
        tensordict.update(self.value_op(tensordict))
        return tensordict

    def _prepare_predictor_batch(self, tensordict: TensorDict):
        try:
            ot_data = tensordict["next"]["agents"]["OT"]
        except KeyError:
            empty = torch.empty(0, device=self.device)
            return empty, empty, empty.long(), {"predictor_valid_windows": 0.0}

        pred_input = ot_data["pred_input"]
        target_pos = ot_data["target_pos_world"]
        strategic_label = ot_data["strategic_label"].squeeze(-1).long()
        valid = ot_data["valid"].squeeze(-1) > 0.5
        if pred_input.shape[1] <= self.predictor_horizon * self.predictor_future_stride:
            empty = torch.empty(0, device=pred_input.device)
            return empty, empty, empty.long(), {"predictor_valid_windows": 0.0}

        future_span = self.predictor_horizon * self.predictor_future_stride
        pos_windows = (
            target_pos.unfold(dimension=1, size=future_span + 1, step=1)
            .transpose(2, 3)
        )
        window_count = pos_windows.shape[1]
        future_pos = pos_windows[:, :, self.predictor_future_stride::self.predictor_future_stride]
        future_pos = future_pos[:, :, : self.predictor_horizon]
        current_pos = pos_windows[:, :, 0:1]
        future_delta = future_pos - current_pos

        env_done = tensordict["next"]["done"].reshape(*target_pos.shape[:2], -1).bool().any(dim=-1)
        done_windows = env_done.unfold(dimension=1, size=future_span + 1, step=1)[:, :window_count]
        continuity_valid = ~done_windows.any(dim=-1)
        valid_windows = valid[:, :window_count] & continuity_valid

        flat_mask = valid_windows.reshape(-1)
        selected_inputs = pred_input[:, :window_count].reshape(
            -1, *pred_input.shape[2:]
        )[flat_mask]
        selected_future_delta = future_delta.reshape(
            -1, self.predictor_horizon, target_pos.shape[-1]
        )[flat_mask]
        selected_labels = strategic_label[:, :window_count].reshape(-1)[flat_mask]
        counts = valid_windows.sum(dim=-1)
        debug_info = {
            "predictor_valid_windows": float(selected_inputs.shape[0]),
            "predictor_valid_windows_min": float(counts.min().item()) if counts.numel() else 0.0,
            "predictor_valid_windows_max": float(counts.max().item()) if counts.numel() else 0.0,
        }
        return selected_inputs, selected_future_delta, selected_labels, debug_info

    def update_predictor(self, batch: TensorDict) -> Dict[str, Any]:
        if not self.use_predictor_net or self.predictor_net is None:
            return {"predictor_loss_total": 0.0}
        pred_input = batch["pred_input"].to(self.device)
        future_delta = batch["future_delta"].to(self.device)
        strategic_label = batch["strategic_label"].to(self.device).long()
        pred = self.predictor_net(pred_input)
        loss, metrics = self.predictor_loss_fn(pred, future_delta, strategic_label)

        self.predictor_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.predictor_net.parameters(), 10.0)
        self.predictor_optimizer.step()
        self.predictor_net._ot_train_updates = int(
            getattr(self.predictor_net, "_ot_train_updates", 0)
        ) + 1
        out = {f"predictor_{k}": float(v) for k, v in metrics.items()}
        out["predictor_grad_norm"] = float(grad_norm.item())
        out["predictor_updates"] = float(getattr(self.predictor_net, "_ot_train_updates", 0))
        return out

    def update_actor(self, batch: TensorDict) -> Dict[str, Any]:
        if self.actor_frozen:
            return {
                "policy_loss": 0.0,
                "actor_grad_norm": 0.0,
                "entropy": float(self._get_actor_entropy_bonus().item()) if self._get_actor_entropy_bonus() is not None else 0.0,
                "ESS": 1.0,
                "approx_kl": 0.0,
                "kl_penalty": 0.0,
                "anchor_loss": 0.0,
                "actor_update_skipped": 1.0,
            }
        advantages = batch["advantages"]
        actor_input = batch.select(*self.actor_in_keys, strict=False).clone()
        if self.raw_act_name in batch.keys(True):
            actor_input[self.act_name] = batch[self.raw_act_name]
        actor_input = self._ensure_prev_action(actor_input)
            
        if "is_init" in actor_input.keys():
            actor_input["is_init"] = expand_right(
            actor_input["is_init"], (*actor_input.batch_size, self.agent_spec.n, 1)
        )
        actor_input.batch_size = [*actor_input.batch_size, self.agent_spec.n]

        log_probs_old = batch[self.act_logps_name]
        actor_output = self._run_actor(
            actor_input,
            self.actor_params,
            deterministic=False,
            eval_action=True,
        )

        log_probs_new = actor_output[self.act_logps_name]

        ratio = torch.exp(log_probs_new - log_probs_old)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * advantages
        )
        policy_loss = - torch.mean(torch.min(surr1, surr2) * self.act_dim)
        approx_kl = torch.mean(log_probs_old - log_probs_new)
        kl_penalty = torch.zeros((), device=policy_loss.device)
        if self.ppo_target_kl is not None and self.ppo_kl_coef > 0.0:
            kl_penalty = torch.relu(approx_kl - float(self.ppo_target_kl))
        entropy_bonus = self._get_actor_entropy_bonus()
        if entropy_bonus is None:
            entropy_bonus = -torch.mean(-log_probs_new.detach())
        entropy_loss = -entropy_bonus
        anchor_loss = torch.zeros((), device=policy_loss.device)
        if self.anchor_actor_params is not None and self.anchor_loss_coef > 0.0:
            current_pred = self._run_actor(
                actor_input,
                self.actor_params,
                deterministic=True,
                eval_action=False,
            )[self.act_name]
            with torch.no_grad():
                anchor_pred = self._run_actor(
                    actor_input,
                    self.anchor_actor_params,
                    deterministic=True,
                    eval_action=False,
                )[self.act_name]
            if self.anchor_loss_type == "l1":
                anchor_loss = F.l1_loss(current_pred, anchor_pred)
            else:
                anchor_loss = F.mse_loss(current_pred, anchor_pred)

        total_actor_loss = (
            policy_loss
            + entropy_loss * self.entropy_coef
            + kl_penalty * self.ppo_kl_coef
            + anchor_loss * self.anchor_loss_coef
        )
        if self.ppo_target_kl is not None and self.ppo_kl_hard_stop and float(approx_kl.item()) > float(self.ppo_target_kl):
            return {
                "policy_loss": policy_loss.item(),
                "actor_grad_norm": 0.0,
                "entropy": entropy_bonus.item(),
                "ESS": ((2 * ratio.logsumexp(0) - (2 * ratio).logsumexp(0)).exp().mean() / ratio.shape[0]).item(),
                "approx_kl": approx_kl.item(),
                "kl_penalty": kl_penalty.item(),
                "anchor_loss": anchor_loss.item(),
                "actor_update_skipped": 1.0,
            }

        self.actor_opt.zero_grad()
        total_actor_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actor_opt.param_groups[0]["params"], self.cfg.max_grad_norm
        )
        
        self.actor_opt.step()

        ess = (2 * ratio.logsumexp(0) - (2 * ratio).logsumexp(0)).exp().mean() / ratio.shape[0]
        return {
            "policy_loss": policy_loss.item(),
            "actor_grad_norm": grad_norm.item(),
            "entropy": entropy_bonus.item(),
            "ESS": ess.item(),
            "approx_kl": approx_kl.item(),
            "kl_penalty": kl_penalty.item(),
            "anchor_loss": anchor_loss.item(),
            "actor_update_skipped": 0.0,
        }

    def update_critic(self, batch: TensorDict) -> Dict[str, Any]:
        critic_input = batch.select(*self.critic_in_keys)
        values = self.value_op(critic_input)["state_value"]
        b_values = batch["state_value"]
        b_returns = batch["returns"]
        assert values.shape == b_values.shape == b_returns.shape
        value_pred_clipped = b_values + (values - b_values).clamp(
            -self.clip_param, self.clip_param
        )

        value_loss_clipped = self.critic_loss_fn(b_returns, value_pred_clipped)
        value_loss_original = self.critic_loss_fn(b_returns, values)

        value_loss = torch.max(value_loss_original, value_loss_clipped)

        value_loss.backward()  # do not multiply weights here
        grad_norm = nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.cfg.max_grad_norm
        )
        self.critic_opt.step()
        self.critic_opt.zero_grad(set_to_none=True)
        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        return {
            "value_loss": value_loss.mean(),
            "critic_grad_norm": grad_norm.item(),
            "explained_var": explained_var.item()
        }

    def _get_dones(self, tensordict: TensorDict):
        env_done = tensordict[("next", "done")].unsqueeze(-1)
        agent_done = tensordict.get(
            ("next", f"{self.agent_spec.name}.done"),
            env_done.expand(*env_done.shape[:-2], self.agent_spec.n, 1),
        )
        done = agent_done | env_done
        return done

    def _get_actor_log_std_stats(self) -> Dict[str, float]:
        log_std = self._get_actor_log_std_tensor()
        if log_std is None:
            return {}

        log_std = log_std.clamp(
            self.cfg.actor.get("log_std_min", -5.0),
            self.cfg.actor.get("log_std_max", 1.0),
        )

        return {
            "actor_log_std_mean": log_std.mean().item(),
            "actor_log_std_min": log_std.min().item(),
            "actor_log_std_max": log_std.max().item(),
        }

    def _get_actor_entropy_bonus(self) -> Optional[torch.Tensor]:
        log_std = self._get_actor_log_std_tensor()
        if log_std is None:
            return None

        log_std = log_std.clamp(
            self.cfg.actor.get("log_std_min", -5.0),
            self.cfg.actor.get("log_std_max", 1.0),
        )
        entropy_per_dim = log_std + (0.5 * math.log(2.0 * math.pi * math.e))
        if entropy_per_dim.ndim == 1:
            return entropy_per_dim.sum()
        return entropy_per_dim.sum(dim=-1).mean()

    def train_op(self, tensordict: TensorDict):
        if self.use_predictor_net and self.predictor_net is not None:
            self.predictor_net.train()
        if hasattr(self, "actor"):
            self.actor.train()
        if hasattr(self, "critic"):
            self.critic.train()

        tensordict = tensordict.select(*self.train_in_keys, strict=False)
        next_tensordict = tensordict["next"][:, -1]
        with torch.no_grad():
            value_output = self.value_op(next_tensordict)

        rewards = tensordict.get(("next", *self.reward_name))
        if rewards.shape[-1] != 1:
            rewards = rewards.sum(-1, keepdim=True)

        values = tensordict["state_value"]
        next_value = value_output["state_value"].squeeze(0)

        if hasattr(self, "value_normalizer"):
            values = self.value_normalizer.denormalize(values)
            next_value = self.value_normalizer.denormalize(next_value)

        dones = self._get_dones(tensordict)

        tensordict["advantages"], tensordict["returns"] = compute_gae(
            rewards,
            dones,
            values,
            next_value,
            gamma=self.gae_gamma,
            lmbda=self.gae_lambda,
        )

        advantages_mean = tensordict["advantages"].mean()
        advantages_std = tensordict["advantages"].std()
        if self.normalize_advantages:
            tensordict["advantages"] = (tensordict["advantages"] - advantages_mean) / (
                advantages_std + 1e-8
            )

        if hasattr(self, "value_normalizer"):
            self.value_normalizer.update(tensordict["returns"])
            tensordict["returns"] = self.value_normalizer.normalize(
                tensordict["returns"]
            )

        train_info = []
        predictor_info = []
        predictor_debug_info = {}

        if self.use_predictor_net:
            selected_inputs, selected_future_delta, selected_labels, predictor_debug_info = (
                self._prepare_predictor_batch(tensordict)
            )
            if selected_inputs.shape[0] > 0:
                predictor_tensordict = TensorDict(
                    {
                        "pred_input": selected_inputs,
                        "future_delta": selected_future_delta,
                        "strategic_label": selected_labels,
                    },
                    batch_size=[selected_inputs.shape[0]],
                )
                for _ in range(self.predictor_epochs):
                    dataset = make_dataset_naive(
                        predictor_tensordict,
                        int(self.predictor_num_minibatches),
                        1,
                    )
                    for minibatch in dataset:
                        predictor_info.append(
                            TensorDict(
                                {
                                    **self.update_predictor(minibatch),
                                },
                                batch_size=[],
                            )
                        )
            else:
                logging.warning("OT predictor update skipped: no valid future windows in batch.")

        # RL_update
        for ppo_epoch in range(self.ppo_epoch):
            dataset = make_dataset_naive(
                tensordict,
                int(self.cfg.num_minibatches),
                self.minibatch_seq_len if hasattr(self, "minibatch_seq_len") else 1,
            )            
            for minibatch in dataset:
                train_info.append(
                    TensorDict(
                        {
                            **self.update_actor(minibatch),
                            **self.update_critic(minibatch),
                        },
                        batch_size=[],
                    )
                )
        
        train_info = {k: v.mean().item() for k, v in torch.stack(train_info).items()}
        if self.use_predictor_net:
            if predictor_info:
                predictor_info = {
                    k: v.mean().item() for k, v in torch.stack(predictor_info).items()
                }
                train_info.update(predictor_info)
            else:
                train_info["predictor_loss_total"] = 0.0
            train_info.update(predictor_debug_info)
        train_info["advantages_mean"] = advantages_mean.item()
        train_info["advantages_std"] = advantages_std.item()
        if isinstance(self.agent_spec.action_spec, (BoundedTensorSpec, UnboundedTensorSpec)):
            if self.raw_act_name in tensordict.keys(True):
                raw_action = tensordict[self.raw_act_name]
                cmd_action = tensordict[self.act_name]
                train_info["action_norm"] = raw_action.norm(dim=-1).mean().item()
                train_info["cmd_norm"] = cmd_action.norm(dim=-1).mean().item()
                if self.cfg.actor.tanh:
                    # actor already outputs squashed actions in (-1,1)
                    train_info["action_squash_sat_frac"] = (
                        raw_action.abs() > 0.98
                    ).float().mean().item()
                else:
                    # actor outputs raw unbounded actions; check post-tanh saturation
                    train_info["action_squash_sat_frac"] = (
                        torch.tanh(raw_action).abs() > 0.98
                    ).float().mean().item()
                train_info["cmd_saturation_frac"] = (
                    cmd_action.abs() > 0.98
                ).float().mean().item()
            else:
                train_info["action_norm"] = tensordict[self.act_name].norm(dim=-1).mean().item()
                train_info["cmd_norm"] = train_info["action_norm"]
            train_info.update(self._get_actor_log_std_stats())
        if hasattr(self, "value_normalizer"):
            train_info["value_running_mean"] = self.value_normalizer.running_mean.mean().item()
        
        self.n_updates += 1
        # step LR schedulers if configured
        if hasattr(self, "critic_opt_scheduler"):
            self.critic_opt_scheduler.step()
        if hasattr(self, "actor_opt_scheduler") and not self.actor_frozen:
            self.actor_opt_scheduler.step()
        return {f"{self.agent_spec.name}/{k}": v for k, v in train_info.items()}

    def state_dict(self):
        state_dict = {
            "critic": self.critic.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_params": self.actor_params,
            "actor_uses_module_params": self.actor_params is None,
            "value_normalizer": self.value_normalizer.state_dict(),
            "actor_has_prev_action_conditioning": bool(
                getattr(self, "actor_has_prev_action_conditioning", False)
            ),
            "actor_prev_action_condition_hidden_dim": int(
                self.cfg.actor.prev_action_conditioning.hidden_dim
            )
            if getattr(self, "actor_has_prev_action_conditioning", False)
            else 0,
            "actor_has_rnn": bool(self.cfg.actor.get("rnn", None) is not None),
            "actor_rnn_hidden_size": int(self.cfg.actor.rnn.kwargs.hidden_size)
            if self.cfg.actor.get("rnn", None) is not None
            else 0,
            # Optimizer states for seamless checkpoint resume
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }
        if self.use_predictor_net and self.predictor_net is not None:
            state_dict["ot_predictor"] = self.predictor_net.state_dict()
            state_dict["ot_predictor_updates"] = int(
                getattr(self.predictor_net, "_ot_train_updates", 0)
            )
            if self.predictor_optimizer is not None:
                state_dict["ot_predictor_opt"] = self.predictor_optimizer.state_dict()
        return state_dict

    def set_entropy_coef(self, value: float):
        self.entropy_coef = float(value)

    def set_actor_log_std(self, value: float):
        log_std = self._get_actor_log_std_tensor()
        if log_std is None:
            return None
        value = float(value)
        log_std_min = float(self.cfg.actor.get("log_std_min", -5.0))
        log_std_max = float(self.cfg.actor.get("log_std_max", 1.0))
        value = max(min(value, log_std_max), log_std_min)
        with torch.no_grad():
            log_std.fill_(value)
        return value

    def _load_actor_functional_params_into_module(self, actor_params) -> None:
        loaded_actor_params = actor_params.to(self.device).to_tensordict()
        module_state = self.actor.state_dict()
        translated = {}
        for key in loaded_actor_params.keys(include_nested=True, leaves_only=True):
            value = loaded_actor_params.get(key)
            name = ".".join(str(part) for part in key)
            candidates = [name]
            if name.startswith("module."):
                candidates.append(name[len("module."):])
            else:
                candidates.append(f"module.{name}")
            for candidate in candidates:
                if candidate in module_state and tuple(module_state[candidate].shape) == tuple(value.shape):
                    translated[candidate] = value.detach().to(module_state[candidate].device)
                    break
        if not translated:
            logging.warning("No actor functional params matched the current actor module.")
            return
        module_state.update(translated)
        self.actor.load_state_dict(module_state, strict=False)
    
    def load_state_dict(self, state_dict):
        if self.use_predictor_net and self.predictor_net is not None and "ot_predictor" in state_dict:
            self.predictor_net.load_state_dict(state_dict["ot_predictor"], strict=False)
            self.predictor_net._ot_train_updates = int(state_dict.get("ot_predictor_updates", 0))
            if self.predictor_optimizer is not None and "ot_predictor_opt" in state_dict:
                try:
                    self.predictor_optimizer.load_state_dict(state_dict["ot_predictor_opt"])
                    logging.info("Restored OT predictor optimizer state from checkpoint.")
                except Exception as e:
                    logging.warning("Failed to restore OT predictor optimizer state: %s.", e)
        if self.actor_params is None:
            if "actor" in state_dict:
                self.actor.load_state_dict(state_dict["actor"], strict=False)
            elif state_dict.get("actor_params", None) is not None:
                self._load_actor_functional_params_into_module(state_dict["actor_params"])
        else:
            loaded_actor_params = state_dict["actor_params"].to(self.device).to_tensordict()
            current_actor_params = self.actor_params.to_tensordict()
            current_actor_params.update(loaded_actor_params)
            self.actor_params = TensorDictParams(current_actor_params)
        self._reset_actor_optimizer()
        # Restore optimizer states if available (backward compatible with old checkpoints)
        if "actor_opt" in state_dict:
            try:
                self.actor_opt.load_state_dict(state_dict["actor_opt"])
                logging.info("Restored actor optimizer state from checkpoint.")
            except Exception as e:
                logging.warning("Failed to restore actor optimizer state: %s. Using fresh optimizer.", e)
        else:
            logging.info("No actor optimizer state in checkpoint. Using fresh optimizer.")
        self.critic.load_state_dict(state_dict["critic"])
        if "critic_opt" in state_dict:
            try:
                self.critic_opt.load_state_dict(state_dict["critic_opt"])
                logging.info("Restored critic optimizer state from checkpoint.")
            except Exception as e:
                logging.warning("Failed to restore critic optimizer state: %s. Using fresh optimizer.", e)
        else:
            logging.info("No critic optimizer state in checkpoint. Using fresh optimizer.")
        self.value_normalizer.load_state_dict(state_dict["value_normalizer"])
        self.reset_prev_action_cache()

def make_dataset_naive(
    tensordict: TensorDict, num_minibatches: int = 4, seq_len: int = 1
):
    if seq_len > 1:
        N, T = tensordict.shape
        T = (T // seq_len) * seq_len
        tensordict = tensordict[:, :T].reshape(-1, seq_len)
        perm = torch.randperm(
            (tensordict.shape[0] // num_minibatches) * num_minibatches,
            device=tensordict.device,
        ).reshape(num_minibatches, -1)
        for indices in perm:
            yield tensordict[indices]
    else:
        tensordict = tensordict.reshape(-1)
        perm = torch.randperm(
            (tensordict.shape[0] // num_minibatches) * num_minibatches,
            device=tensordict.device,
        ).reshape(num_minibatches, -1)
        for indices in perm:
            yield tensordict[indices]


from .modules.distributions import (
    DiagGaussian,
    MultiCategoricalModule,
    TanhIndependentNormalModule
)

from .modules.rnn import GRU
from .common import make_encoder

def make_ppo_actor(cfg, observation_spec: TensorSpec, action_spec: TensorSpec):
    encoder = make_encoder(cfg, observation_spec)
    feature_dim = encoder.output_shape.numel()

    if isinstance(action_spec, MultiDiscreteTensorSpec):
        act_dist = MultiCategoricalModule(
            feature_dim, 
            torch.as_tensor(action_spec.nvec.storage().float()).long()
        )
    elif isinstance(action_spec, DiscreteTensorSpec):
        act_dist = MultiCategoricalModule(feature_dim, [action_spec.space.n])
    elif isinstance(action_spec, (UnboundedTensorSpec, BoundedTensorSpec)):
        action_dim = action_spec.shape[-1]
        if cfg.tanh:
            act_dist = TanhIndependentNormalModule(
                feature_dim,
                action_dim,
                log_std_init=cfg.get("log_std_init", 0.0),
                log_std_min=cfg.get("log_std_min", -5.0),
                log_std_max=cfg.get("log_std_max", 1.0),
            )
        else:
            act_dist = DiagGaussian(
                feature_dim,
                action_dim,
                False,
                0.01,
                log_std_init=cfg.get("log_std_init", 0.0),
                log_std_min=cfg.get("log_std_min", -5.0),
                log_std_max=cfg.get("log_std_max", 1.0),
            )
    #     # act_dist = IndependentNormalModule(encoder.output_shape.numel(), action_dim, False)
    else:
        raise NotImplementedError(action_spec)

    if cfg.get("rnn", None):
        rnn_cls = {"gru": GRU}[cfg.rnn.cls.lower()]
        rnn = rnn_cls(input_size=feature_dim, **cfg.rnn.kwargs)
    else:
        rnn = None

    prev_action_conditioner = None
    prev_action_cfg = cfg.get("prev_action_conditioning", None)
    if prev_action_cfg is not None and prev_action_cfg.get("enabled", False):
        hidden_dim = int(prev_action_cfg.get("hidden_dim", feature_dim))
        action_dim = action_spec.shape[-1]
        prev_action_conditioner = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )
        # Preserve checkpoint compatibility: new branch starts as a no-op residual.
        nn.init.zeros_(prev_action_conditioner[-1].weight)
        nn.init.zeros_(prev_action_conditioner[-1].bias)

    return Actor(
        encoder,
        act_dist,
        rnn,
        prev_action_conditioner=prev_action_conditioner,
    )

def make_critic(cfg, state_spec: TensorSpec, reward_spec: TensorSpec, centralized=False):
    assert isinstance(reward_spec, (UnboundedTensorSpec, BoundedTensorSpec))
    encoder = make_encoder(cfg, state_spec)
    
    if cfg.get("rnn", None):
        rnn_cls = {"gru": GRU}[cfg.rnn.cls.lower()]
        rnn = rnn_cls(input_size=encoder.output_shape.numel(), **cfg.rnn.kwargs)
    else:
        rnn = None

    if centralized:
        v_out = nn.Linear(encoder.output_shape.numel(), reward_spec.shape[-2:].numel())
        nn.init.orthogonal_(v_out.weight, cfg.gain)
        return Critic(encoder, rnn, v_out, reward_spec.shape[-2:])
    else:
        v_out = nn.Linear(encoder.output_shape.numel(), reward_spec.shape[-1])
        nn.init.orthogonal_(v_out.weight, cfg.gain)
        return Critic(encoder, rnn, v_out, reward_spec.shape[-1:])

class Actor(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        act_dist: nn.Module,
        rnn: Optional[nn.Module] = None,
        prev_action_conditioner: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.act_dist = act_dist
        self.rnn = rnn
        self.prev_action_conditioner = prev_action_conditioner

    def forward(
        self,
        obs: Union[torch.Tensor, TensorDict],
        prev_action: torch.Tensor = None,
        action: torch.Tensor = None,
        rnn_state=None,
        is_init=None,
        deterministic=False,
        eval_action=False
    ):
        if eval_action and action is None and prev_action is not None and self.prev_action_conditioner is None:
            action = prev_action
            prev_action = None
        actor_features = self.encoder(obs)
        if self.rnn is not None:
            actor_features, rnn_state = self.rnn(actor_features, rnn_state, is_init)    
        else:
            rnn_state = None
        dist_features = actor_features
        if self.prev_action_conditioner is not None and prev_action is not None:
            prev_action = torch.tanh(prev_action).to(dist_features.dtype)
            prev_action_input = torch.cat([actor_features, prev_action], dim=-1)
            dist_features = dist_features + self.prev_action_conditioner(prev_action_input)
        action_dist = self.act_dist(dist_features)

        if eval_action:
            action_log_probs = action_dist.log_prob(action).unsqueeze(-1)
            try:
                action_entropy = action_dist.entropy()
                if action_entropy is None:
                    raise NotImplementedError
                action_entropy = action_entropy.unsqueeze(-1)
            except NotImplementedError:
                action_entropy = -action_log_probs.detach()
            outputs = [action, action_log_probs, action_entropy]
            if self.rnn is not None:
                outputs.append(rnn_state)
            return tuple(outputs)
        else:
            action = action_dist.mode if deterministic else action_dist.sample()
            action_log_probs = action_dist.log_prob(action).unsqueeze(-1)
            try:
                action_entropy = action_dist.entropy()
                if action_entropy is None:
                    raise NotImplementedError
                action_entropy = action_entropy.unsqueeze(-1)
            except NotImplementedError:
                action_entropy = -action_log_probs.detach()
            outputs = [action, action_log_probs, action_entropy]
            if self.rnn is not None:
                outputs.append(rnn_state)
            return tuple(outputs)

class Critic(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        rnn: nn.Module,
        v_out: nn.Module,
        output_shape: torch.Size=torch.Size((-1,)),
    ):
        super().__init__()
        self.base = base
        self.rnn = rnn
        self.v_out = v_out
        self.output_shape = output_shape

    def forward(
        self, 
        critic_input: torch.Tensor,
        rnn_state: torch.Tensor=None,
        is_init: torch.Tensor=None,
    ):
        critic_features = self.base(critic_input)
        if self.rnn is not None:
            critic_features, rnn_state = self.rnn(critic_features, rnn_state, is_init)
        else:
            rnn_state = None

        values = self.v_out(critic_features)

        if len(self.output_shape) > 1:
            values = values.unflatten(-1, self.output_shape)
        return values, rnn_state
