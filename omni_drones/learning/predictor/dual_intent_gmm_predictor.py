from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


STRATEGIC_LABELS = ["Far_Rush", "Close_Evade", "Goal_Priority", "Goal_Sprint"]


@dataclass
class StrategicGMMConfig:
    obs_dim: int = 30
    d_model: int = 128
    lstm_layers: int = 1
    n_heads: int = 4
    n_components: int = 3
    pred_horizon: int = 10
    dropout: float = 0.1
    min_var: float = 1e-3
    n_strategic: int = 4
    gru_layers: int = 1
    detach_intent_for_gmm: bool = False


class GRUTrajectoryDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_components: int,
        gru_layers: int = 1,
        dropout: float = 0.1,
        min_var: float = 1e-3,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.K = int(n_components)
        self.min_var = float(min_var)
        self.gru_layers = int(gru_layers)

        self.start_token = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.z_to_h0 = nn.Sequential(
            nn.Linear(d_model, d_model * gru_layers),
            nn.Tanh(),
        )
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.step_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_components * 3 * 2),
        )
        final = self.step_head[-1]
        nn.init.xavier_uniform_(final.weight, gain=0.05)
        nn.init.zeros_(final.bias)

    def forward(self, z: torch.Tensor, horizon: int) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = z.shape[0]
        h = self.z_to_h0(z).view(self.gru_layers, batch_size, self.d_model).contiguous()
        inp = self.start_token.expand(batch_size, -1).unsqueeze(1)
        means = []
        vars_ = []
        for _ in range(int(horizon)):
            out, h = self.gru(inp, h)
            params = self.step_head(out.squeeze(1)).view(batch_size, self.K, 3, 2)
            means.append(params[..., 0])
            vars_.append(F.softplus(params[..., 1]) + self.min_var)
            inp = out
        return torch.stack(means, dim=2), torch.stack(vars_, dim=2)


class StrategicGMMTrajectoryPredictor(nn.Module):
    """Strategic-intent conditioned GMM predictor with no maneuver-class head."""

    def __init__(self, cfg: StrategicGMMConfig | None = None):
        super().__init__()
        self.cfg = cfg or StrategicGMMConfig()
        self.K = self.cfg.n_components
        self.T = self.cfg.pred_horizon

        hidden_dim = self.cfg.d_model // 2
        if hidden_dim * 2 != self.cfg.d_model:
            raise ValueError("d_model must be even for the BiLSTM encoder.")

        self.encoder = nn.LSTM(
            input_size=self.cfg.obs_dim,
            hidden_size=hidden_dim,
            num_layers=self.cfg.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.cfg.dropout if self.cfg.lstm_layers > 1 else 0.0,
        )
        self.strategic_query = nn.Parameter(torch.randn(1, 1, self.cfg.d_model) * 0.02)
        self.trajectory_query = nn.Parameter(torch.randn(1, 1, self.cfg.d_model) * 0.02)
        self.strategic_attn = nn.MultiheadAttention(
            self.cfg.d_model, self.cfg.n_heads, dropout=self.cfg.dropout, batch_first=True
        )
        self.trajectory_attn = nn.MultiheadAttention(
            self.cfg.d_model, self.cfg.n_heads, dropout=self.cfg.dropout, batch_first=True
        )
        self.strategic_head = nn.Sequential(
            nn.LayerNorm(self.cfg.d_model),
            nn.Linear(self.cfg.d_model, self.cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.d_model // 2, self.cfg.n_strategic),
        )
        fuse_dim = self.cfg.d_model + self.cfg.d_model + self.cfg.n_strategic
        self.fuse = nn.Sequential(
            nn.LayerNorm(fuse_dim),
            nn.Linear(fuse_dim, self.cfg.d_model),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.d_model, self.cfg.d_model),
            nn.GELU(),
        )
        self.weight_head = nn.Linear(self.cfg.d_model, self.K)
        nn.init.zeros_(self.weight_head.weight)
        nn.init.zeros_(self.weight_head.bias)
        self.traj_decoder = GRUTrajectoryDecoder(
            self.cfg.d_model,
            self.K,
            gru_layers=self.cfg.gru_layers,
            dropout=self.cfg.dropout,
            min_var=self.cfg.min_var,
        )

    def forward(
        self,
        obs_history: torch.Tensor,
        gt_strategic: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = obs_history.shape[0]
        h_env, _ = self.encoder(obs_history)

        q_strategic = self.strategic_query.expand(batch_size, -1, -1)
        q_trajectory = self.trajectory_query.expand(batch_size, -1, -1)
        h_strategic, attn_strategic = self.strategic_attn(q_strategic, h_env, h_env)
        h_trajectory, attn_trajectory = self.trajectory_attn(q_trajectory, h_env, h_env)
        h_strategic = h_strategic.squeeze(1)
        h_trajectory = h_trajectory.squeeze(1)

        strategic_logits = self.strategic_head(h_strategic)
        strategic_probs = F.softmax(strategic_logits, dim=-1)
        if gt_strategic is not None:
            strategic_for_gmm = F.one_hot(
                gt_strategic, num_classes=self.cfg.n_strategic
            ).to(dtype=h_env.dtype)
        else:
            strategic_for_gmm = (
                strategic_probs.detach()
                if self.cfg.detach_intent_for_gmm
                else strategic_probs
            )

        z = self.fuse(torch.cat([h_strategic, h_trajectory, strategic_for_gmm], dim=-1))
        gmm_weights = F.softmax(self.weight_head(z), dim=-1)
        gmm_means, gmm_vars = self.traj_decoder(z, horizon=self.T)
        mixed_mu, mixed_var = self._compute_mixture(gmm_weights, gmm_means, gmm_vars)
        return {
            "strategic_logits": strategic_logits,
            "strategic_probs": strategic_probs,
            "gmm_weights": gmm_weights,
            "gmm_means": gmm_means,
            "gmm_vars": gmm_vars,
            "mixed_mu": mixed_mu,
            "mixed_var": mixed_var,
            "attn_strategic": attn_strategic.squeeze(1),
            "attn_trajectory": attn_trajectory.squeeze(1),
        }

    @staticmethod
    def _compute_mixture(
        weights: torch.Tensor,
        means: torch.Tensor,
        vars_: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        w = weights.unsqueeze(-1).unsqueeze(-1)
        mixed_mu = (w * means).sum(dim=1)
        mixed_var = (w * (vars_ + means.square())).sum(dim=1) - mixed_mu.square()
        return mixed_mu, mixed_var.clamp_min(1e-6)


class StrategicGMMLoss(nn.Module):
    def __init__(
        self,
        lambda_strategic: float = 1.0,
        lambda_gmm: float = 1.0,
        lambda_speed: float = 0.02,
        lambda_accel: float = 0.02,
        lambda_height: float = 0.02,
        dt_pred: float = 0.05,
        max_speed: float = 3.0,
        max_accel: float = 3.0,
        min_height_delta: float | None = None,
    ) -> None:
        super().__init__()
        self.lambda_strategic = float(lambda_strategic)
        self.lambda_gmm = float(lambda_gmm)
        self.lambda_speed = float(lambda_speed)
        self.lambda_accel = float(lambda_accel)
        self.lambda_height = float(lambda_height)
        self.dt_pred = float(dt_pred)
        self.max_speed = float(max_speed)
        self.max_accel = float(max_accel)
        self.min_height_delta = min_height_delta

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        gt_future_delta: torch.Tensor,
        gt_strategic: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_strategic = F.cross_entropy(pred["strategic_logits"], gt_strategic)
        l_gmm = self.gmm_nll(
            pred["gmm_weights"],
            pred["gmm_means"],
            pred["gmm_vars"],
            gt_future_delta,
        )
        l_speed, l_accel, l_height = self.physics_losses(pred["gmm_means"])
        loss = (
            self.lambda_strategic * l_strategic
            + self.lambda_gmm * l_gmm
            + self.lambda_speed * l_speed
            + self.lambda_accel * l_accel
            + self.lambda_height * l_height
        )
        with torch.no_grad():
            strategic_acc = (
                pred["strategic_logits"].argmax(dim=-1) == gt_strategic
            ).float().mean()
            gt_expanded = gt_future_delta.unsqueeze(1)
            per_comp_ade = torch.norm(pred["gmm_means"] - gt_expanded, dim=-1).mean(dim=-1)
            minade = per_comp_ade.min(dim=-1).values.mean()
            fde = torch.norm(pred["mixed_mu"][:, -1] - gt_future_delta[:, -1], dim=-1).mean()
            entropy = -(pred["gmm_weights"] * torch.log(pred["gmm_weights"] + 1e-8)).sum(dim=-1).mean()
        return loss, {
            "loss_total": float(loss.detach().cpu()),
            "loss_strategic": float(l_strategic.detach().cpu()),
            "loss_gmm_nll": float(l_gmm.detach().cpu()),
            "loss_speed": float(l_speed.detach().cpu()),
            "loss_accel": float(l_accel.detach().cpu()),
            "loss_height": float(l_height.detach().cpu()),
            "strategic_accuracy": float(strategic_acc.detach().cpu()),
            "minADE": float(minade.detach().cpu()),
            "FDE": float(fde.detach().cpu()),
            "gmm_weight_entropy": float(entropy.detach().cpu()),
        }

    @staticmethod
    def gmm_nll(
        weights: torch.Tensor,
        means: torch.Tensor,
        vars_: torch.Tensor,
        gt: torch.Tensor,
    ) -> torch.Tensor:
        gt = gt.unsqueeze(1)
        log_prob = -0.5 * (
            1.8378770664093453
            + torch.log(vars_ + 1e-6)
            + (gt - means).square() / (vars_ + 1e-6)
        )
        log_component = log_prob.sum(dim=(-1, -2)) + torch.log(weights + 1e-8)
        return -torch.logsumexp(log_component, dim=-1).mean()

    def physics_losses(self, means: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if means.shape[2] < 2:
            zero = means.sum() * 0.0
            return zero, zero, zero
        vel = (means[:, :, 1:] - means[:, :, :-1]) / max(self.dt_pred, 1e-6)
        speed = torch.norm(vel, dim=-1)
        l_speed = torch.relu(speed - self.max_speed).square().mean()
        if vel.shape[2] < 2:
            l_accel = means.sum() * 0.0
        else:
            accel = (vel[:, :, 1:] - vel[:, :, :-1]) / max(self.dt_pred, 1e-6)
            l_accel = torch.relu(torch.norm(accel, dim=-1) - self.max_accel).square().mean()
        if self.min_height_delta is None:
            l_height = means.sum() * 0.0
        else:
            l_height = torch.relu(float(self.min_height_delta) - means[..., 2]).square().mean()
        return l_speed, l_accel, l_height


# Backward-compatible aliases for older imports.
DualIntentGMMConfig = StrategicGMMConfig
DualIntentQueryGMMPredictor = StrategicGMMTrajectoryPredictor
DualIntentGMMLoss = StrategicGMMLoss
