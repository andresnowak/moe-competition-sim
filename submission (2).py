from typing import Tuple

import numpy as np
import torch


MIN_PAR_GAIN = 0.0
MIN_PAR_GAIN_PER_MOVED_SLOT = 5e-4
TRANSFER_PAYOFF_MARGIN = 5.0
BALANCED_COMPUTE_SECONDS = 60.0
EXPERT_BYTES = 88_080_384
TRANSFER_BANDWIDTH_BYTES_PER_SECOND = 900_000_000_000
SHORT_HALF_LIFE_FRAC = 16.0
LONG_HALF_LIFE_FRAC = 4.0
EMA_STATE_BLEND = 0.5
PERSISTENCE_RATIO_FLOOR = 0.8
SHORT_BOOST = 0.6
TOPK_HOT_FRACTION = 0.15
MIN_REPLICA_VALUE_GAIN = 1.5e-4
COLD_MOVE_OVERRIDE_GAIN = 2.0e-3

_PREVIOUS_DEPLOYMENT = None
_PREVIOUS_SIGNATURE = None
_EMA_SHORT = None
_EMA_LONG = None
_EMA_SIGNATURE = None


def balanced_packing(
    weight: torch.Tensor, num_packs: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Greedily pack equal-size groups so pack weights are roughly balanced."""
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(
            weight.size(-1), dtype=torch.int64, device=weight.device
        ).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    indices = weight.float().sort(-1, descending=True).indices.cpu()
    pack_index = torch.full_like(weight, fill_value=-1, dtype=torch.int64, device="cpu")
    rank_in_pack = torch.full_like(pack_index, fill_value=-1)
    for layer in range(num_layers):
        pack_weights = [0.0] * num_packs
        pack_items = [0] * num_packs
        for group in indices[layer]:
            pack = min(
                (idx for idx in range(num_packs) if pack_items[idx] < groups_per_pack),
                key=pack_weights.__getitem__,
            )
            pack_index[layer, group] = pack
            rank_in_pack[layer, group] = pack_items[pack]
            pack_weights[pack] += float(weight[layer, group])
            pack_items[pack] += 1
    return pack_index, rank_in_pack


def replicate_experts(
    weight: torch.Tensor, num_phy: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign redundant physical copies to the largest load-per-copy experts."""
    num_layers, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(
        num_layers, 1
    )
    rank = torch.zeros(num_layers, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(num_layers, num_log, dtype=torch.int64, device=device)
    layer_ids = torch.arange(num_layers, dtype=torch.int64, device=device)

    for physical_idx in range(num_log, num_phy):
        redundant_indices = (weight / logcnt).max(dim=-1).indices
        phy2log[:, physical_idx] = redundant_indices
        rank[:, physical_idx] = logcnt[layer_ids, redundant_indices]
        logcnt[layer_ids, redundant_indices] += 1
    return phy2log, rank, logcnt


def inverse(perm: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm)
    inv.scatter_(
        1,
        perm,
        torch.arange(perm.size(1), dtype=torch.int64, device=perm.device).expand(
            perm.shape
        ),
    )
    return inv


def rebalance_experts_global(
    weight: torch.Tensor,
    num_replicas: int,
    num_gpus: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DeepSeek-style global replication and GPU packing."""
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    assert num_replicas % num_gpus == 0
    phy_experts_per_gpu = num_replicas // num_gpus

    phy2log, phyrank, logcnt = replicate_experts(weight, num_replicas)
    tokens_per_phy = (weight / logcnt).gather(-1, phy2log)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy, num_gpus)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2log = phy2log.gather(-1, pphy2phy)
    pphyrank = phyrank.gather(-1, pphy2phy)
    maxlogcnt = logcnt.max().item()
    log2phy = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    log2phy.view(num_layers, -1).scatter_(
        -1,
        pphy2log * maxlogcnt + pphyrank,
        torch.arange(num_replicas, dtype=torch.int64, device=log2phy.device).expand(
            num_layers, -1
        ),
    )
    return pphy2log, log2phy, logcnt


def calculate_par(load: np.ndarray, deployment: np.ndarray) -> np.ndarray:
    """Calculate per-layer peak-to-average ratio for a deployment."""
    n_layers, n_expert = load.shape
    n_devices, exp_per_dev = deployment.shape[1:]
    pars = np.zeros(n_layers, dtype=np.float64)
    for layer in range(n_layers):
        flat_deployment = deployment[layer].reshape(-1)
        counts = np.bincount(flat_deployment, minlength=n_expert)
        weights = load[layer] / counts
        device_loads = weights[flat_deployment].reshape((n_devices, exp_per_dev)).sum(-1)
        pars[layer] = device_loads.max() / device_loads.mean()
    return pars


def transfer_time_seconds(moved_slots: np.ndarray) -> np.ndarray:
    """Estimate network transfer time for each layer's changed slots."""
    return moved_slots * EXPERT_BYTES / TRANSFER_BANDWIDTH_BYTES_PER_SECOND


def compute_time_saved_seconds(par_improvement: np.ndarray, n_layers: int) -> np.ndarray:
    """Convert layer PAR improvement into modeled end-to-end time savings."""
    return BALANCED_COMPUTE_SECONDS * par_improvement / n_layers


def decay_from_half_life(half_life: float) -> float:
    if half_life <= 0:
        return 0.0
    return float(0.5 ** (1.0 / half_life))


def ema_window_weight(tokens_per_expert: np.ndarray, decay: float) -> np.ndarray:
    """Collapse [time, layers, experts] hotness with newer iterations weighted more."""
    if tokens_per_expert.ndim != 3:
        raise ValueError("tokens_per_expert must have shape [time, layers, experts]")
    if not 0.0 < decay <= 1.0:
        raise ValueError("decay must be in (0, 1]")
    window = tokens_per_expert.astype(np.float64)
    n_steps = window.shape[0]
    weights = decay ** np.arange(n_steps - 1, -1, -1, dtype=np.float64)
    weights = weights / weights.sum()
    return (window * weights[:, None, None]).sum(axis=0)


def update_ema_state(
    hotness: np.ndarray,
    n_layers: int,
    n_experts: int,
) -> tuple[np.ndarray, np.ndarray]:
    global _EMA_SHORT, _EMA_LONG, _EMA_SIGNATURE

    signature = (n_layers, n_experts, hotness.shape[0])
    window = hotness
    short_half_life = max(1.0, window.shape[0] / SHORT_HALF_LIFE_FRAC)
    long_half_life = max(1.0, window.shape[0] / LONG_HALF_LIFE_FRAC)
    short_decay = decay_from_half_life(short_half_life)
    long_decay = decay_from_half_life(long_half_life)
    window_short = ema_window_weight(window, short_decay)
    window_long = ema_window_weight(window, long_decay)

    if _EMA_SIGNATURE != signature or _EMA_SHORT is None or _EMA_LONG is None:
        _EMA_SIGNATURE = signature
        _EMA_SHORT = window_short
        _EMA_LONG = window_long
        return _EMA_SHORT, _EMA_LONG

    _EMA_SHORT = EMA_STATE_BLEND * _EMA_SHORT + (1.0 - EMA_STATE_BLEND) * window_short
    _EMA_LONG = EMA_STATE_BLEND * _EMA_LONG + (1.0 - EMA_STATE_BLEND) * window_long
    return _EMA_SHORT, _EMA_LONG


def blended_load(short_ema: np.ndarray, long_ema: np.ndarray) -> np.ndarray:
    """Blend short/long EMA, down-weighting transient spikes."""
    eps = 1e-9
    ratio = long_ema / (short_ema + eps)
    persistence = np.clip(ratio, 0.0, 1.0)
    boost = (short_ema - long_ema) * persistence
    return long_ema + SHORT_BOOST * boost


def topk_hot_mask(load: np.ndarray, topk_fraction: float) -> np.ndarray:
    """Return a boolean mask of top-k hot experts per layer."""
    n_layers, n_experts = load.shape
    top_k = max(1, int(round(n_experts * topk_fraction)))
    top_k = min(top_k, n_experts)
    mask = np.zeros_like(load, dtype=bool)
    for layer in range(n_layers):
        top_indices = np.argpartition(load[layer], -top_k)[-top_k:]
        mask[layer, top_indices] = True
    return mask


def marginal_replica_gain(
    load: np.ndarray,
    deployment: np.ndarray,
    candidate_mask: np.ndarray,
) -> np.ndarray:
    """Estimate delta PAR for adding one more replica per candidate expert.

    This is an optimistic proxy: it reduces per-copy load for existing replicas
    without changing placement.
    """
    n_layers, n_experts = load.shape
    n_devices, exp_per_dev = deployment.shape[1:]
    gains = np.zeros((n_layers, n_experts), dtype=np.float64)
    for layer in range(n_layers):
        flat = deployment[layer].reshape(-1)
        counts = np.bincount(flat, minlength=n_experts)
        weights = load[layer] / counts
        device_loads = weights[flat].reshape((n_devices, exp_per_dev)).sum(-1)
        old_par = device_loads.max() / device_loads.mean()
        for expert in np.flatnonzero(candidate_mask[layer]):
            if counts[expert] <= 0:
                continue
            old_w = weights[expert]
            new_w = load[layer, expert] / (counts[expert] + 1)
            delta = old_w - new_w
            if delta <= 0:
                continue
            per_device = (deployment[layer] == expert).sum(axis=1)
            new_device_loads = device_loads - delta * per_device
            new_mean = new_device_loads.mean()
            new_par = new_device_loads.max() / new_mean
            gains[layer, expert] = old_par - new_par
    return gains


def rebalance(
    hotness: np.ndarray,
    n_device: int,
    n_red_expert: int,
) -> tuple[bool, np.ndarray, np.ndarray, None]:
    """DeepSeek placement with transfer-aware sticky layer movement."""
    global _PREVIOUS_DEPLOYMENT, _PREVIOUS_SIGNATURE

    n_experts = hotness.shape[-1]
    n_layers = hotness.shape[1]
    n_physical = n_experts + n_red_expert
    n_exp_per_dev = n_physical // n_device
    if n_physical % n_device != 0:
        raise ValueError("n_experts + n_red_expert must be divisible by n_device")

    short_ema, long_ema = update_ema_state(hotness, n_layers, n_experts)
    load = blended_load(short_ema, long_ema)
    hot_mask = topk_hot_mask(load, TOPK_HOT_FRACTION)
    physical_to_logical_map, _, _ = rebalance_experts_global(
        weight=torch.from_numpy(load),
        num_replicas=n_physical,
        num_gpus=n_device,
    )
    candidate = physical_to_logical_map.numpy().reshape(
        (n_layers, n_device, n_exp_per_dev)
    )

    signature = (n_layers, n_experts, n_device, n_red_expert)
    if _PREVIOUS_SIGNATURE != signature or _PREVIOUS_DEPLOYMENT is None:
        _PREVIOUS_SIGNATURE = signature
        _PREVIOUS_DEPLOYMENT = candidate.copy()
        layers_priority = np.arange(n_layers, dtype=np.int64)
        return True, layers_priority, candidate, None

    old_par = calculate_par(load, _PREVIOUS_DEPLOYMENT)
    new_par = calculate_par(load, candidate)
    moved_slots = np.sum(_PREVIOUS_DEPLOYMENT != candidate, axis=(1, 2))
    par_improvement = old_par - new_par
    marginal_gain = marginal_replica_gain(load, candidate, hot_mask)
    max_marginal_gain = marginal_gain.max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        gain_per_moved_slot = np.where(
            moved_slots > 0,
            par_improvement / moved_slots,
            0.0,
        )

    n_expert_scale = 1.0 + 0.25 * np.log1p(n_experts / 32.0)
    min_gain_per_slot = MIN_PAR_GAIN_PER_MOVED_SLOT * n_expert_scale
    min_par_gain = MIN_PAR_GAIN * n_expert_scale

    cold_movement = np.zeros(n_layers, dtype=bool)
    for layer in range(n_layers):
        if moved_slots[layer] == 0:
            continue
        changed = _PREVIOUS_DEPLOYMENT[layer] != candidate[layer]
        moved_experts = np.unique(
            np.concatenate(
                (
                    _PREVIOUS_DEPLOYMENT[layer][changed].reshape(-1),
                    candidate[layer][changed].reshape(-1),
                )
            )
        )
        if len(moved_experts) == 0:
            continue
        cold_hits = ~hot_mask[layer, moved_experts]
        cold_movement[layer] = np.any(cold_hits)

    transfer_time = transfer_time_seconds(moved_slots)
    compute_saved = compute_time_saved_seconds(par_improvement, n_layers)
    accepted = np.flatnonzero(
        (moved_slots > 0)
        & (par_improvement > min_par_gain)
        & (gain_per_moved_slot > min_gain_per_slot)
        & (compute_saved > TRANSFER_PAYOFF_MARGIN * transfer_time)
        & (max_marginal_gain > MIN_REPLICA_VALUE_GAIN)
        & (~cold_movement | (par_improvement > COLD_MOVE_OVERRIDE_GAIN * n_expert_scale))
    )
    if len(accepted) == 0:
        return False, np.array([], dtype=np.int64), _PREVIOUS_DEPLOYMENT.copy(), None

    deployment = _PREVIOUS_DEPLOYMENT.copy()
    deployment[accepted] = candidate[accepted]
    _PREVIOUS_DEPLOYMENT = deployment.copy()
    order = accepted[np.argsort(gain_per_moved_slot[accepted])[::-1]]
    return True, order.astype(np.int64), deployment, None