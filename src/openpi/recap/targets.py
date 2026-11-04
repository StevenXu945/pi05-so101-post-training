"""Paper-style rewards and advantages; iteration scales use explicit task horizons."""
import numpy as np


def fit_scales(episodes, failure_cost_multiplier=1.0):
    if not np.isfinite(failure_cost_multiplier) or failure_cost_multiplier <= 0:
        raise ValueError("failure_cost_multiplier must be positive")
    longest = {}
    for task, length in episodes:
        if length < 1:
            raise ValueError("Episodes must be nonempty")
        longest[task] = max(longest.get(task, 0), int(length))
    if not longest:
        raise ValueError("Cannot fit scales without training episodes")
    return {task: {"max_length": n, "failure_cost": n * failure_cost_multiplier,
                   "scale": max(1.0, n - 1 + n * failure_cost_multiplier)} for task, n in longest.items()}


def rewards_and_returns(length, outcome, scale):
    if length < 1 or outcome not in {"success", "failure"}:
        raise ValueError("Need a nonempty explicitly labeled trajectory")
    denominator = float(scale["scale"])
    failure_cost = float(scale["failure_cost"])
    if "max_length" in scale and length > scale["max_length"]:
        raise ValueError("Episode exceeds saved task max_length; set a task horizon before fitting or use a larger training scale")
    if denominator <= 0 or failure_cost <= 0 or not np.isfinite([denominator, failure_cost]).all():
        raise ValueError("Invalid saved task scale")
    raw = -np.ones(length, np.float32)
    raw[-1] = 0 if outcome == "success" else -failure_cost
    reward = raw / denominator
    value = np.cumsum(reward[::-1], dtype=np.float64)[::-1].astype(np.float32)
    if value.min() < -1 - 1e-6:
        raise ValueError("Episode exceeds saved training scale; do not refit or silently clip")
    return raw, reward, value


def n_step_advantage(rewards, values, lookahead=50):
    reward, value = np.asarray(rewards, np.float64), np.asarray(values, np.float64)
    if lookahead < 1 or reward.ndim != 1 or value.shape != reward.shape or not len(reward):
        raise ValueError("Expected equal nonempty reward/value vectors and positive horizon")
    if not np.isfinite(reward).all() or not np.isfinite(value).all():
        raise ValueError("Nonfinite reward/value")
    prefix = np.r_[0.0, np.cumsum(reward)]
    positions = np.arange(len(reward))
    ends = np.minimum(positions + lookahead, len(reward))
    future = np.zeros(len(value))
    valid = ends < len(value)
    future[valid] = value[ends[valid]]
    return (prefix[ends] - prefix[positions] + future - value).astype(np.float32)


def fit_thresholds(advantages, task_ids, fit_mask, positive_ratio=0.4):
    """Fit per-task thresholds on an explicit calibration subset only."""
    advantage, tasks, mask = np.asarray(advantages), np.asarray(task_ids), np.asarray(fit_mask)
    if advantage.ndim != 1 or tasks.shape != advantage.shape or mask.shape != advantage.shape:
        raise ValueError("Invalid threshold arrays")
    if mask.dtype != np.bool_ or not 0 < positive_ratio <= 1 or not np.isfinite(advantage).all():
        raise ValueError("Invalid threshold mask/ratio/advantages")
    thresholds = {}
    for task in np.unique(tasks):
        selected = mask & (tasks == task)
        if not selected.any():
            raise ValueError(f"No eligible current-round autonomous training rollouts for task {task}")
        thresholds[str(task)] = float(np.quantile(advantage[selected], 1 - positive_ratio))
    return thresholds


def advantage_labels(advantages, interventions, task_ids, positive_ratio=0.4, thresholds=None):
    advantage = np.asarray(advantages)
    human = np.asarray(interventions)
    task_ids = np.asarray(task_ids)
    if not 0 < positive_ratio <= 1 or advantage.shape != human.shape or advantage.shape != task_ids.shape:
        raise ValueError("Invalid indicator arrays or positive ratio")
    if not np.isfinite(advantage).all() or not np.isin(human, [0, 1]).all():
        raise ValueError("Invalid advantages/interventions")
    result = np.full(advantage.shape, "positive", dtype="<U8")
    fitted = {} if thresholds is None else dict(thresholds)
    for task in np.unique(task_ids):
        mask = (task_ids == task) & (human == 0)
        if not mask.any():
            continue
        key = str(task)
        if thresholds is None:
            fitted[key] = float(np.quantile(advantage[mask], 1 - positive_ratio))
        if key not in fitted or not np.isfinite(fitted[key]):
            raise ValueError(f"Missing threshold for {key}")
        result[mask] = np.where(advantage[mask] >= fitted[key], "positive", "negative")
    return result, fitted
