"""Synthetic multi-slice rich-media delivery simulator (educational research sample).

All traffic units are Mbps and each step is a 2-second segment interval.
No simulator is connected to a deployed 5G stack; assumptions are explicit.
"""
from dataclasses import dataclass
import numpy as np

SLICE_NAMES = ("immersive_360", "video", "critical")
ACTIONS = np.array([
    [.50, .35, .15], [.65, .20, .15], [.35, .50, .15],
    [.40, .30, .30], [.55, .25, .20], [.30, .40, .30],
], dtype=float)
# Critical traffic has a 15% reservation and may use additional unused bandwidth.
MIN_SHARE = np.array([.20, .20, .15])
SEGMENT_S = 2.0
RATE_LADDERS = (np.array([1., 2., 4., 6., 8.]), np.array([.5, 1., 2., 3., 5.]))


@dataclass
class State:
    buffer_s: np.ndarray
    previous_rate: np.ndarray
    previous_capacity: float


def make_trace(seed: int, steps: int = 180) -> dict:
    """Nonstationary bandwidth, demand, and a critical-network slice."""
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi)
    capacity = np.empty(steps)
    capacity[0] = rng.uniform(10, 20)
    for t in range(1, steps):
        target = 15 + 5 * np.sin(t / 23 + phase) + 2.5 * np.sin(t / 8)
        if steps // 3 <= t < steps // 3 + 18:
            target = 2.2  # deliberately severe temporary link degradation
        capacity[t] = np.clip(.65 * capacity[t - 1] + .35 * target + rng.normal(0, 1.5), 1.3, 27)
    # Demand refers to Mbps of requested data; 360-degree video higher demand.
    demand = np.column_stack([
        np.clip(rng.normal(8, 1.8, steps), 3, 13),
        np.clip(rng.normal(5.5, 1.4, steps), 2, 10),
        np.clip(rng.normal(1.7, .45, steps), .6, 3),
    ])
    return {"capacity": capacity, "demand": demand}


def features(capacity_prev: float, demands: np.ndarray, buffers: np.ndarray,
             previous_rates: np.ndarray) -> np.ndarray:
    """Only information observable before the current allocation decision."""
    return np.r_[capacity_prev, demands, buffers, previous_rates]


def allocate(capacity: float, demands: np.ndarray, shares: np.ndarray) -> np.ndarray:
    """Reservation-first scheduler with work conservation and per-slice caps.

    The target shares reserve up to capacity*share[i] for each slice. Any leftover
    is redistributed to slices with unmet demand. No slice is allocated above its
    current offered demand; unused reserved capacity can be borrowed.
    """
    shares = np.asarray(shares, dtype=float)
    if np.any(shares < MIN_SHARE - 1e-9) or not np.isclose(shares.sum(), 1.):
        raise ValueError("Shares must sum to one and honor minimum reservations")
    allocation = np.minimum(demands, capacity * shares)
    leftover = max(0., capacity - allocation.sum())
    for _ in range(8):
        unmet = np.maximum(0., demands - allocation)
        if leftover < 1e-8 or unmet.sum() < 1e-8:
            break
        increment = np.minimum(unmet, leftover * unmet / unmet.sum())
        allocation += increment
        leftover = max(0., capacity - allocation.sum())
    return allocation


def abr_rate(estimated_mbps: float, ladder: np.ndarray) -> float:
    """Pick highest representation <= 85% of predicted rate (conservative ABR)."""
    feasible = ladder[ladder <= .85 * max(0., estimated_mbps)]
    return float(feasible[-1] if len(feasible) else ladder[0])


def transition(state: State, offered: np.ndarray, delivered: np.ndarray,
               predicted_mbps: np.ndarray, actual_capacity: float):
    """Advance video buffers and compute surrogate QoE and critical delivery.

    Video segments have duration SEGMENT_S; required downloading time is
    (selected bitrate / delivered throughput) * segment duration. The buffer
    evolves by subtracting download time, adding segment duration, and capping
    at 12 seconds. Rebuffering is download time beyond pre-download buffer.
    """
    new_buf = state.buffer_s.copy()
    new_rates = state.previous_rate.copy()
    quality = np.zeros(2)
    stalls = np.zeros(2)
    switches = np.zeros(2)
    for i in range(2):
        selected = abr_rate(predicted_mbps[i], RATE_LADDERS[i])
        download_time = SEGMENT_S * selected / max(delivered[i], .05)
        stalls[i] = max(0., download_time - new_buf[i])
        new_buf[i] = min(12., max(0., new_buf[i] - download_time) + SEGMENT_S)
        quality[i] = np.log1p(selected) / np.log1p(RATE_LADDERS[i][-1])
        switches[i] = abs(selected - new_rates[i]) / RATE_LADDERS[i][-1]
        new_rates[i] = selected
    critical_delivery = min(1., delivered[2] / max(offered[2], 1e-6))
    # Explicitly subjective and illustrative QoE proxy, not VMAF/PCQM.
    qoe = float((quality[0] + quality[1]) / 2
                - .17 * stalls.mean() - .10 * switches.mean())
    reward = qoe + .55 * critical_delivery - .15 * stalls.max()
    info = {"qoe": qoe, "stall_s": float(stalls.mean()),
            "critical_delivery": critical_delivery,
            "quality": float(quality.mean()), "switching": float(switches.mean()),
            "reward": reward}
    return State(new_buf, new_rates, float(actual_capacity)), info


def step(state: State, offered: np.ndarray, capacity: float,
         shares: np.ndarray, predictor=None):
    delivered = allocate(capacity, offered, shares)
    # Prediction does not look at current actual radio capacity.
    # If a learned predictor is supplied, it forecasts total next-step capacity.
    forecast_capacity = float(predictor(state.previous_capacity) if predictor else state.previous_capacity)
    predicted = allocate(max(1., forecast_capacity), offered, shares)
    return transition(state, offered, delivered, predicted, capacity)


def initial_state(trace: dict) -> State:
    return State(np.array([4., 4.]), np.array([1., .5]), float(trace["capacity"][0]))
