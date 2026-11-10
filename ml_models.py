"""Two ML tasks: one-step capacity forecasting and context-aware slice control."""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from simulator import ACTIONS, State, features, make_trace, step, initial_state


def train_capacity_forecaster(seeds=range(100, 130)):
    X, y = [], []
    for seed in seeds:
        c = make_trace(seed)["capacity"]
        X.extend([[c[t - 1]] for t in range(1, len(c))])
        y.extend(c[1:])
    return Ridge(alpha=15.).fit(np.asarray(X), np.asarray(y))


class CapacityPredictor:
    def __init__(self, model):
        self.model = model

    def __call__(self, previous_capacity):
        return max(1., float(self.model.predict([[previous_capacity]])[0]))


def train_reward_model(predictor, seeds=range(200, 230)):
    """Fit a contextual reward model to one-step counterfactual simulator rewards.

    Exhaustively evaluate each permissible slice action at synthetic training
    states, then train a supervised approximation of reward(state, action).
    This is simulation-trained policy optimization, not online RL or an LLM.
    """
    X, y = [], []
    rng = np.random.default_rng(41)
    for seed in seeds:
        trace = make_trace(seed, steps=100)
        state = initial_state(trace)
        for t in range(1, 100):
            offered = trace["demand"][t]
            context = features(state.previous_capacity, offered, state.buffer_s, state.previous_rate)
            for action in ACTIONS:
                _, info = step(state, offered, trace["capacity"][t], action, predictor)
                X.append(np.r_[context, action])
                y.append(info["reward"])
            # Independent behavior policy generates varied states.
            action = ACTIONS[rng.integers(len(ACTIONS))]
            state, _ = step(state, offered, trace["capacity"][t], action, predictor)
    model = ExtraTreesRegressor(n_estimators=80, min_samples_leaf=3,
                                max_features=1.0, random_state=41, n_jobs=1)
    return model.fit(np.asarray(X), np.asarray(y))


def choose_action(policy, state, offered, reward_model=None):
    if policy == "static":
        return ACTIONS[0]
    if policy == "demand_proportional":
        # Candidate nearest the offered-load proportions.
        target = offered / offered.sum()
        return ACTIONS[np.argmin(np.square(ACTIONS - target).sum(axis=1))]
    if policy == "learned":
        if reward_model is None:
            raise ValueError("Learned policy requires trained reward model")
        ctx = features(state.previous_capacity, offered, state.buffer_s, state.previous_rate)
        X = np.asarray([np.r_[ctx, a] for a in ACTIONS])
        return ACTIONS[np.argmax(reward_model.predict(X))]
    raise ValueError(f"Unknown policy: {policy}")


def evaluate_policy(policy, traces, predictor, reward_model=None):
    metrics = []
    for trace in traces:
        state = initial_state(trace)
        outcomes = []
        for t in range(1, len(trace["capacity"])):
            offered = trace["demand"][t]
            action = choose_action(policy, state, offered, reward_model)
            state, info = step(state, offered, trace["capacity"][t], action, predictor)
            outcomes.append(info)
        metrics.append({key: float(np.mean([x[key] for x in outcomes]))
                        for key in outcomes[0]})
    return {k: (float(np.mean([m[k] for m in metrics])),
                float(np.std([m[k] for m in metrics], ddof=1)))
            for k in metrics[0]}
