"""Reproducible train/evaluate experiment; outputs CSV and capacity errors."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from ml_models import (train_capacity_forecaster, train_reward_model,
                       CapacityPredictor, evaluate_policy)
from simulator import make_trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results")
    parser.add_argument("--test-seeds", type=int, default=20)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model = train_capacity_forecaster()
    pred = CapacityPredictor(model)
    reward_model = train_reward_model(pred)
    traces = [make_trace(seed, steps=180) for seed in range(1000, 1000 + args.test_seeds)]
    actual = np.concatenate([t["capacity"][1:] for t in traces])
    previous = np.concatenate([t["capacity"][:-1] for t in traces])
    forecast = model.predict(previous.reshape(-1, 1))
    forecast_metrics = {"ridge_mae_mbps": float(np.mean(np.abs(actual - forecast))),
                        "persistence_mae_mbps": float(np.mean(np.abs(actual - previous)))}
    policies = ["static", "demand_proportional", "learned"]
    summary = {p: evaluate_policy(p, traces, pred, reward_model) for p in policies}
    with (out / "policy_metrics.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "metric", "mean", "std_across_seeds"])
        for p in policies:
            for metric, (mean, std) in summary[p].items():
                w.writerow([p, metric, f"{mean:.6f}", f"{std:.6f}"])
    (out / "capacity_prediction.json").write_text(json.dumps(forecast_metrics, indent=2))
    (out / "experiment_config.json").write_text(json.dumps({
        "training_capacity_seeds": "100-129", "training_reward_seeds": "200-229",
        "held_out_test_seeds": f"1000-{999 + args.test_seeds}",
        "policies": policies, "segment_seconds": 2, "test_trace_steps": 180}, indent=2))
    print("One-step capacity forecast MAE (Mbps):", forecast_metrics)
    print("\nMean over held-out seeds (standard deviation in parentheses)")
    for p in policies:
        print(p, " | ".join(f"{k}={v[0]:.3f} ({v[1]:.3f})"
                            for k, v in summary[p].items() if k in
                            ["reward", "qoe", "stall_s", "critical_delivery"]))
    print(f"\nSaved CSV and experiment metadata to {out}")


if __name__ == "__main__":
    main()
