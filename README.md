# ML-Based Network Slicing for Rich-Media Delivery

Self-contained research code sample for the Aalto University postdoctoral application.

This is a newly developed illustrative simulator, not the source code or an experimental reproduction of EDGE360, METAVV, or their published results. It demonstrates the design and evaluation of resource-allocation algorithms for a shared, variable-capacity edge link carrying immersive 360-degree media, conventional video, and a critical-communication slice.

## Two ML problems solved

1. One-step capacity prediction. Fit ridge regression to historical bandwidth observations from synthetic traces, then compare out-of-sample mean absolute prediction error (MAE, Mbps) against persistence (using the previous observation).
2. QoE-aware network-slice allocation. Simulate and label the one-step reward of feasible allocation actions, then fit an ExtraTrees regression model estimating `reward(context, action)`. At inference, pick the candidate action with highest predicted reward. Compare against static and demand-proportional allocation on held-out trace seeds. The method is a **contextual reward model with offline simulated labels**, *not* deep reinforcement learning, MLLM or an online-trained production policy.

## Environment and metrics

- Slices: immersive 360-degree video, standard video, and critical traffic.
- Channel: shared, time-varying capacity with a synthetic temporary degradation.
- Scheduler: minimum slice reservations; work-conserving reuse of unused reservations; never allocates beyond demand.
- Video adaptation: conservative bitrate selection from two discrete representation ladders, driven by predicted slice capacity.
- Buffer / stalls: 2-second segment duration, simple download-time buffer model, switching penalty.
- Outcomes: illustrative normalized log-bitrate quality; mean stall time per video segment; switching; critical traffic served fraction; overall surrogate QoE/reward.
- Evaluation: mean and standard deviation across independently seeded test traces, with separate training and test seed ranges.

Metric limitations: The QoE surrogate is *not* VMAF or PCQM. This simplified model omits packet-level routing, link-layer scheduling, slice admission, motion traces, viewport geometry, actual 5G stack behavior, and end-to-end latency guarantees. The critical-slice served fraction is not a verified public-safety reliability or latency metric. Outcomes are synthetic demonstration results; do not present them as measured deployed-network gains.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python run_experiment.py --output results --test-seeds 20
python -m pytest -q
```

Output files: `results/policy_metrics.csv`, `results/capacity_prediction.json`, and `results/experiment_config.json`.

## Study design and leakage control

Capacity regression trains on seeds 100–129. The contextual reward model trains on independently generated seeds 200–229. End-to-end results use held-out seeds 1000 onward. Model features include previous observed capacity, current demand, buffers, and prior selected rates; current true link capacity is unavailable at decision time. Reward labels in the training simulation can use true capacity, but test policy inference cannot. All policies use the same learned capacity predictor and ABR logic; they vary only in slice allocation.

## Files

- `simulator.py`: traffic generator, demand caps, slice scheduler, ABR and buffer/reward dynamics
- `ml_models.py`: ridge throughput predictor, contextual reward estimator, baselines
- `run_experiment.py`: train, test, report summary statistics
- `tests/test_simulator.py`: deterministic and scheduling invariants

## Research extension toward multimodal networking

Represent an immersive session with observed scene complexity, viewport concentration, semantic urgency, and predicted task importance. Compare how semantic priorities modify allocation while respecting slice reservations and budget constraints; validate against real bandwidth and viewport traces. A multimodal model could generate task-level importance signals, but no such model is implemented here. This sample focuses on the experimentally inspectable network-control pipeline.

## Authorship and use

Generated as an original illustrative coding sample for the applicant; please review and understand the design and assumptions before submitting. Do not describe it as an archived repository of previously published research code.
