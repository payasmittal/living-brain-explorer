# Living Brain Explorer

A three-factor learning rule using node perturbation with
reward-baseline subtraction, applied to a recurrent network.
Investigates how learning speed scales with hidden layer size.

## Finding

Trials to reach a 90% accuracy criterion vary non-monotonically
with hidden layer size, falling from ~3,400 at N=32 to ~1,600 at
N=128, then rising to ~6,000 at N=512. A 16-fold increase in
network size costs only a 2.6-fold increase in trials.

## Note on earlier versions

An earlier implementation contained two defects: activations were
clipped rather than squashed, causing output saturation, and the
neuromodulatory signal lacked baseline subtraction. Results from
that version are superseded.

## Files

- `research_evaluation.ipynb` — main notebook
- `sweep_*.npy` — scaling results per network size
- `rerun_128.npy` — N=128 with disjoint-seed selection
- `plateau_fastbrain.npy` — learning curves
- `*.png` — figures
- `requirements.txt` — dependencies
