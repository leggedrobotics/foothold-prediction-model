# Foothold Prediction Model (FPM)

Training and evaluation code for perceptive humanoid locomotion with a foothold prediction model on the Unitree G1
in Isaac Lab. The FPM predicts the next two footholds from proprioception and the velocity command, and the policy
observes the elevation map only in 7 x 7 patches around these footholds. The FPM and the policy are trained together.

## Installation

Requires Isaac Sim 5.1, Isaac Lab 2.3.0 and `rsl-rl-lib==3.3.0`. In the Isaac Lab Python environment:

```bash
pip install -e .
```

## Structure

```
fpm/
  foot_pred/    FPM model, foothold labels, refitting and training schedule
  mdp/          foothold-gated height scan, rewards, events, commands, symmetry augmentation
  terrains/     training, locomotion validation and corridor validation terrains
  env_cfg.py    training, play and validation environments
  agents/       PPO configuration
  assets/       Unitree G1 (29 DoF) USD
scripts/
  train.py
  play.py
  validate_locomotion.py
  validate_corridor.py
  validate_foot_pred.py
```

## Training

```bash
python scripts/train.py --headless --seed 1
```

For the first 500 iterations the footholds are fixed under the base. After that, the FPM is retrained from scratch
every 500 iterations on 5 x 10^5 transitions of the current policy. FPM checkpoints are saved next to the policy
checkpoints as `foot_model_<iteration>.pt` (TorchScript, input normalization included). To resume a run, pass
`--resume <run>/model_<iteration>.pt`.

## Playback and export

```bash
python scripts/play.py --checkpoint logs/rsl_rl/g1_fpm/<run>/model_40000.pt --show_footholds
```

Writes `policy.pt`, `policy.onnx` and `foot_model.onnx` to `<run>/exported/`. The FPM takes the 70-dimensional
system state and returns two planar footholds in the base frame. Use `--joystick` for gamepad control and `--video`
to record a clip.

## Evaluation

Each script loads the policy checkpoint and the latest `foot_model_*.pt` from the same directory. Set
`PYTHONHASHSEED=0` so that the corridor terrain is identical across runs.

```bash
export PYTHONHASHSEED=0
python scripts/validate_locomotion.py --headless --checkpoint logs/rsl_rl/g1_fpm/<run>/model_40000.pt
python scripts/validate_corridor.py --headless --checkpoint logs/rsl_rl/g1_fpm/<run>/model_40000.pt
python scripts/validate_foot_pred.py --headless --checkpoint logs/rsl_rl/g1_fpm/<run>/model_40000.pt
```

Add `--save_results` to save the per-episode results in the run directory.
