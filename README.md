# Watch Your Step: Anticipated Perception for Humanoid Locomotion

<p align="center">
  <a href="https://leggedrobotics.github.io/watch-your-step/"><b>Project page</b></a> &nbsp;·&nbsp;
  <a href="https://youtu.be/3bLU6_OPv6s"><b>Video</b></a> &nbsp;·&nbsp;
  <a href="https://arxiv.org/abs/2610.29612"><b>arXiv</b></a> &nbsp;·&nbsp;
  <a href="#citation"><b>Citation</b></a>
</p>

<p align="center">
  Matthias Heyrman, Chenhao Li, Ioannis Dadiotis, Marco Hutter<br>
  Robotic Systems Lab and ETH AI Center, ETH Zurich
</p>

<p align="center">
  <a href="https://leggedrobotics.github.io/watch-your-step/"><img src="https://leggedrobotics.github.io/watch-your-step/assets/readme/blocks.webp" alt="Blocks, with the two patches the policy reads" width="32%"></a>
  <a href="https://leggedrobotics.github.io/watch-your-step/"><img src="https://leggedrobotics.github.io/watch-your-step/assets/readme/gaps.webp" alt="Gaps" width="32%"></a>
  <a href="https://leggedrobotics.github.io/watch-your-step/"><img src="https://leggedrobotics.github.io/watch-your-step/assets/readme/drops.webp" alt="Drops" width="32%"></a>
  <br>
  <a href="https://leggedrobotics.github.io/watch-your-step/"><img src="https://leggedrobotics.github.io/watch-your-step/assets/readme/course.webp" alt="Obstacle course" width="32%"></a>
  <a href="https://leggedrobotics.github.io/watch-your-step/"><img src="https://leggedrobotics.github.io/watch-your-step/assets/readme/corridor.webp" alt="Walled corridor in simulation" width="32%"></a>
  <a href="https://leggedrobotics.github.io/watch-your-step/"><img src="https://leggedrobotics.github.io/watch-your-step/assets/readme/robot_view.webp" alt="The robot's onboard elevation map on stairs" width="32%"></a>
</p>

Training and evaluation code for perceptive humanoid locomotion with a foothold prediction model (FPM) on the Unitree G1 in Isaac Lab. The FPM predicts the next two footholds from proprioception and the velocity command, and the policy observes the elevation map only in 7 x 7 patches around these footholds. The FPM and the policy are trained together. The policy transfers zero-shot to the real robot; see the [project page](https://leggedrobotics.github.io/watch-your-step/) for the hardware experiments and a version of the policy running in your browser.

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
demo/           pretrained policy and FPM
```

## Training

```bash
python scripts/train.py --headless --seed 1
```

For the first 500 iterations the footholds are fixed under the base. After that, the FPM is retrained from scratch every 500 iterations on 5 x 10^5 transitions of the current policy. FPM checkpoints are saved next to the policy checkpoints as `foot_model_<iteration>.pt` (TorchScript, input normalization included). To resume a run, pass `--resume <run>/model_<iteration>.pt`.

## Playback and export

```bash
python scripts/play.py --show_footholds
```

Without `--checkpoint`, the latest checkpoint in `logs/rsl_rl/g1_fpm/` is played, or the pretrained policy in `demo/` if there are no logs. Writes `policy.pt`, `policy.onnx` and `foot_model.onnx` to `exported/` next to the checkpoint. The FPM takes the 70-dimensional system state and returns two planar footholds in the base frame. Use `--joystick` for gamepad control and `--video` to record a clip.

## Evaluation

Each script loads the policy checkpoint and the latest `foot_model_*.pt` from the same directory. Set `PYTHONHASHSEED=0` so that the corridor terrain is identical across runs.

```bash
export PYTHONHASHSEED=0
python scripts/validate_locomotion.py --headless --checkpoint logs/rsl_rl/g1_fpm/<run>/model_40000.pt
python scripts/validate_corridor.py --headless --checkpoint logs/rsl_rl/g1_fpm/<run>/model_40000.pt
python scripts/validate_foot_pred.py --headless --checkpoint logs/rsl_rl/g1_fpm/<run>/model_40000.pt
```

Add `--save_results` to save the per-episode results in the run directory.

## Citation

```bibtex
@misc{heyrman2026watchstepanticipatedperception,
  title         = {Watch Your Step: Anticipated Perception for Humanoid Locomotion},
  author        = {Matthias Heyrman and Chenhao Li and Ioannis Dadiotis and Marco Hutter},
  year          = {2026},
  eprint        = {2610.29612},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2610.29612}
}
```

## License

© 2026 ETH Zurich<br>
Created by: Matthias Heyrman, Chenhao Li, Ioannis Dadiotis, and Marco Hutter<br>
Licensed under the MIT License (see [LICENSE](LICENSE)).

Third-party files keep their original license (see [licenses/dependencies](licenses/dependencies)):

- `fpm/assets/data/G1/` is the Unitree G1 (29 DoF, rev 1.0) model from [unitree_ros](https://github.com/unitreerobotics/unitree_ros), converted to USD, BSD-3-Clause.
