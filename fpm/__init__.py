import gymnasium as gym

_AGENT = f"{__name__}.agents.rsl_rl_ppo_cfg:FpmPPORunnerCfg"

for task_id, env_cfg in (
    ("FPM-G1-v0", "FpmEnvCfg"),
    ("FPM-G1-Play-v0", "FpmEnvCfg_PLAY"),
    ("FPM-G1-Validate-Locomotion-v0", "FpmLocomotionValidationEnvCfg"),
    ("FPM-G1-Validate-FootPred-v0", "FpmFootPredValidationEnvCfg"),
    ("FPM-G1-Validate-Corridor-v0", "FpmCorridorValidationEnvCfg"),
):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:{env_cfg}", "rsl_rl_cfg_entry_point": _AGENT},
    )
