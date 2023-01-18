from gymnasium.envs.registration import register


register(
    id='hp_model_3d-v0',
    entry_point='HPModel3D_env.envs:HPModel3D',
    max_episode_steps=2000,
)
