from gymnasium.envs.registration import register


def register_envs():
    register(
        id="cheetah-sigmap-v0",
        entry_point="sigmap.drl.envs.cheetah:HalfCheetahEnv",
        max_episode_steps=1000,
    )
    register(
        id="obstacles-sigmap-v0",
        entry_point="sigmap.drl.envs.obstacles:Obstacles",
        max_episode_steps=500,
    )
    register(
        id="reacher-sigmap-v0",
        entry_point="sigmap.drl.envs.reacher:Reacher7DOFEnv",
        max_episode_steps=500,
    )
    register(
        id="wireless-sigmap-v0",
        entry_point="sigmap.drl.envs.wireless:WirelessEnv",
        max_episode_steps=1000,
    )
