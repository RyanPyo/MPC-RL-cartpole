from stable_baselines3 import PPO, SAC
from cartpole_MPC_RL_env import CartPoleMPCResidualEnv
import torch


def main():
    algorithm = "PPO"

    env = CartPoleMPCResidualEnv(
        mass_nominal=0.1,
        mass_min=0.05,
        mass_max=1.0,
        domain_randomize_mass=True,   # 질량 랜덤
        train_random_push=True,       # 학습 중 랜덤 push
        max_residual=20.0,
    )

    if algorithm == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            tensorboard_log="./tb_log_mpc_rl/",
        )
    elif algorithm == "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            tensorboard_log="./tb_log_mpc_rl/",
        )

    print("Using device:", model.device)

    model.learn(total_timesteps=100_000)
    # Used 600,000 for PPO and 100,000 for SAC

    if algorithm =="PPO":
        model.save("ppo_cartpole_mpc_residual")
    elif algorithm == "SAC":
        model.save("ppo_cartpole_mpc_residual")
    env.close()


if __name__ == "__main__":
    main()
