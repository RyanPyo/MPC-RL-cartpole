import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, SAC
from cartpole_MPC_RL_env import CartPoleMPCResidualEnv  # 너가 쓰는 env 이름에 맞게 수정


# ===========================================================
#   Utility: 결과 저장 폴더 (스크립트 기준 상대경로)
# ===========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "results_PPO_SAC_ExpC")
os.makedirs(SAVE_DIR, exist_ok=True)


# ===========================================================
#   한 에피소드 실행하면서 x, theta trajectory 기록
# ===========================================================
def run_episode(env, model, T=5.0):
    """
    Env 하나에서 주어진 model(PPO 또는 SAC)로
    T초 동안 에피소드를 진행하고,
    - 성공 여부 (넘어지지 않았는지)
    - x(t), theta(t) trajectory
    를 반환.
    """
    obs, info = env.reset()
    dt = env.dt
    max_steps = int(T / dt)

    history_x = []
    history_theta = []

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info_step = env.step(action)

        x, x_dot, theta, theta_dot = obs
        history_x.append(x)
        history_theta.append(theta)

        if terminated or truncated:
            return False, np.array(history_x), np.array(history_theta)

    return True, np.array(history_x), np.array(history_theta)


# ===========================================================
#   Experiment C: Mass mismatch + Push
#   - MPC+PPO vs MPC+SAC 비교
# ===========================================================
def experiment_C_compare_PPO_SAC(model_ppo, model_sac):
    print("\n=== Experiment C: Mass mismatch + push (PPO vs SAC) ===")

    # 실험 C에서 사용하는 질량들
    masses = [0.05, 0.1, 0.20, 1.00]
    T = 5.0

    env = CartPoleMPCResidualEnv(
        mass_nominal=0.1,
        domain_randomize_mass=False,
        train_random_push=False,
        max_residual=20.0,  # 네 학습 설정에 맞게 맞추면 됨
    )

    # 전체 mass 결과를 한 그림에 보기 위한 figure
    fig_x, ax_x = plt.subplots(figsize=(8, 4))
    fig_t, ax_t = plt.subplots(figsize=(8, 4))

    for m in masses:
        env.set_mass_true(m)

        # push disturbance 설정 (실험 C 셋업과 맞게)
        env.push_time = 1.0
        env.push_theta_dot = 0.5
        env.push_x_dot = 0.5

        # ----- MPC + PPO -----
        ok_ppo, hx_ppo, ht_ppo = run_episode(env, model_ppo, T=T)

        # ----- MPC + SAC -----
        # env를 다시 reset하기 때문에 같은 조건으로 실행됨
        env.set_mass_true(m)
        env.push_time = 1.0
        env.push_theta_dot = 0.5
        env.push_x_dot = 0.5
        ok_sac, hx_sac, ht_sac = run_episode(env, model_sac, T=T)

        print(f"mass={m:.2f} | PPO success={ok_ppo} | SAC success={ok_sac}")

        # 시간 축
        t_ppo = np.arange(len(hx_ppo)) * env.dt
        t_sac = np.arange(len(hx_sac)) * env.dt

        # ---- x position plot ----
        ax_x.plot(
            t_ppo,
            hx_ppo,
            linestyle='-',
            label=f"PPO m={m:.2f}"
        )
        ax_x.plot(
            t_sac,
            hx_sac,
            linestyle='--',
            label=f"SAC m={m:.2f}"
        )

        # ---- theta plot ----
        ax_t.plot(
            t_ppo,
            ht_ppo,
            linestyle='-',
            label=f"PPO m={m:.2f}"
        )
        ax_t.plot(
            t_sac,
            ht_sac,
            linestyle='--',
            label=f"SAC m={m:.2f}"
        )

    # ----- x figure 설정 및 저장 -----
    ax_x.set_title("Experiment C - X Position (MPC+PPO vs MPC+SAC)")
    ax_x.set_xlabel("Time [s]")
    ax_x.set_ylabel("x [m]")
    ax_x.grid(True)
    ax_x.legend(fontsize=8, ncol=2)
    fig_x.tight_layout()
    fig_x.savefig(os.path.join(SAVE_DIR, "expC_PPO_vs_SAC_x.png"))

    # ----- theta figure 설정 및 저장 -----
    ax_t.set_title("Experiment C - Theta (MPC+PPO vs MPC+SAC)")
    ax_t.set_xlabel("Time [s]")
    ax_t.set_ylabel("theta [rad]")
    ax_t.grid(True)
    ax_t.legend(fontsize=8, ncol=2)
    fig_t.tight_layout()
    fig_t.savefig(os.path.join(SAVE_DIR, "expC_PPO_vs_SAC_theta.png"))

    plt.close(fig_x)
    plt.close(fig_t)

    print(f"\n[Saved] {SAVE_DIR}/expC_PPO_vs_SAC_x.png")
    print(f"[Saved] {SAVE_DIR}/expC_PPO_vs_SAC_theta.png\n")


# ===========================================================
#   Main
# ===========================================================
def main():
    # 너가 저장한 모델 이름에 맞춰서 수정하면 됨
    model_ppo = PPO.load("ppo_cartpole_mpc_residual")
    model_sac = SAC.load("sac_cartpole_mpc_residual")

    experiment_C_compare_PPO_SAC(model_ppo, model_sac)


if __name__ == "__main__":
    main()
