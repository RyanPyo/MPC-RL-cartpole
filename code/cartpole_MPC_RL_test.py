import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC

from cartpole_MPC_RL_env import CartPoleMPCResidualEnv


# ===========================================================
#   Utility: Create results directory relative to script
# ===========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(SAVE_DIR, exist_ok=True)


# ===========================================================
#   Run one episode and record trajectory
# ===========================================================
def run_episode(env, model=None, use_rl=True, T=5.0):
    obs, info = env.reset()
    dt = env.dt
    max_steps = int(T / dt)

    history_x = []
    history_theta = []

    for _ in range(max_steps):
        if use_rl and model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.array([0.0], dtype=np.float32)

        obs, reward, terminated, truncated, info_step = env.step(action)

        x, x_dot, theta, theta_dot = obs
        history_x.append(x)
        history_theta.append(theta)

        if terminated or truncated:
            return False, np.array(history_x), np.array(history_theta)

    return True, np.array(history_x), np.array(history_theta)


# ===========================================================
#   Experiment A: Mass Only (No Push)
# ===========================================================
def experiment_A(model):
    print("\n=== Experiment A: Mass variation, no push ===")

    masses = [0.05, 0.10, 0.20, 1.00]
    T = 5.0
    n_episodes = 1   # trajectory는 1개만 그려도 됨

    env = CartPoleMPCResidualEnv(
        mass_nominal=0.1,
        domain_randomize_mass=False,
        train_random_push=False,
        max_residual=30.0,
    )

    # 그래프 초기화
    fig_x, ax_x = plt.subplots(figsize=(8,4))
    fig_t, ax_t = plt.subplots(figsize=(8,4))

    for m in masses:
        env.set_mass_true(m)
        env.push_time = None

        # ----- MPC only -----
        ok_mpc, hx_mpc, ht_mpc = run_episode(env, model=None, use_rl=False, T=T)

        # ----- MPC + RL -----
        ok_rl, hx_rl, ht_rl = run_episode(env, model=model, use_rl=True, T=T)

        print(f"mass={m:.2f} | MPC success={ok_mpc} | MPC+RL success={ok_rl}")

        t_axis_mpc = np.arange(len(hx_mpc)) * env.dt
        t_axis_rl  = np.arange(len(hx_rl)) * env.dt

        # x position
        ax_x.plot(t_axis_mpc, hx_mpc, label=f"MPC mass={m}", linestyle='-')
        ax_x.plot(t_axis_rl,  hx_rl,  label=f"RL mass={m}", linestyle='--')

        # theta
        ax_t.plot(t_axis_mpc, ht_mpc, label=f"MPC mass={m}", linestyle='-')
        ax_t.plot(t_axis_rl,  ht_rl,  label=f"RL mass={m}", linestyle='--')

    ax_x.set_title("Experiment A - X Position")
    ax_x.set_xlabel("Time [s]")
    ax_x.set_ylabel("x [m]")
    ax_x.grid(True)
    ax_x.legend()

    ax_t.set_title("Experiment A - Theta")
    ax_t.set_xlabel("Time [s]")
    ax_t.set_ylabel("theta [rad]")
    ax_t.grid(True)
    ax_t.legend()

    fig_x.savefig(os.path.join(SAVE_DIR, "expA_x.png"))
    fig_t.savefig(os.path.join(SAVE_DIR, "expA_theta.png"))
    plt.close(fig_x)
    plt.close(fig_t)


# ===========================================================
#   Experiment B: Nominal Mass + Push
# ===========================================================
def experiment_B(model):
    print("\n=== Experiment B: Nominal mass push ===")

    mass = 0.10
    T = 5.0

    env = CartPoleMPCResidualEnv(
        mass_nominal=0.1,
        domain_randomize_mass=False,
        train_random_push=False,
        max_residual=20.0,
    )
    env.set_mass_true(mass)

    env.push_time = 1.0
    env.push_theta_dot = 0.5
    env.push_x_dot = 0.5

    # ---- MPC only ----
    ok_mpc, hx_mpc, ht_mpc = run_episode(env, model=None, use_rl=False, T=T)

    # ---- MPC + RL ----
    ok_rl, hx_rl, ht_rl = run_episode(env, model=model, use_rl=True, T=T)

    print(f"MPC success={ok_mpc} | MPC+RL success={ok_rl}")

    # Plot
    t_mpc = np.arange(len(hx_mpc)) * env.dt
    t_rl  = np.arange(len(hx_rl)) * env.dt

    fig_x, ax_x = plt.subplots(figsize=(8,4))
    fig_t, ax_t = plt.subplots(figsize=(8,4))

    ax_x.plot(t_mpc, hx_mpc, linestyle='-', label="MPC")
    ax_x.plot(t_rl,  hx_rl,  linestyle='--', label="MPC+RL")
    ax_x.set_title("Experiment B - X Position")
    ax_x.grid(); ax_x.legend()

    ax_t.plot(t_mpc, ht_mpc, linestyle='-', label="MPC")
    ax_t.plot(t_rl,  ht_rl,  linestyle='--', label="MPC+RL")
    ax_t.set_title("Experiment B - Theta")
    ax_t.grid(); ax_t.legend()

    fig_x.savefig(os.path.join(SAVE_DIR, "expB_x.png"))
    fig_t.savefig(os.path.join(SAVE_DIR, "expB_theta.png"))

    plt.close(fig_x)
    plt.close(fig_t)


# ===========================================================
#   Experiment C: Mass mismatch + push
# ===========================================================
def experiment_C(model):
    print("\n=== Experiment C: Mass mismatch + push ===")

    masses = [0.05, 0.1, 0.20, 1.00]
    T = 5.0

    env = CartPoleMPCResidualEnv(
        mass_nominal=0.1,
        domain_randomize_mass=False,
        train_random_push=False,
        max_residual=20.0,
    )

    fig_x, ax_x = plt.subplots(figsize=(8,4))
    fig_t, ax_t = plt.subplots(figsize=(8,4))

    for m in masses:
        env.set_mass_true(m)

        env.push_time = 1.0
        env.push_theta_dot = 0.5
        env.push_x_dot = 0.5

        ok_mpc, hx_mpc, ht_mpc = run_episode(env, model=None, use_rl=False, T=T)
        ok_rl, hx_rl, ht_rl = run_episode(env, model=model, use_rl=True, T=T)

        print(f"mass={m:.2f} | MPC success={ok_mpc} | RL success={ok_rl}")

        # time axis
        t_mpc = np.arange(len(hx_mpc)) * env.dt
        t_rl  = np.arange(len(hx_rl)) * env.dt

        ax_x.plot(t_mpc, hx_mpc, linestyle='-', label=f"MPC mass={m}")
        ax_x.plot(t_rl,  hx_rl,  linestyle='--', label=f"RL mass={m}")

        ax_t.plot(t_mpc, ht_mpc, linestyle='-', label=f"MPC mass={m}")
        ax_t.plot(t_rl,  ht_rl,  linestyle='--', label=f"RL mass={m}")

    ax_x.set_title("Experiment C - X Position")
    ax_x.set_xlabel("Time [s]")
    ax_x.set_ylabel("x [m]")
    ax_x.grid(True)
    ax_x.legend()

    ax_t.set_title("Experiment C - Theta")
    ax_t.set_xlabel("Time [s]")
    ax_t.set_ylabel("theta [rad]")
    ax_t.grid(True)
    ax_t.legend()

    fig_x.savefig(os.path.join(SAVE_DIR, "expC_x.png"))
    fig_t.savefig(os.path.join(SAVE_DIR, "expC_theta.png"))
    plt.close(fig_x)
    plt.close(fig_t)


# ===========================================================
#   Main
# ===========================================================
def main():
    model = PPO.load("ppo_cartpole_mpc_residual")

    experiment_A(model)
    experiment_B(model)
    experiment_C(model)

    print(f"\nGraphs saved to: {SAVE_DIR}\n")


if __name__ == "__main__":
    main()
