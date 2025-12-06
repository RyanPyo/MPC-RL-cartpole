import math
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pyMPC.mpc import MPCController

from calc_function import calc_AB  # 네가 만든 A,B 계산 함수


class CartPoleMPCResidualEnv(gym.Env):
    """
    CartPole + MPC + RL residual 환경

    상태: x = [x, x_dot, theta, theta_dot]
    입력: u_total = u_mpc + u_residual

    - 실제 dynamics: 비선형 CartPole
    - MPC 모델: nominal mass 기반 선형 모델 (Ad_nom, Bd_nom)
    - RL: residual force (연속 action ∈ [-1,1]) → [-max_residual, +max_residual]
    """

    metadata = {"render_modes": ["human"], "render_fps": 100}

    def __init__(
        self,
        render_mode=None,
        mass_nominal: float = 0.1,
        mass_min: float = 0.05,
        mass_max: float = 1.0,
        domain_randomize_mass: bool = True,   # 학습: True / 테스트: False
        train_random_push: bool = False,      # 학습 때 랜덤 push 쓸지 여부
        max_residual: float = 4.0,
    ):
        super().__init__()
        self.render_mode = render_mode

        # 물리 파라미터
        self.gravity = 9.8
        self.masscart = 1.0
        self.length = 0.5   # 절반 길이 (전체 1.0m)
        self.dt = 0.01
        self.kinematics_integrator = "euler"

        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.max_episode_steps = 1500

        # 질량 관련
        self.mass_nominal = mass_nominal
        self.mass_true = mass_nominal
        self.mass_min = mass_min
        self.mass_max = mass_max

        self.domain_randomize_mass = domain_randomize_mass
        self.train_random_push = train_random_push

        self._update_mass_dependent_params()

        # RL action / observation space
        self.max_residual = float(max_residual)

        # action: residual scale in [-1,1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        high = np.array(
            [
                self.x_threshold,
                np.finfo(np.float32).max,
                2 * self.theta_threshold_radians,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # A, B 계산 (nominal mass 기준)
        M = 1.0
        m = self.mass_nominal
        I = 0.033
        l = self.length
        Ad, Bd = calc_AB(M=M, m=m, I=I, l=l)
        self.Ad_nom = np.asarray(Ad, dtype=np.float64)
        self.Bd_nom = np.asarray(Bd, dtype=np.float64)

        # MPC 설정
        self.Np = 20
        self.Qx = np.diag([1.0, 0.1, 20.0, 0.1])
        self.QxN = self.Qx.copy()
        self.Qu = np.diag([0.01])
        self.QDu = np.diag([0.01])

        xmin = np.array(
            [-self.x_threshold, -10.0, -0.3, -10.0], dtype=np.float64
        )
        xmax = np.array(
            [self.x_threshold, 10.0, 0.3, 10.0], dtype=np.float64
        )

        self.force_mag = 20.0
        umin = np.array([-self.force_mag])
        umax = np.array([+self.force_mag])

        Dumin = np.array([-10.0])
        Dumax = np.array([+10.0])

        self.xref = np.zeros(4, dtype=np.float64)
        self.uminus1 = np.array([0.0])

        self.mpc = MPCController(
            self.Ad_nom,
            self.Bd_nom,
            Np=self.Np,
            x0=self.xref,
            xref=self.xref,
            uminus1=self.uminus1,
            Qx=self.Qx,
            QxN=self.QxN,
            Qu=self.Qu,
            QDu=self.QDu,
            xmin=xmin,
            xmax=xmax,
            umin=umin,
            umax=umax,
            Dumin=Dumin,
            Dumax=Dumax,
            eps_feas=1e3,
        )
        self.mpc.setup()

        self.state: Optional[np.ndarray] = None
        self.x = np.zeros(4, dtype=np.float64)
        self.step_count = 0

        # push 관련
        self.t = 0.0
        self.push_time: Optional[float] = None
        self.push_theta_dot: float = 0.0
        self.push_x_dot: float = 0.0
        self.pushed: bool = False

    # ----------------- 물리 파라미터 업데이트 -----------------
    def _update_mass_dependent_params(self):
        self.masspole = float(self.mass_true)
        self.total_mass = self.masscart + self.masspole
        self.polemass_length = self.masspole * self.length

    def set_mass_true(self, m_true: float):
        self.mass_true = float(m_true)
        self._update_mass_dependent_params()

    # ----------------- dynamics step -----------------
    def _cartpole_dynamics_step(self, force: float):
        assert self.state is not None, "Call reset() before step()."

        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.dt * x_dot
            x_dot = x_dot + self.dt * xacc
            theta = theta + self.dt * theta_dot
            theta_dot = theta_dot + self.dt * thetaacc
        else:
            x_dot = x_dot + self.dt * xacc
            x = x + self.dt * x_dot
            theta_dot = theta_dot + self.dt * thetaacc
            theta = theta + self.dt * theta_dot

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)

    # ----------------- reset -----------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # 1) 질량 설정
        if self.domain_randomize_mass:
            sampled_mass = self.np_random.uniform(self.mass_min, self.mass_max)
            self.set_mass_true(sampled_mass)
        else:
            # 테스트 모드에서는 set_mass_true() 를 밖에서 호출해놓는다고 가정
            self.set_mass_true(self.mass_true)

        # 2) 초기 상태
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.x = self.state.copy()
        self.step_count = 0

        self.t = 0.0
        self.pushed = False

        # 3) 학습 시 랜덤 push, 테스트 시 외부 설정 유지
        if self.train_random_push:
            self.push_time = float(self.np_random.uniform(0.3, 1.2))
            self.push_theta_dot = float(self.np_random.uniform(-0.5, 0.5))
            self.push_x_dot = float(self.np_random.uniform(-0.5, 0.5))

            # 70% 확률로만 push
            if self.np_random.random() > 0.7:
                self.push_time = None
                self.push_theta_dot = 0.0
                self.push_x_dot = 0.0
        else:
            # 테스트에서는 push_time 등은 밖에서 설정
            # 여기서 건들지 않음 (t, pushed만 초기화)
            pass

        # 4) 참조 상태
        self.xref = np.zeros(4, dtype=np.float64)

        # 5) MPC 초기화
        self.mpc.update(self.x, xref=self.xref)

        obs = self.x.astype(np.float32)
        info = {"mass_true": self.mass_true}
        return obs, info

    # ----------------- step -----------------
    def step(self, action):
        self.step_count += 1

        # 1) MPC 제어 입력
        u_mpc = float(self.mpc.output()) * (-1.0)

        # 2) RL residual
        if np.isscalar(action):
            a = float(action)
        else:
            a = float(np.clip(action[0], -1.0, 1.0))
        u_residual = self.max_residual * a

        # 3) 총 입력
        u_total = u_mpc + u_residual

        # 3.5) push 적용
        if (
            self.push_time is not None
            and (not self.pushed)
            and (self.t >= self.push_time)
        ):
            x, x_dot, theta, theta_dot = self.state
            x_dot += self.push_x_dot
            theta_dot += self.push_theta_dot
            self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
            self.pushed = True

        # 4) dynamics 진행
        self._cartpole_dynamics_step(u_total)

        # 5) 상태 갱신
        self.x = self.state.copy()

        # 6) MPC update
        self.mpc.update(self.x, xref=self.xref)

        # 7) 종료 조건 및 reward
        x, x_dot, theta, theta_dot = self.state

        terminated = bool(
            (x < -self.x_threshold)
            or (x > self.x_threshold)
            or (theta < -self.theta_threshold_radians)
            or (theta > self.theta_threshold_radians)
        )
        truncated = self.step_count >= self.max_episode_steps

        reward = 1.0
        reward -= 5.0 * (theta**2)
        reward -= 2.0 * (x**2)
        reward -= 0.05 * (u_residual**2)
        reward = float(reward)

        obs = self.x.astype(np.float32)
        info = {
            "u_mpc": u_mpc,
            "u_residual": u_residual,
            "u_total": u_total,
            "mass_true": self.mass_true,
            "pushed": self.pushed,
        }

        self.t += self.dt
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
