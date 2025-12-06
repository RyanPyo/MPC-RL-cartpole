# CartPole Control using MPC and Residual RL

This project investigates the robustness of a hybrid control approach that combines a **Model Predictive Controller (MPC)** with **Residual Reinforcement Learning (RL)**.  
A custom CartPole environment is used to compare:

- **MPC only**
- **MPC + PPO (Residual RL)**
- **MPC + SAC (Residual RL)**

The project evaluates robustness under **mass variation**, **push disturbances**, and **combined mass mismatch + disturbances** through Experiments A, B, and C.

---

##  Features

- Custom Gymnasium environment with:
  - Exact CartPole physics
  - Linear MPC controller with OSQP 
    (pyMPC- https://github.com/forgi86/pyMPC)
  - Residual reinforcement learning interface
  - Domain Randomization for training
  - Push disturbance injection
- Complete training and testing scripts

---

## Project Structure

cartpole_MPC_residual_RL/
│
├── trained_models/
│ ├── ppo_cartpole_mpc_residual.zip
│ └── sac_cartpole_mpc_residual.zip
│
├── code/
│ ├── calc_function.py
│ ├── cartpole_MPC_RL_env.py
│ ├── cartpole_MPC_RL_test.py
│ ├── cartpole_MPC_RL_train.py
│ ├── PPO_VS_SAC_expC_test.py
│
├── report/
│ ├── Term Project.ppt
│ ├── Term Project.pdf
│
├── README.md
└── requirements.txt

---

## Training the RL Model

### Training
```
python code/cartpole_MPC_RL_train.py
```
Inside the main function, change the "algorithm" variable to "PPO" or "SAC".

---

## Running Experiments

Run the evaluation for all experiments (A, B, C):
```
python code/cartpole_MPC_RL_test.py
```
This script will:
- Run all experiments
- Save all plots
- Print success statistics

Run evaluation for all experiment C to compare PPO vs SAC
```
python code/PPO_VS_SAC_expC_test.py
```

- Save comparison graphs (MPC vs PPO vs SAC)
---

## Experiment Summary

**Experiment A: Mass Variation (No Push)**
Tests robustness against mass changes  
(m = 0.05, 0.1, 0.2, 1.0)

**Experiment B: Push Disturbance**
Tests disturbance rejection with nominal mass

**Experiment C: Mass Mismatch + Push**
Most difficult test → real-world like conditions.

---

## Pretrained Models

Download pretrained models from:

- **PPO Model:**  
  `trained_models/ppo_cartpole_mpc_residual.zip`

- **SAC Model:**  
  `trained_models/sac_cartpole_mpc_residual.zip`

---
## Installation

Create a virtual environment (recommended):
```
conda create -n mpc_rl python=3.10
conda activate mpc_rl
```

Install dependencies:
```
pip install -r requirements.txt
```
---
## Dependencies (requirements.txt)

See **requirements.txt** section.