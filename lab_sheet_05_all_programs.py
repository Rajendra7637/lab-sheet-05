# %% [markdown]
# # Lab Sheet-05: Reinforcement Learning
# **MCA III Semester (Session 2026-2027) - COER University, Roorkee**
#
# Each `# %%` block is one experiment (a separate cell in VS Code / Jupyter).
#
# Libraries: NumPy, Pandas, Matplotlib, Gymnasium, PyTorch
# Environments: FrozenLake-v1, a custom Grid World, and CartPole-v1
#
# **Run time:** about 10-15 minutes in total (the DQN experiments take the longest).
# Set `QUICK_MODE = True` in the setup cell for a fast, rough run.

# %% [markdown]
# ## Setup: imports, settings and helper functions

# %%
import json
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
    if BASE_DIR.name == "notebooks":
        BASE_DIR = BASE_DIR.parent

MODEL_DIR = BASE_DIR / "models"
OUT_DIR = BASE_DIR / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# ---- Settings -------------------------------------------------------------
SEED = 42
QUICK_MODE = False            # True = fewer episodes, much faster, rougher results
FROZEN_EPISODES = 10000       # main Q-Learning training on FrozenLake
SWEEP_EPISODES = 5000         # each run in the learning-rate / gamma / epsilon studies
GRID_EPISODES = 2000          # Q-Learning on the custom Grid World
TABULAR_CARTPOLE_EPISODES = 3000
DQN_EPISODES = 300            # main DQN training on CartPole
ABLATION_EPISODES = 150       # each DQN experiment (replay, target network, learning rate)
if QUICK_MODE:
    FROZEN_EPISODES, SWEEP_EPISODES, GRID_EPISODES = 3000, 2000, 800
    TABULAR_CARTPOLE_EPISODES, DQN_EPISODES, ABLATION_EPISODES = 1000, 100, 60

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

RESULTS = {}  # key numbers are collected here and saved in Program 35


def moving_average(values, window):
    """Smooth a list of numbers with a moving average (same length as the input)."""
    values = np.asarray(values, dtype=float)
    window = max(1, min(window, len(values)))
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    smooth = (cumulative[window:] - cumulative[:-window]) / window
    head = [values[: i + 1].mean() for i in range(window - 1)]
    return np.concatenate([head, smooth])


def first_episode_reaching(smooth_values, threshold):
    """First episode number (1-based) where the smoothed value reaches the threshold."""
    hits = np.flatnonzero(np.asarray(smooth_values) >= threshold)
    return int(hits[0] + 1) if len(hits) else None


def save_and_show(file_name):
    """Save the current figure into the outputs folder and display it."""
    plt.savefig(OUT_DIR / file_name, dpi=120, bbox_inches="tight")
    plt.show()


# %% [markdown]
# # Part A: Gymnasium and the FrozenLake Environment (Programs 1-5)

# %% [markdown]
# ## Program 1: Install and configure the Gymnasium library

# %%
# Install in the terminal:  pip install gymnasium
print("Gymnasium version:", gym.__version__)
available = [name for name in ["FrozenLake-v1", "CartPole-v1", "Taxi-v3"] if name in gym.registry]
print("Environments found:", available)

# %% [markdown]
# ## Program 2: Create and execute a simple Reinforcement Learning environment

# %%
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
state, info = env.reset(seed=SEED)
print("Start state:", state)

total_reward = 0
for step_number in range(1, 101):
    action = env.action_space.sample()                       # pick a random action
    state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    print(f"Step {step_number}: action={action}, new state={state}, reward={reward}")
    if terminated or truncated:
        break
print("Episode finished. Total reward:", total_reward)
env.close()

# %% [markdown]
# ## Program 3: Explore the observation space and action space

# %%
frozen_env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
cartpole_env = gym.make("CartPole-v1")

print("FrozenLake observation space:", frozen_env.observation_space)
print("FrozenLake action space     :", frozen_env.action_space)
print("   -> number of states :", frozen_env.observation_space.n)
print("   -> number of actions:", frozen_env.action_space.n)

print("\nCartPole observation space  :", cartpole_env.observation_space)
print("CartPole action space       :", cartpole_env.action_space)
print("   -> state values: cart position, cart velocity, pole angle, pole angular velocity")
cartpole_env.close()

# %% [markdown]
# ## Program 4: States, actions, rewards and termination conditions

# %%
grid_map = frozen_env.unwrapped.desc.astype(str)
print("FrozenLake 4x4 map (S=start, F=frozen, H=hole, G=goal):")
for row in grid_map:
    print("  ", " ".join(row))

print("\nStates : 0 to 15 (row * 4 + column)")
print("Actions: 0=Left, 1=Down, 2=Right, 3=Up")
print("Rewards: +1 for reaching the goal, 0 for every other step")
print("Episode ends (terminated): the agent falls in a hole (H) or reaches the goal (G)")
print("Episode ends (truncated) : after 100 steps")
print("\nThe ice is slippery: each action works only 1/3 of the time, and the agent")
print("slides sideways the other 2/3. Transitions from state 0, action 'Down':")
for probability, next_state, reward, done in frozen_env.unwrapped.P[0][1]:
    print(f"   probability={probability:.2f} -> state {next_state}, reward {reward}, done={done}")

# %% [markdown]
# ## Program 5: Simulate random actions in FrozenLake

# %%
random_rng = np.random.default_rng(SEED)
random_successes, random_steps = 0, []
for episode in range(1000):
    state, _ = frozen_env.reset(seed=SEED if episode == 0 else None)
    done, steps = False, 0
    while not done:
        action = int(random_rng.integers(4))
        state, reward, terminated, truncated, _ = frozen_env.step(action)
        done = terminated or truncated
        steps += 1
    random_successes += reward
    random_steps.append(steps)

random_success_rate = random_successes / 1000
print(f"Random agent reached the goal in {int(random_successes)} of 1000 episodes")
print(f"Success rate: {random_success_rate:.1%} | average steps per episode: {np.mean(random_steps):.1f}")
RESULTS["frozenlake_random_success_rate"] = random_success_rate

# %% [markdown]
# # Part B: Q-Learning on FrozenLake (Programs 6-15)

# %% [markdown]
# ## Program 6: Implement the Q-Learning algorithm for FrozenLake

# %%
def epsilon_greedy_action(q_values_of_state, epsilon, rng):
    """Epsilon-greedy policy (also explained in Program 15).

    With probability epsilon  -> explore: pick a random action.
    With probability 1-epsilon -> exploit: pick the action with the highest Q-value
    (ties are broken randomly, which matters while the Q-table is still all zeros).
    """
    if rng.random() < epsilon:
        return int(rng.integers(len(q_values_of_state)))
    best_actions = np.flatnonzero(q_values_of_state == np.max(q_values_of_state))
    return int(rng.choice(best_actions))


def train_q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon_start=1.0,
                     epsilon_end=0.01, epsilon_decay=0.9995, fixed_epsilon=None,
                     seed=SEED, print_every=0):
    """Tabular Q-Learning.

    Update rule after every step:
        Q(s, a) = Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
    """
    rng = np.random.default_rng(seed)
    n_states, n_actions = env.observation_space.n, env.action_space.n
    q_table = np.zeros((n_states, n_actions))          # Program 7: initialize Q-table
    episode_rewards, episode_steps, episode_td_error, epsilon_history = [], [], [], []
    start_time = time.time()

    for episode in range(episodes):
        state, _ = env.reset(seed=seed) if episode == 0 else env.reset()
        epsilon = fixed_epsilon if fixed_epsilon is not None else max(
            epsilon_end, epsilon_start * epsilon_decay ** episode)
        done, total_reward, steps, td_errors = False, 0.0, 0, []

        while not done:
            action = epsilon_greedy_action(q_table[state], epsilon, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Program 7: update the Q-table
            best_next_value = 0.0 if terminated else np.max(q_table[next_state])
            td_error = reward + gamma * best_next_value - q_table[state, action]
            q_table[state, action] += alpha * td_error

            td_errors.append(abs(td_error))
            total_reward += reward
            steps += 1
            state = next_state
            done = terminated or truncated

        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        episode_td_error.append(float(np.mean(td_errors)))
        epsilon_history.append(epsilon)

        if print_every and (episode + 1) % print_every == 0:
            recent = np.mean(episode_rewards[-print_every:])
            print(f"Episode {episode + 1:6d} | average reward (last {print_every}): "
                  f"{recent:.3f} | epsilon: {epsilon:.3f}")

    return {"q_table": q_table, "rewards": np.array(episode_rewards),
            "steps": np.array(episode_steps), "td_error": np.array(episode_td_error),
            "epsilons": np.array(epsilon_history), "time": time.time() - start_time}


def evaluate_q_table(env, q_table, episodes=1000, seed=SEED + 1):
    """Run the greedy policy (no exploration) and report how well it does."""
    rng = np.random.default_rng(seed)
    rewards, steps_list, successes = [], [], 0
    for episode in range(episodes):
        state, _ = env.reset(seed=seed) if episode == 0 else env.reset()
        done, total, steps, reward = False, 0.0, 0, 0.0
        while not done:
            action = epsilon_greedy_action(q_table[state], 0.0, rng)
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            steps += 1
            done = terminated or truncated
        rewards.append(total)
        steps_list.append(steps)
        successes += int(reward > 0)          # the last reward is positive only at the goal
    return {"success_rate": successes / episodes, "avg_reward": float(np.mean(rewards)),
            "avg_steps": float(np.mean(steps_list))}


print("Q-Learning functions defined.")
print("Update rule: Q(s,a) = Q(s,a) + alpha * [reward + gamma * max Q(s',a') - Q(s,a)]")

# %% [markdown]
# ## Program 7: Initialize and update the Q-table during training

# %%
demo_q_table = np.zeros((16, 4))
print("Initial Q-table (all zeros), first 3 states:\n", demo_q_table[:3])

# One manual update: the agent is in state 14, takes action 2 (Right) and reaches the goal (state 15)
state_demo, action_demo, reward_demo, next_state_demo = 14, 2, 1.0, 15
alpha_demo, gamma_demo = 0.1, 0.99
old_value = demo_q_table[state_demo, action_demo]
new_value = old_value + alpha_demo * (
    reward_demo + gamma_demo * 0.0 - old_value)       # goal is a terminal state
demo_q_table[state_demo, action_demo] = new_value
print(f"\nUpdate for state {state_demo}, action {action_demo}: "
      f"{old_value:.3f} + 0.1 * (1 + 0 - {old_value:.3f}) = {new_value:.3f}")
print("Q-table row for state 14 after the update:", demo_q_table[14])

# %% [markdown]
# ## Program 8: Train the agent for multiple episodes using Q-Learning

# %%
frozen_env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
frozen_result = train_q_learning(frozen_env, FROZEN_EPISODES, alpha=0.1, gamma=0.99,
                                 print_every=FROZEN_EPISODES // 10)
print(f"\nTraining finished in {frozen_result['time']:.1f} seconds")

# %% [markdown]
# ## Program 9: Display the learned Q-table after training

# %%
frozen_q_table = frozen_result["q_table"]
q_table_df = pd.DataFrame(frozen_q_table.round(4), columns=["Left", "Down", "Right", "Up"])
q_table_df.index.name = "State"
print(q_table_df)

# The best action in each state, drawn on the 4x4 map
arrows = {0: "<", 1: "v", 2: ">", 3: "^"}
print("\nLearned policy (best action in each state):")
for row in range(4):
    line = []
    for col in range(4):
        cell = grid_map[row][col]
        if cell in ("H", "G"):
            line.append(cell)
        else:
            line.append(arrows[int(np.argmax(frozen_q_table[row * 4 + col]))])
    print("  ", " ".join(line))
np.save(MODEL_DIR / "frozenlake_q_table.npy", frozen_q_table)

# %% [markdown]
# ## Program 10: Evaluate the trained Q-Learning agent

# %%
frozen_eval = evaluate_q_table(frozen_env, frozen_q_table, episodes=1000)
print(f"Greedy agent success rate : {frozen_eval['success_rate']:.1%}")
print(f"Average reward            : {frozen_eval['avg_reward']:.3f}")
print(f"Average steps per episode : {frozen_eval['avg_steps']:.1f}")
print(f"(Random agent success rate was {random_success_rate:.1%})")

# %% [markdown]
# ## Program 11: Plot the cumulative rewards obtained during training

# %%
frozen_rewards = frozen_result["rewards"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(np.cumsum(frozen_rewards), color="tab:blue")
axes[0].set_title("Cumulative reward during training")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Cumulative reward")
axes[0].grid(True)

axes[1].plot(moving_average(frozen_rewards, 200), color="tab:green")
axes[1].set_title("Success rate (moving average of 200 episodes)")
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Success rate")
axes[1].grid(True)
plt.tight_layout()
save_and_show("p11_frozenlake_cumulative_rewards.png")

frozen_smooth = moving_average(frozen_rewards, 200)
RESULTS["frozenlake"] = {
    "train_time_sec": round(frozen_result["time"], 2),
    "eval_success_rate": round(frozen_eval["success_rate"], 4),
    "eval_avg_steps": round(frozen_eval["avg_steps"], 2),
    "episodes": FROZEN_EPISODES,
    "converge_episode": first_episode_reaching(frozen_smooth, 0.9 * frozen_smooth[-1]),
}

# %% [markdown]
# ## Program 12: Effect of different learning rates on Q-Learning

# %%
learning_rates = [0.01, 0.1, 0.5, 0.9]
lr_study = []
plt.figure(figsize=(8, 4))
for alpha_value in learning_rates:
    run = train_q_learning(frozen_env, SWEEP_EPISODES, alpha=alpha_value, gamma=0.99,
                           epsilon_decay=0.999)
    evaluation = evaluate_q_table(frozen_env, run["q_table"], episodes=500)
    lr_study.append({"Learning rate": alpha_value,
                     "Training success rate": round(run["rewards"][-500:].mean(), 3),
                     "Greedy success rate": round(evaluation["success_rate"], 3)})
    plt.plot(moving_average(run["rewards"], 200), label=f"alpha = {alpha_value}")
plt.title("Effect of learning rate (alpha)")
plt.xlabel("Episode")
plt.ylabel("Success rate (moving average)")
plt.legend()
plt.grid(True)
save_and_show("p12_learning_rate_effect.png")
lr_study_df = pd.DataFrame(lr_study)
print(lr_study_df)
RESULTS["learning_rate_study"] = lr_study

# %% [markdown]
# ## Program 13: Effect of different discount factor (Gamma) values

# %%
gamma_values = [0.5, 0.9, 0.99, 0.999]
gamma_study = []
plt.figure(figsize=(8, 4))
for gamma_value in gamma_values:
    run = train_q_learning(frozen_env, SWEEP_EPISODES, alpha=0.1, gamma=gamma_value,
                           epsilon_decay=0.999)
    evaluation = evaluate_q_table(frozen_env, run["q_table"], episodes=500)
    gamma_study.append({"Gamma": gamma_value,
                        "Training success rate": round(run["rewards"][-500:].mean(), 3),
                        "Greedy success rate": round(evaluation["success_rate"], 3)})
    plt.plot(moving_average(run["rewards"], 200), label=f"gamma = {gamma_value}")
plt.title("Effect of discount factor (gamma)")
plt.xlabel("Episode")
plt.ylabel("Success rate (moving average)")
plt.legend()
plt.grid(True)
save_and_show("p13_gamma_effect.png")
print(pd.DataFrame(gamma_study))
print("\nA small gamma makes the agent short-sighted. The reward only comes at the end,")
print("so it needs a high gamma to value the distant goal.")
RESULTS["gamma_study"] = gamma_study

# %% [markdown]
# ## Program 14: Compare exploration and exploitation using different epsilon values

# %%
epsilon_settings = [("epsilon = 0.0 (only exploit)", 0.0), ("epsilon = 0.1", 0.1),
                    ("epsilon = 0.5", 0.5), ("epsilon = 1.0 (only explore)", 1.0),
                    ("decaying 1.0 -> 0.01", None)]
epsilon_study = []
plt.figure(figsize=(8, 4))
for label, epsilon_value in epsilon_settings:
    run = train_q_learning(frozen_env, SWEEP_EPISODES, alpha=0.1, gamma=0.99,
                           fixed_epsilon=epsilon_value, epsilon_decay=0.999)
    evaluation = evaluate_q_table(frozen_env, run["q_table"], episodes=500)
    epsilon_study.append({"Setting": label,
                          "Training success rate": round(run["rewards"][-500:].mean(), 3),
                          "Greedy success rate": round(evaluation["success_rate"], 3)})
    plt.plot(moving_average(run["rewards"], 200), label=label)
plt.title("Exploration vs exploitation")
plt.xlabel("Episode")
plt.ylabel("Success rate (moving average)")
plt.legend(fontsize=8)
plt.grid(True)
save_and_show("p14_epsilon_effect.png")
print(pd.DataFrame(epsilon_study))
print("\nNote: with epsilon = 0 the agent never explores, so it rarely finds the goal.")
print("Q-Learning is 'off-policy': it can learn good Q-values even from random")
print("experience, so the greedy success rate can stay high even with epsilon = 1.0,")
print("but the success rate DURING training stays low because the agent keeps acting randomly.")
RESULTS["epsilon_study"] = epsilon_study

# %% [markdown]
# ## Program 15: Implement an epsilon-greedy action selection policy

# %%
# The function epsilon_greedy_action() was defined in Program 6. Here we test it.
test_q_values = np.array([0.1, 0.5, 0.2, 0.3])          # action 1 is the best
test_rng = np.random.default_rng(SEED)
print("Q-values:", test_q_values, "-> best action is 1\n")
for epsilon_value in [0.0, 0.1, 0.5, 1.0]:
    chosen = [epsilon_greedy_action(test_q_values, epsilon_value, test_rng) for _ in range(10000)]
    share_best = np.mean(np.array(chosen) == 1)
    print(f"epsilon = {epsilon_value:.1f} -> best action chosen {share_best:.1%} of the time")
print("\nWith epsilon = 0.1 the agent exploits about 90% of the time (plus 1/4 of the")
print("random 10%), so the best action is picked about 92.5% of the time.")

# %% [markdown]
# # Part C: Custom Grid World (Programs 16-20)

# %% [markdown]
# ## Program 16: Design a simple Grid World environment

# %%
class GridWorldEnv(gym.Env):
    """A 5x5 grid. The agent starts at the top-left and must reach the goal.

    Cells : S = start (0,0), G = goal (4,4), P = pit (1,3) and (3,1), # = wall (2,2)
    Actions: 0=Up, 1=Right, 2=Down, 3=Left
    Rewards: -1 for every step, +20 for the goal, -10 for a pit
    The episode ends at the goal, in a pit, or after 50 steps.
    """

    def __init__(self):
        super().__init__()
        self.size = 5
        self.start = (0, 0)
        self.goal = (4, 4)
        self.pits = {(1, 3), (3, 1)}
        self.walls = {(2, 2)}
        self.max_steps = 50
        self.observation_space = gym.spaces.Discrete(self.size * self.size)
        self.action_space = gym.spaces.Discrete(4)
        self.moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        self.position = self.start
        self.steps = 0

    def _state(self):
        return self.position[0] * self.size + self.position[1]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.position = self.start
        self.steps = 0
        return self._state(), {}

    def step(self, action):
        row_change, col_change = self.moves[int(action)]
        new_row = min(max(self.position[0] + row_change, 0), self.size - 1)
        new_col = min(max(self.position[1] + col_change, 0), self.size - 1)
        if (new_row, new_col) not in self.walls:       # walls block the move
            self.position = (new_row, new_col)
        self.steps += 1

        reward, terminated = -1.0, False
        if self.position == self.goal:
            reward, terminated = 20.0, True
        elif self.position in self.pits:
            reward, terminated = -10.0, True
        truncated = self.steps >= self.max_steps and not terminated
        return self._state(), reward, terminated, truncated, {}

    def layout(self):
        """Return the grid as a list of text rows."""
        rows = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                cell = (r, c)
                row.append("S" if cell == self.start else "G" if cell == self.goal
                           else "P" if cell in self.pits else "#" if cell in self.walls else ".")
            rows.append(" ".join(row))
        return rows


grid_env = GridWorldEnv()
print("Grid World layout:")
print("\n".join(grid_env.layout()))
print("\nStates:", grid_env.observation_space.n, "| Actions:", grid_env.action_space.n)

# %% [markdown]
# ## Program 17: Train a Q-Learning agent in the custom Grid World

# %%
grid_result = train_q_learning(grid_env, GRID_EPISODES, alpha=0.1, gamma=0.95,
                               epsilon_decay=0.998, print_every=GRID_EPISODES // 5)
grid_eval = evaluate_q_table(grid_env, grid_result["q_table"], episodes=200)
print(f"\nTraining time: {grid_result['time']:.1f} s")
print(f"Greedy success rate: {grid_eval['success_rate']:.1%} | "
      f"average reward: {grid_eval['avg_reward']:.2f} | average steps: {grid_eval['avg_steps']:.1f}")

# %% [markdown]
# ## Program 18: Visualize the optimal path learned by the agent

# %%
def follow_greedy_path(env, q_table, max_moves=30):
    """Return the list of cells visited when the agent always takes its best action."""
    state, _ = env.reset()
    path = [env.position]
    for _ in range(max_moves):
        action = int(np.argmax(q_table[state]))
        state, _, terminated, truncated, _ = env.step(action)
        path.append(env.position)
        if terminated or truncated:
            break
    return path


optimal_path = follow_greedy_path(grid_env, grid_result["q_table"])
print("Path:", " -> ".join(str(cell) for cell in optimal_path))
print("Number of moves:", len(optimal_path) - 1, "(the shortest possible path is 8 moves)")

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-0.5, 4.5)
ax.set_ylim(4.5, -0.5)
ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.grid(True)
for r in range(5):
    for c in range(5):
        cell = (r, c)
        if cell in grid_env.pits:
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="tomato"))
            ax.text(c, r, "Pit", ha="center", va="center")
        elif cell in grid_env.walls:
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="gray"))
            ax.text(c, r, "Wall", ha="center", va="center", color="white")
        elif cell == grid_env.goal:
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="lightgreen"))
            ax.text(c, r, "Goal", ha="center", va="center")
        elif cell == grid_env.start:
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="lightskyblue"))
            ax.text(c, r, "Start", ha="center", va="center")
path_rows = [cell[0] for cell in optimal_path]
path_cols = [cell[1] for cell in optimal_path]
ax.plot(path_cols, path_rows, "o-", color="navy", linewidth=2)
ax.set_title("Optimal path learned by the agent")
save_and_show("p18_gridworld_optimal_path.png")

# %% [markdown]
# ## Program 19: Compare agent performance in FrozenLake and Grid World

# %%
grid_smooth = moving_average(grid_result["rewards"], 100)
RESULTS["gridworld"] = {
    "train_time_sec": round(grid_result["time"], 2),
    "eval_success_rate": round(grid_eval["success_rate"], 4),
    "eval_avg_steps": round(grid_eval["avg_steps"], 2),
    "episodes": GRID_EPISODES,
    "converge_episode": first_episode_reaching(grid_smooth, 0.9 * grid_smooth[-1]) if grid_smooth[-1] > 0 else None,
}
env_comparison = pd.DataFrame({
    "FrozenLake (slippery)": [frozen_eval["success_rate"], frozen_eval["avg_steps"],
                              frozen_result["time"], FROZEN_EPISODES],
    "Grid World (deterministic)": [grid_eval["success_rate"], grid_eval["avg_steps"],
                                   grid_result["time"], GRID_EPISODES],
}, index=["Success rate", "Average steps", "Training time (s)", "Training episodes"]).round(3)
print(env_comparison)
print("\nGrid World is deterministic, so the agent can reach a perfect policy.")
print("FrozenLake is slippery (random), so even the best policy fails sometimes.")

# %% [markdown]
# ## Program 20: Analyze the convergence behavior of Q-Learning

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(moving_average(frozen_result["td_error"], 100), label="FrozenLake")
axes[0].plot(moving_average(grid_result["td_error"], 100), label="Grid World")
axes[0].set_title("Average update size (TD error) per episode")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Mean |TD error|")
axes[0].set_yscale("log")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(frozen_result["epsilons"], label="FrozenLake")
axes[1].plot(grid_result["epsilons"], label="Grid World")
axes[1].set_title("Epsilon decay")
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Epsilon")
axes[1].legend()
axes[1].grid(True)
plt.tight_layout()
save_and_show("p20_convergence.png")
print("Converged (90% of final performance) after episode:")
print("  FrozenLake:", RESULTS["frozenlake"]["converge_episode"])
print("  Grid World:", RESULTS["gridworld"]["converge_episode"])
print("\nAs the Q-values settle, the updates get smaller. In FrozenLake they never reach")
print("zero because the random slipping keeps producing new, slightly different rewards.")

# %% [markdown]
# # Part D: Deep Q-Network (DQN) on CartPole (Programs 21-30)

# %% [markdown]
# ## Program 21: Install the required libraries for Deep Q-Network implementation

# %%
# Install in the terminal:  pip install torch gymnasium
# (the CPU version is enough. See pytorch.org for the install command of your system)
print("PyTorch version:", torch.__version__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training device:", DEVICE)

# %% [markdown]
# ## Program 22: Implement a basic Deep Q-Network using PyTorch

# %%
class QNetwork(nn.Module):
    """A small neural network: state (4 numbers) -> Q-value of each action (2 numbers)."""

    def __init__(self, state_size, action_size, hidden_size=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x):
        return self.layers(x)


class ReplayBuffer:
    """Replay memory: stores past transitions and returns random mini-batches."""

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size, rng):
        indices = rng.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.as_tensor(np.array(states), dtype=torch.float32, device=DEVICE),
                torch.as_tensor(actions, dtype=torch.int64, device=DEVICE).unsqueeze(1),
                torch.as_tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(1),
                torch.as_tensor(np.array(next_states), dtype=torch.float32, device=DEVICE),
                torch.as_tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(1))

    def __len__(self):
        return len(self.memory)


def train_dqn(episodes, use_replay=True, use_target=True, learning_rate=1e-3, gamma=0.99,
              batch_size=64, buffer_size=10000, target_update_steps=250,
              epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=2000,
              learn_start=300, seed=SEED, print_every=0):
    """Train a DQN agent on CartPole-v1.

    use_replay=False : learn only from the latest transition (no replay memory)
    use_target=False : use the same network to compute the target (no target network)
    """
    env = gym.make("CartPole-v1")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    state_size, action_size = env.observation_space.shape[0], env.action_space.n

    policy_net = QNetwork(state_size, action_size).to(DEVICE)
    target_net = QNetwork(state_size, action_size).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    loss_function = nn.SmoothL1Loss()
    buffer = ReplayBuffer(buffer_size)

    def learn(states, actions, rewards, next_states, dones):
        """One gradient step on the Bellman error."""
        current_q = policy_net(states).gather(1, actions)
        with torch.no_grad():
            network_for_target = target_net if use_target else policy_net
            next_q = network_for_target(next_states).max(1, keepdim=True)[0]
            target_q = rewards + gamma * next_q * (1 - dones)
        loss = loss_function(current_q, target_q)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
        optimizer.step()
        return loss.item()

    episode_rewards, episode_losses = [], []
    total_steps, start_time = 0, time.time()

    for episode in range(episodes):
        state, _ = env.reset(seed=seed) if episode == 0 else env.reset()
        done, total_reward, losses = False, 0.0, []

        while not done:
            epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end)
                          * total_steps / epsilon_decay_steps)
            if rng.random() < epsilon:
                action = int(rng.integers(action_size))
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.as_tensor(state, dtype=torch.float32,
                                                          device=DEVICE).unsqueeze(0))
                action = int(q_values.argmax(dim=1).item())

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, float(terminated))
            state = next_state
            total_reward += reward
            total_steps += 1

            if total_steps >= learn_start:
                if use_replay and len(buffer) >= batch_size:
                    losses.append(learn(*buffer.sample(batch_size, rng)))
                elif not use_replay:
                    latest = buffer.memory[-1]
                    batch = (torch.as_tensor(np.array([latest[0]]), dtype=torch.float32, device=DEVICE),
                             torch.as_tensor([[latest[1]]], dtype=torch.int64, device=DEVICE),
                             torch.as_tensor([[latest[2]]], dtype=torch.float32, device=DEVICE),
                             torch.as_tensor(np.array([latest[3]]), dtype=torch.float32, device=DEVICE),
                             torch.as_tensor([[latest[4]]], dtype=torch.float32, device=DEVICE))
                    losses.append(learn(*batch))

            if use_target and total_steps % target_update_steps == 0:
                target_net.load_state_dict(policy_net.state_dict())   # copy weights

        episode_rewards.append(total_reward)
        episode_losses.append(float(np.mean(losses)) if losses else np.nan)

        if print_every and (episode + 1) % print_every == 0:
            print(f"Episode {episode + 1:4d} | reward: {total_reward:6.1f} | "
                  f"average (last {print_every}): {np.mean(episode_rewards[-print_every:]):6.1f} "
                  f"| epsilon: {epsilon:.2f}")

    env.close()
    return {"policy_net": policy_net, "rewards": np.array(episode_rewards),
            "losses": np.array(episode_losses), "time": time.time() - start_time}


def evaluate_dqn(network, episodes=20, seed=SEED + 100):
    """Run the trained network greedily (no exploration)."""
    env = gym.make("CartPole-v1")
    scores = []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        done, total = False, 0.0
        while not done:
            with torch.no_grad():
                q_values = network(torch.as_tensor(state, dtype=torch.float32,
                                                   device=DEVICE).unsqueeze(0))
            state, reward, terminated, truncated, _ = env.step(int(q_values.argmax(dim=1).item()))
            total += reward
            done = terminated or truncated
        scores.append(total)
    env.close()
    return np.array(scores)


sample_network = QNetwork(4, 2)
print(sample_network)
print("Number of trainable parameters:", sum(p.numel() for p in sample_network.parameters()))
print("Q-values for a sample state:", sample_network(torch.zeros(1, 4)).detach().numpy().round(3))

# %% [markdown]
# ## Program 23: Train a DQN agent on the CartPole environment

# %%
print(f"Training DQN for {DQN_EPISODES} episodes (this takes a few minutes)...")
dqn_result = train_dqn(DQN_EPISODES, print_every=max(1, DQN_EPISODES // 10))
print(f"\nTraining finished in {dqn_result['time']:.1f} seconds")

# %% [markdown]
# ## Program 24: Plot the episode-wise reward during DQN training

# %%
dqn_rewards = dqn_result["rewards"]
plt.figure(figsize=(9, 4))
plt.plot(dqn_rewards, alpha=0.4, label="Episode reward")
plt.plot(moving_average(dqn_rewards, 20), color="red", linewidth=2, label="Moving average (20)")
plt.axhline(195, color="green", linestyle="--", label="'Solved' level (195)")
plt.title("DQN training on CartPole")
plt.xlabel("Episode")
plt.ylabel("Reward (steps the pole stayed up)")
plt.legend()
plt.grid(True)
save_and_show("p24_dqn_training_rewards.png")

# %% [markdown]
# ## Program 25: Evaluate the trained DQN agent

# %%
dqn_scores = evaluate_dqn(dqn_result["policy_net"], episodes=20)
print("Scores of 20 test episodes:", dqn_scores.astype(int))
print(f"Average score: {dqn_scores.mean():.1f} | best: {dqn_scores.max():.0f} | worst: {dqn_scores.min():.0f}")
print("(500 is the maximum score in CartPole-v1)")

dqn_smooth = moving_average(dqn_rewards, 20)
RESULTS["dqn"] = {
    "train_time_sec": round(dqn_result["time"], 2),
    "episodes": DQN_EPISODES,
    "eval_mean_reward": round(float(dqn_scores.mean()), 2),
    "last_20_training_mean": round(float(dqn_rewards[-20:].mean()), 2),
    "converge_episode_195": first_episode_reaching(dqn_smooth, 195),
}

# %% [markdown]
# ## Program 26: Compare Q-Learning and DQN performance

# %%
# Q-Learning needs a table, so it cannot handle CartPole's continuous states directly.
# To compare both on the SAME environment, we turn the 4 continuous numbers into
# a few "bins" (discretization) and run ordinary Q-Learning on the bins.
def train_tabular_cartpole(episodes, seed=SEED):
    """Q-Learning on CartPole using binned (discretized) states."""
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed)
    bins = (1, 1, 6, 12)                       # cart position, cart velocity, angle, angular velocity
    lower = [-2.4, -3.0, -0.21, -3.5]
    upper = [2.4, 3.0, 0.21, 3.5]
    q_table = np.zeros(bins + (2,))
    rewards, start_time = [], time.time()

    def to_bins(observation):
        indices = []
        for value, low, high, n_bins in zip(observation, lower, upper, bins):
            ratio = (np.clip(value, low, high) - low) / (high - low)
            indices.append(min(n_bins - 1, int(ratio * n_bins)))
        return tuple(indices)

    for episode in range(episodes):
        decay = max(0.01, min(1.0, 1.0 - np.log10((episode + 1) / 25)))
        alpha, epsilon = max(0.1, decay), decay
        observation, _ = env.reset(seed=seed) if episode == 0 else env.reset()
        state, done, total = to_bins(observation), False, 0.0
        while not done:
            action = epsilon_greedy_action(q_table[state], epsilon, rng)
            observation, reward, terminated, truncated, _ = env.step(action)
            next_state = to_bins(observation)
            best_next = 0.0 if terminated else np.max(q_table[next_state])
            q_table[state + (action,)] += alpha * (reward + 0.99 * best_next - q_table[state + (action,)])
            state, total = next_state, total + reward
            done = terminated or truncated
        rewards.append(total)
    env.close()
    return {"rewards": np.array(rewards), "time": time.time() - start_time}


tabular_result = train_tabular_cartpole(TABULAR_CARTPOLE_EPISODES)
tabular_smooth = moving_average(tabular_result["rewards"], 100)

# Random-action baseline on CartPole
baseline_env = gym.make("CartPole-v1")
random_cartpole_rewards = []
for episode in range(100):
    baseline_env.reset(seed=SEED + episode)
    done, total = False, 0.0
    while not done:
        _, reward, terminated, truncated, _ = baseline_env.step(baseline_env.action_space.sample())
        total += reward
        done = terminated or truncated
    random_cartpole_rewards.append(total)
baseline_env.close()

cartpole_comparison = pd.DataFrame({
    "Random actions": [np.mean(random_cartpole_rewards), "-", "-"],
    "Q-Learning (binned states)": [np.mean(tabular_result["rewards"][-100:]),
                                   TABULAR_CARTPOLE_EPISODES, round(tabular_result["time"], 1)],
    "DQN (neural network)": [dqn_scores.mean(), DQN_EPISODES, round(dqn_result["time"], 1)],
}, index=["Average reward", "Training episodes", "Training time (s)"])
print(cartpole_comparison)
RESULTS["cartpole_random_mean"] = round(float(np.mean(random_cartpole_rewards)), 2)
RESULTS["tabular_cartpole"] = {
    "train_time_sec": round(tabular_result["time"], 2),
    "episodes": TABULAR_CARTPOLE_EPISODES,
    "last_100_training_mean": round(float(tabular_result["rewards"][-100:].mean()), 2),
    "converge_episode_195": first_episode_reaching(tabular_smooth, 195),
}

# %% [markdown]
# ## Program 27: Analyze the effect of replay memory on DQN performance

# %%
print(f"Training DQN without replay memory ({ABLATION_EPISODES} episodes)...")
no_replay_result = train_dqn(ABLATION_EPISODES, use_replay=False)
with_replay_rewards = dqn_result["rewards"][:ABLATION_EPISODES]   # same settings, same seed

plt.figure(figsize=(9, 4))
plt.plot(moving_average(with_replay_rewards, 20), label="With replay memory")
plt.plot(moving_average(no_replay_result["rewards"], 20), label="Without replay memory")
plt.title("Effect of replay memory (moving average of 20 episodes)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
save_and_show("p27_replay_memory_effect.png")
replay_table = pd.DataFrame({
    "With replay": [with_replay_rewards[-20:].mean(), with_replay_rewards.max()],
    "Without replay": [no_replay_result["rewards"][-20:].mean(), no_replay_result["rewards"].max()],
}, index=["Mean reward (last 20 episodes)", "Best episode"]).round(1)
print(replay_table)
print("\nReplay memory lets the agent learn from random older experiences, which breaks")
print("the strong link between consecutive steps and makes learning more stable.")
RESULTS["replay_effect"] = {
    "with_replay_last20": round(float(with_replay_rewards[-20:].mean()), 2),
    "without_replay_last20": round(float(no_replay_result["rewards"][-20:].mean()), 2),
}

# %% [markdown]
# ## Program 28: Study the role of the target network in DQN

# %%
print(f"Training DQN without a target network ({ABLATION_EPISODES} episodes)...")
no_target_result = train_dqn(ABLATION_EPISODES, use_target=False)

plt.figure(figsize=(9, 4))
plt.plot(moving_average(with_replay_rewards, 20), label="With target network")
plt.plot(moving_average(no_target_result["rewards"], 20), label="Without target network")
plt.title("Effect of the target network (moving average of 20 episodes)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
save_and_show("p28_target_network_effect.png")
target_table = pd.DataFrame({
    "With target network": [with_replay_rewards[-20:].mean(), with_replay_rewards.max()],
    "Without target network": [no_target_result["rewards"][-20:].mean(), no_target_result["rewards"].max()],
}, index=["Mean reward (last 20 episodes)", "Best episode"]).round(1)
print(target_table)
print("\nThe target network is a slowly updated copy of the main network. It keeps the")
print("learning target steady, instead of 'chasing a moving target'.")
RESULTS["target_effect"] = {
    "with_target_last20": round(float(with_replay_rewards[-20:].mean()), 2),
    "without_target_last20": round(float(no_target_result["rewards"][-20:].mean()), 2),
}

# %% [markdown]
# ## Program 29: Save the trained DQN model

# %%
try:
    torch.save(dqn_result["policy_net"].state_dict(), MODEL_DIR / "dqn_cartpole.pt")
    print("DQN model saved:", MODEL_DIR / "dqn_cartpole.pt")
    print("Q-table saved  :", MODEL_DIR / "frozenlake_q_table.npy")
except OSError as err:
    print("Could not save model:", err)

# %% [markdown]
# ## Program 30: Load the saved DQN model and perform testing

# %%
try:
    loaded_network = QNetwork(4, 2).to(DEVICE)
    loaded_network.load_state_dict(torch.load(MODEL_DIR / "dqn_cartpole.pt", map_location=DEVICE))
    loaded_network.eval()
    loaded_scores = evaluate_dqn(loaded_network, episodes=10, seed=SEED + 500)
    print("Scores of the loaded model in 10 test episodes:", loaded_scores.astype(int))
    print(f"Average score: {loaded_scores.mean():.1f}")
except FileNotFoundError:
    print("Model file not found. Run Program 29 first.")

# %% [markdown]
# # Part E: Comparison and Analysis (Programs 31-35)

# %% [markdown]
# ## Program 31: Compare cumulative rewards using different RL algorithms

# %%
episodes_to_compare = min(DQN_EPISODES, TABULAR_CARTPOLE_EPISODES)
plt.figure(figsize=(9, 4))
plt.plot(np.cumsum(dqn_result["rewards"][:episodes_to_compare]), label="DQN")
plt.plot(np.cumsum(tabular_result["rewards"][:episodes_to_compare]), label="Q-Learning (binned states)")
plt.plot(np.cumsum(random_cartpole_rewards[:100] * (episodes_to_compare // 100 + 1))[:episodes_to_compare],
         label="Random actions")
plt.title(f"Cumulative reward on CartPole (first {episodes_to_compare} episodes)")
plt.xlabel("Episode")
plt.ylabel("Cumulative reward")
plt.legend()
plt.grid(True)
save_and_show("p31_cumulative_rewards_comparison.png")
print("Total reward in the first", episodes_to_compare, "episodes:")
print("  DQN                       :", int(dqn_result["rewards"][:episodes_to_compare].sum()))
print("  Q-Learning (binned states):", int(tabular_result["rewards"][:episodes_to_compare].sum()))

# %% [markdown]
# ## Program 32: Visualize the learning curve of the RL agent

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
dqn_mean = moving_average(dqn_rewards, 20)
dqn_std = np.array([dqn_rewards[max(0, i - 19): i + 1].std() for i in range(len(dqn_rewards))])
axes[0].plot(dqn_mean, color="tab:red", label="Mean (20 episodes)")
axes[0].fill_between(range(len(dqn_mean)), dqn_mean - dqn_std, dqn_mean + dqn_std,
                     color="tab:red", alpha=0.2, label="+/- 1 std")
axes[0].set_title("Learning curve: DQN on CartPole")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Reward")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(moving_average(frozen_rewards, 200), color="tab:blue")
axes[1].set_title("Learning curve: Q-Learning on FrozenLake")
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Success rate")
axes[1].grid(True)
plt.tight_layout()
save_and_show("p32_learning_curves.png")

# %% [markdown]
# ## Program 33: Compare training time and convergence of Q-Learning and DQN

# %%
time_table = pd.DataFrame({
    "Q-Learning (FrozenLake)": [RESULTS["frozenlake"]["train_time_sec"], RESULTS["frozenlake"]["episodes"],
                                RESULTS["frozenlake"]["converge_episode"]],
    "Q-Learning (Grid World)": [RESULTS["gridworld"]["train_time_sec"], RESULTS["gridworld"]["episodes"],
                                RESULTS["gridworld"]["converge_episode"]],
    "Q-Learning (CartPole bins)": [RESULTS["tabular_cartpole"]["train_time_sec"],
                                   RESULTS["tabular_cartpole"]["episodes"],
                                   RESULTS["tabular_cartpole"]["converge_episode_195"]],
    "DQN (CartPole)": [RESULTS["dqn"]["train_time_sec"], RESULTS["dqn"]["episodes"],
                       RESULTS["dqn"]["converge_episode_195"]],
}, index=["Training time (s)", "Episodes trained", "Convergence episode"]).T
print(time_table)
print("\nConvergence episode: FrozenLake and Grid World = first episode where the smoothed")
print("reward reached 90% of its final value. CartPole = first episode where the average")
print("of 20 (DQN) or 100 (binned Q-Learning) episodes reached 195. 'None' = never reached.")

# %% [markdown]
# ## Program 34: Analyze the impact of hyperparameters on learning performance

# %%
print("Q-Learning hyperparameter studies (from Programs 12-14):")
print("\nLearning rate:\n", pd.DataFrame(RESULTS["learning_rate_study"]).to_string(index=False))
print("\nGamma:\n", pd.DataFrame(RESULTS["gamma_study"]).to_string(index=False))
print("\nEpsilon:\n", pd.DataFrame(RESULTS["epsilon_study"]).to_string(index=False))

print("\nDQN learning-rate study:")
dqn_lr_values = [1e-4, 1e-3, 1e-2]
dqn_lr_study, dqn_lr_curves = [], {}
for lr_value in dqn_lr_values:
    if lr_value == 1e-3:
        rewards_for_lr = with_replay_rewards                  # already trained above
    else:
        print(f"  training with learning rate {lr_value} ...")
        rewards_for_lr = train_dqn(ABLATION_EPISODES, learning_rate=lr_value)["rewards"]
    dqn_lr_curves[lr_value] = rewards_for_lr
    dqn_lr_study.append({"Learning rate": lr_value,
                         "Mean reward (last 20)": round(float(rewards_for_lr[-20:].mean()), 1),
                         "Best episode": float(rewards_for_lr.max())})
print(pd.DataFrame(dqn_lr_study).to_string(index=False))
RESULTS["dqn_learning_rate_study"] = dqn_lr_study

plt.figure(figsize=(9, 4))
for lr_value, curve in dqn_lr_curves.items():
    plt.plot(moving_average(curve, 20), label=f"learning rate = {lr_value}")
plt.title("DQN: effect of the learning rate")
plt.xlabel("Episode")
plt.ylabel("Reward (moving average of 20)")
plt.legend()
plt.grid(True)
save_and_show("p34_dqn_learning_rate_effect.png")

# %% [markdown]
# ## Program 35: Comparative report summarizing the RL algorithms

# %%
summary_table = pd.DataFrame([
    {"Algorithm": "Random actions", "Environment": "FrozenLake",
     "Performance": f"{RESULTS['frozenlake_random_success_rate']:.1%} success", "Training time (s)": "-"},
    {"Algorithm": "Q-Learning", "Environment": "FrozenLake",
     "Performance": f"{RESULTS['frozenlake']['eval_success_rate']:.1%} success",
     "Training time (s)": RESULTS["frozenlake"]["train_time_sec"]},
    {"Algorithm": "Q-Learning", "Environment": "Grid World",
     "Performance": f"{RESULTS['gridworld']['eval_success_rate']:.1%} success",
     "Training time (s)": RESULTS["gridworld"]["train_time_sec"]},
    {"Algorithm": "Random actions", "Environment": "CartPole",
     "Performance": f"{RESULTS['cartpole_random_mean']} avg reward", "Training time (s)": "-"},
    {"Algorithm": "Q-Learning (binned states)", "Environment": "CartPole",
     "Performance": f"{RESULTS['tabular_cartpole']['last_100_training_mean']} avg reward (last 100)",
     "Training time (s)": RESULTS["tabular_cartpole"]["train_time_sec"]},
    {"Algorithm": "DQN", "Environment": "CartPole",
     "Performance": f"{RESULTS['dqn']['eval_mean_reward']} avg reward (test)",
     "Training time (s)": RESULTS["dqn"]["train_time_sec"]},
])
print(summary_table.to_string(index=False))

report_lines = [
    "# Comparative Summary of Reinforcement Learning Algorithms", "",
    "```", summary_table.to_string(index=False), "```",
    "", "## Key points",
    "- Q-Learning stores a value for every (state, action) pair, so it works well for small,",
    "  discrete problems like FrozenLake and Grid World.",
    "- CartPole has continuous states. Q-Learning needs binning there, while DQN uses a",
    "  neural network that generalizes between similar states.",
    "- DQN needs more computing time per step and is less stable. Replay memory and a target",
    "  network are the two tricks that make it work.",
    "- Hyperparameters (learning rate, gamma, epsilon) change learning speed and final quality.",
]
try:
    (OUT_DIR / "comparative_summary.md").write_text("\n".join(report_lines), encoding="utf-8")
    with open(OUT_DIR / "results_summary.json", "w", encoding="utf-8") as results_file:
        json.dump(RESULTS, results_file, indent=2, default=str)
    print("\nSaved: outputs/comparative_summary.md and outputs/results_summary.json")
    print("Next step: run  python scripts/build_report.py  to build the Word report.")
except OSError as err:
    print("Could not save summary:", err)

# %% [markdown]
# ## Conclusion
# All 35 experiments of Lab Sheet-05 were completed: the Gymnasium environments
# were explored, Q-Learning was implemented and tuned on FrozenLake and a custom
# Grid World, a DQN was trained on CartPole with replay memory and a target
# network, and the algorithms were compared.
