

## About this project

This is my work for Lab Sheet-05 (Lab Assessment on Reinforcement Learning).
It has 35 programs. An agent learns by trying actions and getting rewards.
I used Q-Learning on FrozenLake and on my own Grid World, and a Deep Q-Network
(DQN) on CartPole.

## What is in the folder

| File / Folder | What it does |
|---|---|

| `scripts/build_report.py` | Builds the Word report . |
| `report/` | The report is saved |
| `models/` | The saved Q-table and DQN model |
| `outputs/` | The graphs and the results summary  |


## Environments (no dataset is needed)

| Environment | Details |
|---|---|
| FrozenLake-v1 (4x4, slippery) | 16 states, 4 actions, reward 1 at the goal |
| Grid World (custom, 5x5) | 25 states, 4 actions, -1 per step, +20 goal, -10 pit |
| CartPole-v1 | 4 continuous numbers as state, 2 actions, reward 1 per step |




- RL training uses random numbers, so your results can be a little different
  from run to run. DQN especially can go up and down during training.
- If DQN does not reach a good score, increase `DQN_EPISODES` (for example to 500).


