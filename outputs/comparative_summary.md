# Comparative Summary of Reinforcement Learning Algorithms

```
                 Algorithm Environment                  Performance Training time (s)
            Random actions  FrozenLake                 2.0% success                 -
                Q-Learning  FrozenLake                73.0% success             18.16
                Q-Learning  Grid World               100.0% success              0.96
            Random actions    CartPole             23.03 avg reward                 -
Q-Learning (binned states)    CartPole 261.35 avg reward (last 100)              96.2
                       DQN    CartPole     198.55 avg reward (test)            259.04
```

## Key points
- Q-Learning stores a value for every (state, action) pair, so it works well for small,
  discrete problems like FrozenLake and Grid World.
- CartPole has continuous states. Q-Learning needs binning there, while DQN uses a
  neural network that generalizes between similar states.
- DQN needs more computing time per step and is less stable. Replay memory and a target
  network are the two tricks that make it work.
- Hyperparameters (learning rate, gamma, epsilon) change learning speed and final quality.