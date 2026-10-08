

## About this project

This is my work for Lab Sheet-05 (Lab Assessment on Reinforcement Learning).
It has 35 programs. An agent learns by trying actions and getting rewards.
I used Q-Learning on FrozenLake and on my own Grid World, and a Deep Q-Network
(DQN) on CartPole.

## What is in the folder

| File / Folder | What it does |
|---|---|
| `lab_sheet_05_all_programs.py` | All 35 programs in one file. Each program is its own cell. |

| `scripts/build_report.py` | Builds the Word report . |
| `report/` | The report is saved here as `Lab_Sheet_05_Report.docx`. |
| `models/` | The saved Q-table and DQN model (made by Programs 9 and 29). |
| `outputs/` | The graphs and the results summary (made when the programs run). |
| `requirements.txt` | The list of libraries to install. |

## Environments (no dataset is needed)

| Environment | Details |
|---|---|
| FrozenLake-v1 (4x4, slippery) | 16 states, 4 actions, reward 1 at the goal |
| Grid World (custom, 5x5) | 25 states, 4 actions, -1 per step, +20 goal, -10 pit |
| CartPole-v1 | 4 continuous numbers as state, 2 actions, reward 1 per step |

## Libraries used

NumPy, Pandas, Matplotlib, Gymnasium, PyTorch, python-docx (for the report).
Python version: 3.11 or above.

## How to run it

1. Open the project folder in VS Code.
2. Open the terminal and make a virtual environment:

   ```
   python -m venv venv
   venv\Scripts\activate
   ```

   (On Linux or Mac, use `source venv/bin/activate`.)

3. Install the libraries:

   ```
   pip install -r requirements.txt
   ```

   If `torch` is slow or fails, get the CPU version from pytorch.org.

4. Run all programs:

   ```
   python lab_sheet_05_all_programs.py
   ```

   It takes about 10-15 minutes because of the DQN training. For a quick test,
   set `QUICK_MODE = True` near the top of the file.
   Or open the file in VS Code and click **Run Cell** above any program.

5. Build the report:

   ```
   python scripts/build_report.py
   ```

   Then open `report/Lab_Sheet_05_Report.docx`, add your name and screenshots,
   and write the Observation and Conclusion in your own words.

## What the programs cover

- **Programs 1-5:** Gymnasium setup, spaces, states, actions, rewards, random agent.
- **Programs 6-15:** Q-Learning on FrozenLake, Q-table, evaluation, learning rate,
  gamma, epsilon, and the epsilon-greedy policy.
- **Programs 16-20:** My own Grid World, training, the learned path, comparison
  with FrozenLake, and convergence.
- **Programs 21-30:** DQN with PyTorch on CartPole, training, evaluation,
  replay memory, target network, saving and loading the model.
- **Programs 31-35:** Comparison of the algorithms, learning curves, training time,
  hyperparameters, and the summary.

## Notes

- RL training uses random numbers, so your results can be a little different
  from run to run. DQN especially can go up and down during training.
- If DQN does not reach a good score, increase `DQN_EPISODES` (for example to 500).


