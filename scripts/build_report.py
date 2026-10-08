"""Build the Lab Sheet-05 report (Word file) from YOUR OWN run.

Steps:
    1. Run the programs first:   python lab_sheet_05_all_programs.py
       (this creates outputs/results_summary.json and the graphs)
    2. Build the report:         python scripts/build_report.py
       -> report/Lab_Sheet_05_Report.docx

The report follows the format asked in the lab sheet:
    Objective, Algorithm/Procedure, Source Code, Output,
    Performance Analysis, Observation, Conclusion

What is filled in automatically: objective, procedure, source code, your graphs,
and the tables with the numbers from your run.
What YOU must write: the Observation and Conclusion sections (they are marked
in the file). Open the .docx, replace the yellow notes with your own words,
and add your name, roll number and screenshots.

Needs:  pip install python-docx
"""
import json
from pathlib import Path

try:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt, RGBColor
except ImportError:
    raise SystemExit("python-docx is not installed. Run:  pip install python-docx")

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "outputs"
REPORT_DIR = BASE_DIR / "report"
SOURCE_FILE = BASE_DIR / "lab_sheet_05_all_programs.py"
RESULTS_FILE = OUT_DIR / "results_summary.json"
REPORT_FILE = REPORT_DIR / "Lab_Sheet_05_Report.docx"


def load_results():
    """Read the numbers saved by Program 35."""
    if not RESULTS_FILE.exists():
        raise SystemExit("outputs/results_summary.json not found.\n"
                         "Run  python lab_sheet_05_all_programs.py  first.")
    with open(RESULTS_FILE, encoding="utf-8") as file:
        return json.load(file)


def add_table(document, header, rows):
    """Add a simple table with a header row."""
    table = document.add_table(rows=1, cols=len(header))
    table.style = "Table Grid"
    for cell, text in zip(table.rows[0].cells, header):
        cell.text = ""
        run = cell.paragraphs[0].add_run(str(text))
        run.bold = True
    for row in rows:
        cells = table.add_row().cells
        for cell, text in zip(cells, row):
            cell.text = str(text)
    document.add_paragraph()


def add_numbered(document, steps):
    """Add a numbered list that always starts at 1."""
    for number, text in enumerate(steps, start=1):
        paragraph = document.add_paragraph(f"{number}. {text}")
        paragraph.paragraph_format.left_indent = Inches(0.3)
        paragraph.paragraph_format.first_line_indent = Inches(-0.25)


def add_note(document, text):
    """A highlighted note telling the student what to write."""
    paragraph = document.add_paragraph()
    run = paragraph.add_run(text)
    run.italic = True
    run.font.color.rgb = RGBColor(0xB0, 0x40, 0x00)
    run.font.highlight_color = 7  # yellow


def add_figure(document, file_name, caption):
    """Add one of your saved graphs, if it exists."""
    path = OUT_DIR / file_name
    if path.exists():
        document.add_picture(str(path), width=Inches(5.8))
        document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption_paragraph = document.add_paragraph(caption)
        caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption_paragraph.runs[0].italic = True


def build():
    results = load_results()
    document = Document()
    document.styles["Normal"].font.name = "Calibri"
    document.styles["Normal"].font.size = Pt(11)

    # ---------------- Title page ----------------
    title = document.add_heading("Lab Sheet-05: Reinforcement Learning", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for line in ["MCA III Semester (Session 2026-2027)", "COER University, Roorkee, Uttarakhand",
                 "Name: ____________________", "Roll No: ____________________"]:
        paragraph = document.add_paragraph(line)
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ---------------- 1. Objective ----------------
    document.add_heading("1. Objective", level=1)
    for text in [
        "To understand the basic ideas of Reinforcement Learning (RL): agent, environment, "
        "state, action and reward.",
        "To implement Q-Learning on the FrozenLake environment and on a custom Grid World.",
        "To implement a Deep Q-Network (DQN) with PyTorch and train it on CartPole.",
        "To study the effect of hyperparameters, replay memory and the target network.",
        "To compare the performance of the algorithms.",
    ]:
        document.add_paragraph(text, style="List Bullet")

    # ---------------- 2. Algorithm / Procedure ----------------
    document.add_heading("2. Algorithm / Procedure", level=1)
    document.add_heading("2.1 Q-Learning", level=2)
    add_numbered(document, [
        "Create the environment and a Q-table of zeros (one row per state, one column per action).",
        "For each episode, reset the environment and choose actions with the epsilon-greedy policy "
        "(random action with probability epsilon, otherwise the best known action).",
        "After each step, update: Q(s,a) = Q(s,a) + alpha * [ r + gamma * max Q(s',a') - Q(s,a) ].",
        "Reduce epsilon slowly so the agent explores first and exploits later.",
        "After training, test the greedy policy (epsilon = 0) and measure the success rate.",
    ])
    document.add_heading("2.2 Deep Q-Network (DQN)", level=2)
    add_numbered(document, [
        "Build a neural network that takes the CartPole state (4 numbers) and gives a Q-value for each action.",
        "Store every transition (state, action, reward, next state, done) in the replay memory.",
        "At each step, sample a random mini-batch from the memory and compute the loss between the "
        "predicted Q-value and the target r + gamma * max Q_target(s', a').",
        "Update the network with the Adam optimizer. Copy its weights to the target network every 250 steps.",
        "Repeat for many episodes, then test the trained network, save it with torch.save and load it again.",
    ])
    document.add_heading("2.3 Environments and libraries", level=2)
    add_table(document, ["Environment", "Type", "Used for"], [
        ["FrozenLake-v1 (4x4, slippery)", "Discrete states (16), 4 actions", "Programs 2-15"],
        ["Custom Grid World (5x5)", "Discrete states (25), 4 actions", "Programs 16-20"],
        ["CartPole-v1", "Continuous states (4 numbers), 2 actions", "Programs 21-34"],
    ])
    document.add_paragraph("Libraries: Python 3.11+, NumPy, Pandas, Matplotlib, Gymnasium, PyTorch.")

    # ---------------- 3. Source Code ----------------
    document.add_heading("3. Source Code", level=1)
    document.add_paragraph("The complete code of all 35 programs (file lab_sheet_05_all_programs.py):")
    code_text = SOURCE_FILE.read_text(encoding="utf-8")
    for line in code_text.splitlines():
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(0)
        run = paragraph.add_run(line if line else " ")
        run.font.name = "Consolas"
        run.font.size = Pt(7.5)

    # ---------------- 4. Output ----------------
    document.add_page_break()
    document.add_heading("4. Output", level=1)
    document.add_paragraph("Graphs produced by the programs in my run:")
    figures = [
        ("p11_frozenlake_cumulative_rewards.png", "Program 11: Cumulative reward and success rate (FrozenLake)"),
        ("p12_learning_rate_effect.png", "Program 12: Effect of the learning rate"),
        ("p13_gamma_effect.png", "Program 13: Effect of the discount factor"),
        ("p14_epsilon_effect.png", "Program 14: Exploration vs exploitation"),
        ("p18_gridworld_optimal_path.png", "Program 18: Optimal path in Grid World"),
        ("p20_convergence.png", "Program 20: Convergence of Q-Learning"),
        ("p24_dqn_training_rewards.png", "Program 24: DQN rewards during training"),
        ("p27_replay_memory_effect.png", "Program 27: Effect of replay memory"),
        ("p28_target_network_effect.png", "Program 28: Effect of the target network"),
        ("p31_cumulative_rewards_comparison.png", "Program 31: Cumulative reward comparison"),
        ("p32_learning_curves.png", "Program 32: Learning curves"),
        ("p34_dqn_learning_rate_effect.png", "Program 34: DQN learning rate study"),
    ]
    for file_name, caption in figures:
        add_figure(document, file_name, caption)
    add_note(document, "ADD YOUR SCREENSHOTS HERE: the Q-table (Program 9), the terminal or notebook "
                       "output of the main programs, and a screenshot of the code running.")

    # ---------------- 5. Performance Analysis ----------------
    document.add_heading("5. Performance Analysis", level=1)
    document.add_paragraph("All numbers below come from my run of the programs.")

    frozen, grid = results.get("frozenlake", {}), results.get("gridworld", {})
    dqn, tab = results.get("dqn", {}), results.get("tabular_cartpole", {})

    document.add_heading("5.1 Q-Learning", level=2)
    add_table(document, ["Environment", "Episodes", "Success rate (greedy)", "Avg steps",
                         "Training time (s)", "Converged at episode"], [
        ["FrozenLake", frozen.get("episodes"), f"{frozen.get('eval_success_rate', 0):.1%}",
         frozen.get("eval_avg_steps"), frozen.get("train_time_sec"), frozen.get("converge_episode")],
        ["Grid World", grid.get("episodes"), f"{grid.get('eval_success_rate', 0):.1%}",
         grid.get("eval_avg_steps"), grid.get("train_time_sec"), grid.get("converge_episode")],
    ])
    document.add_paragraph(f"A random agent reached the FrozenLake goal in "
                           f"{results.get('frozenlake_random_success_rate', 0):.1%} of the episodes.")

    document.add_heading("5.2 Hyperparameter studies (FrozenLake)", level=2)
    document.add_paragraph("Learning rate (alpha):")
    add_table(document, ["Learning rate", "Training success rate", "Greedy success rate"],
              [[r["Learning rate"], r["Training success rate"], r["Greedy success rate"]]
               for r in results.get("learning_rate_study", [])])
    document.add_paragraph("Discount factor (gamma):")
    add_table(document, ["Gamma", "Training success rate", "Greedy success rate"],
              [[r["Gamma"], r["Training success rate"], r["Greedy success rate"]]
               for r in results.get("gamma_study", [])])
    document.add_paragraph("Epsilon:")
    add_table(document, ["Setting", "Training success rate", "Greedy success rate"],
              [[r["Setting"], r["Training success rate"], r["Greedy success rate"]]
               for r in results.get("epsilon_study", [])])

    document.add_heading("5.3 CartPole: Random vs Q-Learning vs DQN", level=2)
    add_table(document, ["Method", "Episodes", "Average reward", "Training time (s)"], [
        ["Random actions", "-", results.get("cartpole_random_mean"), "-"],
        ["Q-Learning (binned states)", tab.get("episodes"), tab.get("last_100_training_mean"),
         tab.get("train_time_sec")],
        ["DQN", dqn.get("episodes"), dqn.get("eval_mean_reward"), dqn.get("train_time_sec")],
    ])

    document.add_heading("5.4 DQN studies", level=2)
    replay, target = results.get("replay_effect", {}), results.get("target_effect", {})
    add_table(document, ["Experiment", "Mean reward (last 20 episodes)"], [
        ["With replay memory", replay.get("with_replay_last20")],
        ["Without replay memory", replay.get("without_replay_last20")],
        ["With target network", target.get("with_target_last20")],
        ["Without target network", target.get("without_target_last20")],
    ])
    add_table(document, ["DQN learning rate", "Mean reward (last 20)", "Best episode"],
              [[r["Learning rate"], r["Mean reward (last 20)"], r["Best episode"]]
               for r in results.get("dqn_learning_rate_study", [])])

    # ---------------- 6. Observation ----------------
    document.add_heading("6. Observation", level=1)
    add_note(document, "WRITE THIS SECTION YOURSELF, in your own words, using your graphs and tables above. "
                       "Delete this note when you are done. Questions to answer: "
                       "(1) How much better was the trained Q-Learning agent than the random agent? "
                       "(2) What happened with a very small and a very large learning rate? "
                       "(3) Why did a low gamma or epsilon = 0 hurt learning? "
                       "(4) Did DQN reach the 'solved' level of 195? How did training look: smooth or noisy? "
                       "(5) What changed without replay memory and without the target network?")
    for _ in range(4):
        document.add_paragraph("_" * 90)

    # ---------------- 7. Conclusion ----------------
    document.add_heading("7. Conclusion", level=1)
    add_note(document, "WRITE 3-4 LINES YOURSELF: what you learned about Q-Learning and DQN, which "
                       "algorithm suits which kind of problem, and which hyperparameters mattered most. "
                       "Delete this note when you are done.")
    for _ in range(3):
        document.add_paragraph("_" * 90)

    REPORT_DIR.mkdir(exist_ok=True)
    document.save(REPORT_FILE)
    print("Report saved:", REPORT_FILE)
    print("Now open it in Word, add your name, screenshots, and write the Observation and Conclusion.")


if __name__ == "__main__":
    build()
