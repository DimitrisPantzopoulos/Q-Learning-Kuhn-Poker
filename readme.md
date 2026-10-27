# Q-Learning for Kuhn Poker

**Three cards. Two players. One learning agent.**

A compact Python project that uses tabular Q-learning to learn a strategy for Kuhn Poker. Train against a random opponent, follow learning progress in TensorBoard, and evaluate the resulting policy over thousands of hands. This project does not use any Reinforcement Learning Libraries
to implement Tabular Q-Learning using only Numpy but the code was largely inspired from by my other Q-Learning project.

## Overview

- **Complete game environment:** dealing, legal actions, betting, folds, showdowns, and net rewards.
- **Tabular learning:** a small 12 × 4 Q-table with legal-action filtering.
- **Exploration and learning schedules:** decaying epsilon-greedy exploration and a visit-adjusted learning rate.
- **Both player positions:** the agent alternates seats during training and evaluation.
- **Training visibility:** progress bars and TensorBoard metrics for rewards, win rate, cumulative return, and exploration.
- **Custom opponents:** replace the random policy with your own Python function.

## Quick start

Requires **Python 3.10+**. Run the following commands from the project folder.

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

NumPy handles tables and random sampling, `tensorboardX` writes training logs, and `tqdm` displays progress. The `tensorboard` package provides the log viewer.

### 3. Train and evaluate

```bash
python main.py
```

The default experiment trains for **100,000 episodes**, logs every **1,000 episodes**, then evaluates the greedy policy over **50,000 hands** against the random opponent.

Evaluation prints total profit, mean profit per hand, and win/loss/draw counts. Rewards are net chip profits: **±1** for an uncalled bet or a checked showdown, and **±2** for a called bet. Mean profit per hand accounts for these different stakes; win rate alone does not.

### 4. Explore the training logs

```bash
tensorboard --logdir runs/KuhnPoker
```

Open the local URL printed by TensorBoard. Each training call creates a timestamped run under `runs/KuhnPoker/`.

| Metric | Meaning |
| --- | --- |
| `train/mean_reward` | Mean net profit per hand within the logging interval |
| `train/win_rate` | Fraction of hands with positive profit within the interval |
| `train/cumulative_return` | Total net profit since training started |
| `train/epsilon` | Current exploration probability |

## The game

Kuhn Poker is a small poker game with hidden information and room for bluffing:

1. Two players each ante one chip and receive one private card from a three-card deck: **Jack < Queen < King**. The third card stays unused.
2. Player 0 acts first and may check or bet one chip.
3. After a check, Player 1 may check or bet. Two checks lead to a showdown.
4. A player facing a bet may fold or call one chip. There are no raises.
5. A fold awards the pot to the other player; at showdown, the higher card wins.

Rewards are zero-sum and include the cost of each player's contributions to the pot.

## How learning works

The agent observes its own card and the public action history. The opponent's card stays hidden.

```text
state_id = card_id * 4 + history_id
```

Cards use IDs `0` (Jack), `1` (Queen), and `2` (King). The four decision histories are:

| History ID | Public history | Available actions |
| --- | --- | --- |
| `0` | Start of hand | Check, bet |
| `1` | Check | Check, bet |
| `2` | Bet | Fold, call |
| `3` | Check → bet | Fold, call |

The Q-table has 12 state rows and four action columns. Training and evaluation only select legal actions. Training uses epsilon-greedy exploration; evaluation selects a highest-valued legal action, breaking ties randomly.

The update follows:

```text
target = reward + gamma * max(Q[next_state, next_legal_actions])
Q[state, action] += adjusted_alpha * (target - Q[state, action])
```

For terminal transitions, the target is just the final reward. The learning rate decreases with visits to each state-action pair:

```text
adjusted_alpha = alpha / (1 + visits / 100)
```

## Configure an experiment

Edit `main.py`, or use the classes directly:

```python
from kuhn_poker import TrainingKuhnPokerEnv, random_opponent_policy
from q_learning import QKuhnPoker, QLearningHyperparams

hyperparams = QLearningHyperparams()
hyperparams.alpha = 0.01
hyperparams.gamma = 1.0
hyperparams.epsilon = 1.0
hyperparams.min_epsilon = 0.02
hyperparams.decay_rate = 0.9995

env = TrainingKuhnPokerEnv(random_opponent_policy, seed=42)
model = QKuhnPoker(hyperparams, seed=42)

model.train(env, num_episodes=100_000, log_every=1_000, run_name="experiment-")
model.evaluate(env, num_hands=50_000)
```

These hyperparameter values match the defaults. Epsilon decays after each training episode and stops at `min_epsilon`.

### Try a different opponent

An opponent policy receives the observation and legal action IDs, then returns one of those IDs:

```python
from kuhn_poker import Action, TrainingKuhnPokerEnv

def passive_opponent(obs: int, legal_actions: tuple[int, ...]) -> int:
    """Check when possible; otherwise call."""
    check_id = Action.CHECK.value - 1
    call_id = Action.CALL.value - 1
    return check_id if check_id in legal_actions else call_id

env = TrainingKuhnPokerEnv(passive_opponent, seed=42)
```

**Action IDs:** `0` = check, `1` = bet, `2` = fold, `3` = call. Policies and `TrainingKuhnPokerEnv.step()` use these zero-based IDs. The base `KuhnPokerEnv.step()` accepts `Action` members or their enum values (`1`–`4`).

## Project layout

```text
QLearningKuhnPoker/
├── main.py          # Default training and evaluation experiment
├── kuhn_poker.py    # Game rules, training wrapper, and random opponent
├── q_learning.py    # Q-table, learning schedules, training, and evaluation
├── requirements.txt # Python dependencies and TensorBoard viewer
├── readme.md
└── runs/
    └── KuhnPoker/   # Generated TensorBoard event logs
```

## Experiment notes

- **Opponent-specific learning:** the default experiment measures performance against a random policy. It does not establish a Nash equilibrium or measure exploitability.
- **Model storage:** the learned Q-table lives in memory as `model.decision_q_table`. Training writes metrics to disk; it does not save model checkpoints.
- **Random seeds:** the environment and agent default to seed `42`. The built-in random opponent uses a separate module-level generator seeded with `123`; its state advances across calls within the same process.
- **Configuration:** change experiment settings in Python. The project currently has no command-line argument interface.
