from kuhn_poker import TrainingKuhnPokerEnv, MAX_STATES, NO_ACTIONS
from tensorboardX import SummaryWriter
from datetime import datetime
from typing import Tuple
from tqdm import tqdm

import numpy as np

class QLearningHyperparams:
    epsilon     : float = 1.0
    min_epsilon : float = 0.02
    decay_rate  : float = 0.9995

    gamma : float = 1.0
    alpha : float = 0.01

    def decay_epsilon(self) -> None:
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.decay_rate,
        )

    def visit_alpha(self, visit_counts : np.ndarray, state : int, action : int) -> float:
        visit_counts[state, action] += 1

        visits : int = visit_counts[state, action]

        return self.alpha / (1.0 + visits / 100.0)

class QKuhnPoker:
    def __init__(self, hyperparams : QLearningHyperparams, seed : int=42) -> None:
        self.rng : np.random.Generator = np.random.default_rng(seed)

        # Hyperparameters
        self.hyperparams : QLearningHyperparams = hyperparams

        # Q Tables
        self.decision_q_table : np.ndarray = self._init_q_table((MAX_STATES, NO_ACTIONS))

        # Visit Tables
        self.visit_counts : np.ndarray = np.zeros_like(self.decision_q_table, dtype=np.int64)

    def _init_q_table(self, shape : Tuple[int, ...]) -> np.ndarray:
        return np.zeros(shape, dtype=np.float32)

    def _max_q_value(self, q_table : np.ndarray, state : int | None, next_legal_actions : Tuple[int, ...]) -> np.float64:
        return np.max(q_table[state, next_legal_actions])

    def _greedy_action(self, q_table : np.ndarray, state : int, legal_moves : Tuple[int, ...]) -> int:
        values : np.ndarray = q_table[state, list(legal_moves)]

        best_positions : np.ndarray = np.flatnonzero(values == values.max())
        position       : int        = int(self.rng.choice(best_positions))

        return legal_moves[position]
    
    def _sample_random_action(self, legal_moves : Tuple[int, ...]) -> int:
        return self.rng.choice(legal_moves)

    def _should_explore(self) -> bool:
        return True if self.rng.random() < self.hyperparams.epsilon else False

    def _epsilon_greedy_action(self, q_table : np.ndarray, state : int, legal_moves : Tuple[int, ...]) -> int:
        # Exploration
        if self._should_explore():
            return self._sample_random_action(legal_moves)

        # Exploitation
        return self._greedy_action(q_table, state, legal_moves)

    def _td_target(self, q_table : np.ndarray, next_state : int | None, next_legal_actions : Tuple[int, ...], reward : float, done : bool) -> float:
        future_return : float = (
            self.hyperparams.gamma * self._max_q_value(q_table, next_state, next_legal_actions) if not done else 0.0
        )

        return reward + future_return

    def _td_error(self, q_table : np.ndarray, target : float, state : int, action : int) -> float:
        return target - q_table[state, action]

    def _q_learning_update(
            self, 
            q_table    : np.ndarray, 
            state      : int,
            action     : int, 
            reward     : float, 
            next_state : int | None, 
            done       : bool,
            next_legal_actions : Tuple[int, ...],
        ) -> None:
        target : float = self._td_target(
            q_table, next_state, next_legal_actions, reward, done
        )

        adjusted_alpha : float = self.hyperparams.visit_alpha(self.visit_counts, state, action)

        q_table[state, action] += (
            adjusted_alpha * (target - q_table[state, action])
        )

    def _interaction_step(
            self,
            env     : TrainingKuhnPokerEnv,
            q_table : np.ndarray,
            state   : int,
        ) -> Tuple[int | None, float, bool]:

        action_id : int = self._epsilon_greedy_action(
            q_table,
            state,
            env.legal_actions()
        )

        next_state, reward, done = env.step(action_id)

        self._q_learning_update(
            q_table, state, action_id, reward, next_state, done, env.legal_actions()
        )

        return next_state, reward, done   

    def _run_training_episode(self, env : TrainingKuhnPokerEnv) -> float:
        obs : int = env.reset()

        hand_profit : float = 0.0

        done : bool = env.terminated

        while not done:
            obs, reward, done = self._interaction_step(env, self.decision_q_table, obs) # type: ignore

            hand_profit += reward
        return hand_profit

    def _training_phase(self, env : TrainingKuhnPokerEnv, num_episodes : int, log_every : int, writer : SummaryWriter) -> None:
        # Training Statistics
        total_return  : float       = 0.0
        block_returns : list[float] = []

        for episode in tqdm(range(1, num_episodes + 1), unit='episodes', desc=f'Training...'):
            env.switch_agent_id()

            total_reward : float = self._run_training_episode(env)

            self.hyperparams.decay_epsilon()

            block_returns.append(total_reward)
            total_return += total_reward

            if episode % log_every == 0 or episode == num_episodes:
                returns : np.ndarray = np.asarray(block_returns)

                writer.add_scalar(f"train/mean_reward",       float(returns.mean()), episode)
                writer.add_scalar(f"train/win_rate",          float(np.mean(returns > 0)), episode)
                writer.add_scalar(f"train/cumulative_return", total_return, episode)
                writer.add_scalar(f"train/epsilon",           self.hyperparams.epsilon, episode)

                block_returns.clear()
                writer.flush()

    def train(self, env : TrainingKuhnPokerEnv, num_episodes : int=100_000, log_every : int=1_00, run_name : str="") -> None:
        if log_every <= 0:
            raise ValueError("log_every must be positive")
        if num_episodes <= 0:
            raise ValueError("Both training phases must have positive episode counts")

        date_time : str           = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        writer    : SummaryWriter = SummaryWriter(f"runs/KuhnPoker/{run_name + date_time}")

        try:
            self._training_phase(env, num_episodes, log_every, writer)
        finally:
            writer.close()

    def evaluate(
        self,
        env       : TrainingKuhnPokerEnv,
        num_hands : int = 200_000,
    ) -> None:
        if num_hands <= 0:
            raise ValueError("num_hands must be positive")

        profits : list[float] = []

        for _ in tqdm(range(num_hands), unit="hands", desc="Evaluating..."):
            env.switch_agent_id()

            state  : int = env.reset()
            done   : bool = env.terminated
            profit : float = 0.0

            while not done:
                action : int = self._greedy_action(
                    self.decision_q_table,
                    state,
                    env.legal_actions(),
                )

                state, reward, done = env.step(action) # type: ignore
                profit += reward
            profits.append(profit)
        results = np.asarray(profits, dtype=np.float64)

        wins   = int(np.count_nonzero(results > 0))
        losses = int(np.count_nonzero(results < 0))
        draws  = int(np.count_nonzero(results == 0))

        print(f"\nHands evaluated:      {num_hands:,}")
        print(f"Total profit:         {results.sum():+.2f}")
        print(f"Mean profit per hand: {results.mean():+.5f}")
        print(f"Wins:                 {wins:,} ({wins / num_hands:.2%})")
        print(f"Losses:               {losses:,} ({losses / num_hands:.2%})")
        print(f"Draws:                {draws:,} ({draws / num_hands:.2%})")



