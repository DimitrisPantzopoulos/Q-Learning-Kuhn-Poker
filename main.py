from kuhn_poker import TrainingKuhnPokerEnv, random_opponent_policy
from q_learning import QKuhnPoker, QLearningHyperparams

# Train and Evaluate on a random opponent
print("Q Learning Vs Random Opponent")

hyperparams  = QLearningHyperparams()
training_env = TrainingKuhnPokerEnv(random_opponent_policy)

model = QKuhnPoker(hyperparams)

model.train(training_env,    num_episodes=100_000, log_every=1_000, run_name='Train vs Random')
model.evaluate(training_env, num_hands=50_000)