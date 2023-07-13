import gymnasium as gym
import wandb
import random
from agent import RandomAgent, SARSAAgent, QLearningAgent

# enable/disable wandb with the environment variable WANDB_DISABLED
import os
os.environ["WANDB_DISABLED"] = "false"


batch_size = 50


episodes = 5000
steps = 1000
learning_rate = 0.1
alpha = 0.05
gamma = 0.9
epsilon = 0.1

wandb.init(
    project="cliff_walking_rl",
    # track hyperparameters and run metadata
    config={
    "learning_rate": alpha,
    "architecture": "Q-Learning",
    "episodes": episodes,
    "steps": steps,
    "env": "PointMaze_UMazeDense-v3",
    "alpha": alpha,
    "gamma": gamma,
    }
)

env = gym.make('CliffWalking-v0', render_mode='rgb_array')
env.reset()

agent = QLearningAgent(env, alpha=alpha, gamma=gamma, epsilon=epsilon)
try:
    agent.load("./q_table.npy")
    print(f"Loaded q_table: {agent.Q}")
    print(f"{agent.Q.sum()}")
except:
    print("No q_table found, starting from scratch")
    pass

for episode in range(episodes):
    for step in range(steps):
        done = agent.step()
        if done:
            break
    if episode % batch_size == 0:
        agent.save(f"q_table.npy")
    print(f"final_reward: {agent.total_reward}")
    wandb.log({"total_reward": agent.total_reward, "episode": episode, "epsilon": agent.epsilon, "Q table": agent.Q})
    agent.save("q_table.npy")
    agent.reset()
# [optional] finish the wandb run, necessary in notebooks
wandb.finish()