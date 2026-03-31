from collections import defaultdict

import numpy as np
import gymnasium as gym

import os
from Overcooked_mdp_gym import OvercookedEnv

class OvercookedAgent:
    def __init__(self, env: gym.Env, initial_epsilon: float, decay_rate: float, final_epsilon: float, discount_factor: float = 0.99):
        """Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        self.env = env
        
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        # Track number of updates for each (state, action)
        self.update_counts = defaultdict(lambda: np.zeros(env.action_space.n))
        
        # Learning factors
        self.discount_factor = discount_factor  # Discount factor
        
        # Exploration factors
        self.epsilon = initial_epsilon  # Exploration rate
        self.epsilon_decay = decay_rate
        self.final_epsilon = final_epsilon
        
    def select_action(self, state):
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (stand) or 1 (hit)
        """
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_table[state]))

    def update(self, state, action: int, reward: float, terminated:bool, next_state):
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
        # Q-learning update rule (off-policy)
        
        future_q_value = 0 if terminated else np.max(self.q_table[next_state])
        
        # Weight for weighted average
        weight = 1.0 / (1 + self.update_counts[state][action])
        td_target = reward + self.discount_factor * future_q_value
        # Convex combination update
        old_w_avg = (1 - weight) * self.q_table[state][action]
        new_w_value = weight * td_target
        
        # Set q-value for the current state, action to new weighted average
        self.q_table[state][action] = old_w_avg + new_w_value
        
        # Increment update count for this (state, action)
        self.update_counts[state][action] += 1
        
    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, filename):
        """Merges current Q-values into an existing save file, or creates one if none exists."""
        if os.path.exists(filename):
            saved = np.load(filename, allow_pickle=True).item()
            saved.update(dict(self.q_table))
        else:
            saved = dict(self.q_table)
        np.save(filename, saved)

    def load(self, filename):
        """Loads a previously saved Q-table from a file."""
        saved = np.load(filename, allow_pickle=True).item()
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n), saved)


def run_training(
    n_episodes: int = 1000,
    save_file: str = "q_table.npy",
    save_every: int = 50,
    render: bool = False,
):
    """Train the Q-learning agent on Overcooked.

    Args:
        n_episodes:  Number of episodes to run.
        save_file:   Path to persist the Q-table (merged each save_every episodes).
        save_every:  How often (in episodes) to checkpoint the Q-table.
        render:      Whether to display the live game window during training.
    """

    render_mode = "human" if render else None
    env = OvercookedEnv(render_mode=render_mode)

    agent = OvercookedAgent(
        env=env,
        initial_epsilon=1.0,
        decay_rate=0.999,
        final_epsilon=0.1,
        discount_factor=0.99,
    )

    # Resume from checkpoint if one exists
    if os.path.exists(save_file):
        agent.load(save_file)
        print(f"Loaded existing Q-table from {save_file}")

    episode_rewards = []

    for episode in range(1, n_episodes + 1):
        obs, _ = env.reset()
        state = tuple(obs)
        total_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = tuple(next_obs)

            agent.update(state, action, reward, terminated, next_state)

            state = next_state
            total_reward += reward

        # Decay epsilon after each episode
        agent.epsilon = max(
            agent.final_epsilon,
            agent.epsilon - agent.epsilon_decay
        )

        episode_rewards.append(total_reward)

        print(f"Episode {episode:>5}/{n_episodes}  reward={total_reward:>8.2f}  epsilon={agent.epsilon:.3f}")

        if episode % save_every == 0:
            agent.save(save_file)
            print(f"  -> Q-table saved to {save_file}")

    # Final save
    agent.save(save_file)
    env.close()
    print(f"\nTraining complete. Average reward: {sum(episode_rewards)/len(episode_rewards):.2f}")


if __name__ == "__main__":
    # In order to run training, make sure that overcooked is open and visable on screen, then run
    # python Overcooked_Q_Agent.py
    
    n_episodes: int = 1000,
    save_file: str = "q_table.npy",
    save_every: int = 50,
    render: bool = False,
    
    run_training(n_episodes=n_episodes, save_file=save_file, save_every=save_every, render=render)